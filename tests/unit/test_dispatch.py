"""Dispatch admission, safe validation, and ordered effects."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillrunner.domain.errors import RunnerError
from skillrunner.model.protocol import ModelToolCall
from skillrunner.runtime.budgets import Deadline, UsageLedger
from skillrunner.tools import dispatch, schemas


def call(name="write", arguments=None, ident="1"):
    return ModelToolCall(ident, name, arguments or {"path": "generated/a", "content": "hi"}, "{}")


def ledger(count=10):
    return UsageLedger(max_steps=10, max_tokens=10000, max_tool_calls=count)


async def test_dot_listing_returns_recoverable_root_error_then_named_root_succeeds(tmp_path):
    from skillrunner.tools.files import FileTools

    files = FileTools()
    files.register_root("scratch", tmp_path)
    (tmp_path / "answer.txt").write_text("answer")
    registry = dispatch.ToolRegistry()

    async def listing(args):
        return files.list_files(args.path, offset=args.offset, limit=args.limit)

    registry.register("list_files", "List workspace files", schemas.ListFilesArgs, listing)
    results = await registry.dispatch_batch(
        [call("list_files", {"path": "."}), call("list_files", {"path": "scratch"}, "2")],
        ledger(),
        Deadline(10),
    )
    assert results[0].error["code"] == "file_access_denied"
    assert results[1].ok
    assert results[1].value["entries"][0]["path"] == "scratch/answer.txt"


async def test_order_and_attempts_include_unknown_validation_and_denials():
    seen = []
    registry = dispatch.ToolRegistry()

    async def write(args):
        seen.append(args.path)
        if args.path == "denied":
            raise RunnerError("file_access_denied", "Use an approved root.")
        return {"written": args.path}

    registry.register("write", "Write", schemas.WriteFileArgs, write)
    usage = ledger()
    results = await registry.dispatch_batch(
        [
            call("missing"),
            call(arguments={"path": 42, "content": "SECRET"}),
            call(arguments={"path": "denied", "content": ""}),
            call(),
        ],
        usage,
        Deadline(10),
    )
    assert usage.charged_tool_calls == 4
    assert seen == ["denied", "generated/a"]
    assert [r.ok for r in results] == [False, False, False, True]
    assert [r.executed for r in results] == [False, False, False, True]
    assert "SECRET" not in json.dumps([r.as_dict() for r in results])
    assert registry.model_schemas()[0]["function"]["parameters"]["additionalProperties"] is False


async def test_oversized_and_mixed_batches_have_no_effects():
    registry = dispatch.ToolRegistry()
    seen = []

    async def handler(args):
        seen.append(args)

    for name, kind in [("write", "action"), ("activate", "activation"), ("finish", "completion")]:
        registry.register(name, name, schemas.WriteFileArgs, handler, kind=kind)
    usage = ledger(1)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await registry.dispatch_batch([call(), call()], usage, Deadline(10))
    assert usage.requested_tool_calls == 2 and usage.charged_tool_calls == 0
    for first in ["activate", "finish"]:
        mixed_usage = ledger()
        results = await registry.dispatch_batch([call(first), call()], mixed_usage, Deadline(10))
        assert all(r.error["code"] == "model_protocol_error" for r in results)
        assert mixed_usage.charged_tool_calls == 2
    assert not seen


@pytest.mark.parametrize(
    "code",
    [
        "budget_exhausted",
        "context_capacity_exceeded",
        "cancelled",
        "terminated",
        "external_outcome_unknown",
    ],
)
async def test_terminal_failure_preserves_previous_result_and_releases_slots(code):
    registry = dispatch.ToolRegistry()
    observed = []

    async def handler(args):
        if args.path == "stop":
            raise RunnerError(code, "Stop.")
        return "ok"

    async def observe(result):
        observed.append(result)

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    usage = ledger()
    with pytest.raises(RunnerError, match=code):
        await registry.dispatch_batch(
            [call(), call(arguments={"path": "stop", "content": ""}), call()],
            usage,
            Deadline(10),
            on_result=observe,
        )
    assert len(observed) == 2
    assert observed == registry.last_results
    assert observed[0].ok and not observed[1].ok
    assert usage.charged_tool_calls == 2 and usage.available_tool_calls == 8


async def test_safe_normalization_and_bounding_except_activation():
    @dataclass
    class Record:
        path: Path
        text: str

    registry = dispatch.ToolRegistry(max_result_bytes=256)

    async def handler(args):
        return Record(Path("a"), "é" * 10000)

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    registry.register("activate", "Activate", schemas.WriteFileArgs, handler, kind="activation")
    result = (await registry.dispatch_batch([call()], ledger(), Deadline(10)))[0]
    assert result.value["truncated"] is True
    assert len(json.dumps(result.value, ensure_ascii=True).encode()) <= 256
    result = (await registry.dispatch_batch([call("activate")], ledger(), Deadline(10)))[0]
    assert result.value == {"path": "a", "text": "é" * 10000}


async def test_unknown_exception_does_not_disclose_details():
    registry = dispatch.ToolRegistry()

    async def handler(args):
        raise RuntimeError("SECRET")

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    result = (await registry.dispatch_batch([call()], ledger(), Deadline(10)))[0]
    assert result.error["code"] == "internal_error"
    assert "SECRET" not in json.dumps(result.as_dict())


async def test_pending_cancellation_checked_before_handler():
    registry = dispatch.ToolRegistry()
    seen = []

    async def handler(args):
        seen.append(True)

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    task = asyncio.create_task(registry.dispatch_batch([call()], ledger(), Deadline(10)))
    asyncio.get_running_loop().call_soon(task.cancel)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not seen


@pytest.mark.parametrize(
    "model,value",
    [
        ("ReadTextArgs", {"path": "a", "offset": True}),
        ("ListFilesArgs", {"path": "a", "limit": 0}),
        ("WriteFileArgs", {"path": "a", "content": "", "overwrite": "false"}),
        ("EditFileArgs", {"path": "a", "expected_sha256": "bad", "old": "", "new": ""}),
        ("RunCommandArgs", {"executable": "python", "argv": [], "timeout": -1}),
        ("FinishRunArgs", {"outcome": "invented", "report": ""}),
        ("ActivateSkillArgs", {"name": "a", "reason": "x", "extra": "bad"}),
    ],
)
def test_builtin_schemas_are_strict(model, value):
    with pytest.raises(ValidationError):
        getattr(schemas, model).model_validate(value)


async def test_completion_returns_normalized_proposal_without_truncation():
    registry = dispatch.ToolRegistry(max_result_bytes=100)

    async def finish(args):
        return args

    registry.register("finish_run", "Finish", schemas.FinishRunArgs, finish, kind="completion")
    result = (
        await registry.dispatch_batch(
            [call("finish_run", {"outcome": "succeeded", "report": "x" * 1000})],
            ledger(),
            Deadline(10),
        )
    )[0]
    assert result.ok and result.value["outcome"] == "succeeded"
    assert result.value["report"] == "x" * 1000


async def test_handler_deadline_is_terminal_and_records_attempt():
    registry = dispatch.ToolRegistry()

    async def hang(args):
        await asyncio.Event().wait()

    registry.register("write", "Write", schemas.WriteFileArgs, hang)
    usage = ledger()
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await registry.dispatch_batch([call()], usage, Deadline(0.01))
    assert usage.charged_tool_calls == 1
    assert registry.last_results[0].error["code"] == "budget_exhausted"


def test_every_builtin_schema_supports_approved_arguments():
    examples = {
        "ActivateSkillArgs": {"name": "skill", "reason": "needed"},
        "ListFilesArgs": {"path": "generated"},
        "ReadTextArgs": {"path": "generated/file"},
        "SearchTextArgs": {"root": "generated", "query": "hello"},
        "ReadMediaArgs": {"path": "generated/photo", "representation": "image"},
        "WriteFileArgs": {"path": "generated/a", "content": "hi"},
        "EditFileArgs": {
            "path": "generated/a",
            "expected_sha256": "a" * 64,
            "old": "hi",
            "new": "bye",
        },
        "RunCommandArgs": {
            "executable": "python",
            "argv": ["--version"],
            "cwd": "generated",
            "env_refs": {},
            "timeout": 1,
        },
        "RegisterArtifactArgs": {
            "path": "generated/a",
            "format": "txt",
            "role": "primary",
            "description": "Output",
        },
        "FinishRunArgs": {"outcome": "succeeded", "report": "Done"},
    }
    for name, example in examples.items():
        model = getattr(schemas, name)
        model.model_validate(example)
        with pytest.raises(ValidationError):
            model.model_validate({**example, "unknown": True})
    with pytest.raises(ValidationError):
        schemas.WriteFileArgs(path="a", content="", overwrite=True)


async def test_repeated_activation_can_share_response_with_actions():
    registry = dispatch.ToolRegistry(activation_is_new=lambda call: False)
    seen = []

    async def handler(args):
        seen.append(args)
        return "ok"

    registry.register("activate", "Activate", schemas.WriteFileArgs, handler, kind="activation")
    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    results = await registry.dispatch_batch([call("activate"), call()], ledger(), Deadline(10))
    assert all(result.ok for result in results) and len(seen) == 2


async def test_activation_classifier_failure_rejects_mixture():
    def broken(call):
        raise RuntimeError("secret")

    registry = dispatch.ToolRegistry(activation_is_new=broken)
    seen = []

    async def handler(args):
        seen.append(args)

    registry.register("activate", "Activate", schemas.WriteFileArgs, handler, kind="activation")
    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    results = await registry.dispatch_batch([call("activate"), call()], ledger(), Deadline(10))
    assert not any(result.ok for result in results) and not seen


async def test_handler_timeout_is_not_misclassified_as_execution_deadline():
    registry = dispatch.ToolRegistry()

    async def handler(args):
        raise TimeoutError("private connection details")

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    result = (await registry.dispatch_batch([call()], ledger(), Deadline(10)))[0]
    assert result.error["code"] == "internal_error"
    assert "private" not in json.dumps(result.as_dict())


@pytest.mark.parametrize("timeout", [False, True])
async def test_cancellation_retains_cleanup_notes_and_unknown_outcome(timeout):
    registry = dispatch.ToolRegistry()
    entered = asyncio.Event()
    note = "process_cleanup_failed: Could not settle the owned process."

    async def handler(args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.add_note(note)
            error.outcome_unknown = True
            raise

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    task = asyncio.create_task(
        registry.dispatch_batch(
            [call(), call(ident="2")], ledger(), Deadline(0.05 if timeout else 10)
        )
    )
    await entered.wait()
    if not timeout:
        task.cancel()
    with pytest.raises(RunnerError if timeout else asyncio.CancelledError) as caught:
        await task
    if timeout:
        assert caught.value.code == "budget_exhausted"
        assert caught.value.details["cleanup_errors"] == [note]
        assert caught.value.details["outcome_unknown"] is True
    else:
        assert caught.value.__notes__ == [note]
    assert len(registry.last_results) == 1
    details = registry.last_results[0].error["details"]
    assert details["outcome_unknown"] is True
    assert details["cleanup_errors"] == [note]


async def test_too_small_error_limit_still_records_charged_outcome():
    registry = dispatch.ToolRegistry(max_result_bytes=2)
    usage = ledger()
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await registry.dispatch_batch([call("missing")], usage, Deadline(10))
    assert len(registry.last_results) == 1
    assert registry.last_results[0].error["code"] == "budget_exhausted"
    assert usage.charged_tool_calls == 1


async def test_full_result_envelope_fits_byte_limit():
    registry = dispatch.ToolRegistry(max_result_bytes=256)

    async def handler(args):
        return "x" * 250

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    result = (await registry.dispatch_batch([call()], ledger(), Deadline(10)))[0]
    assert len(json.dumps(result.as_dict(), ensure_ascii=True).encode()) <= 256
    assert result.as_dict()["value"]["truncated"] is True


async def test_terminal_diagnostics_bounded_but_original_exception_preserved():
    registry = dispatch.ToolRegistry(max_result_bytes=256)
    failure = RunnerError("budget_exhausted", "x" * 10000, details={"note": "y" * 10000})

    async def handler(args):
        raise failure

    registry.register("write", "Write", schemas.WriteFileArgs, handler)
    with pytest.raises(RunnerError) as caught:
        await registry.dispatch_batch([call()], ledger(), Deadline(10))
    assert caught.value is failure
    assert caught.value.details["tool"] == "write"
    result = registry.last_results[0]
    assert len(json.dumps(result.as_dict(), ensure_ascii=True).encode()) <= 256
    assert result.error["code"] == "budget_exhausted"


async def test_long_call_identity_and_tiny_limit_remain_bounded():
    registry = dispatch.ToolRegistry(max_result_bytes=2)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await registry.dispatch_batch([call("missing", ident="x" * 10000)], ledger(), Deadline(10))
    assert registry.last_results[0].as_dict() == {}
    assert registry.last_results[0].call_id == "x" * 10000


def test_result_limit_must_fit_a_json_object():
    with pytest.raises(ValueError):
        dispatch.ToolRegistry(max_result_bytes=1)


async def test_invalid_arguments_identify_only_declared_fields():
    from skillrunner.tools.schemas import WriteFileArgs

    registry = dispatch.ToolRegistry()
    registry.register("write_file", "Write", WriteFileArgs, lambda args: None)
    arguments = {
        "path": "scratch/out",
        "content": "data",
        "expected_sha256": "",
        "secret-extra-field": "private",
    }
    results = await registry.dispatch_batch(
        [ModelToolCall("call", "write_file", arguments, json.dumps(arguments))],
        UsageLedger(max_steps=1, max_tool_calls=1, max_tokens=1000),
        Deadline(10),
    )
    issues = results[0].error["issues"]
    assert any(issue.get("field") == "expected_sha256" for issue in issues)
    assert "secret-extra-field" not in json.dumps(issues)
    assert "private" not in json.dumps(issues)


@pytest.mark.parametrize("extra", [False, True])
async def test_overwrite_digest_guidance_is_safe_and_does_not_execute(extra):
    registry = dispatch.ToolRegistry()
    seen = []

    async def handler(args):
        seen.append(args)

    registry.register("write_file", "Write", schemas.WriteFileArgs, handler)
    arguments = {"path": "scratch/private", "content": "PRIVATE-CONTENT", "overwrite": True}
    if extra:
        arguments["SECRET-FIELD"] = "SECRET-VALUE"
    usage = ledger()
    result = (await registry.dispatch_batch([call("write_file", arguments)], usage, Deadline(10)))[
        0
    ]
    assert usage.charged_tool_calls == 1
    assert not result.executed and not seen
    assert result.error["code"] == "invalid_arguments"
    encoded = json.dumps(result.as_dict())
    assert all(
        secret not in encoded
        for secret in ("PRIVATE-CONTENT", "SECRET-FIELD", "SECRET-VALUE", "scratch/private")
    )
    if not extra:
        assert result.error["issues"] == [
            {
                "type": "overwrite_digest_required",
                "field": "expected_sha256",
                "guidance": (
                    "Read the current file with read_text and copy its full SHA-256 into "
                    "expected_sha256. Use overwrite=true only for intentional replacement."
                ),
            }
        ]


async def test_custom_validator_cannot_impersonate_overwrite_guidance():
    from pydantic import model_validator
    from pydantic_core import PydanticCustomError

    class CustomArgs(schemas.WriteFileArgs):
        @model_validator(mode="after")
        def reject(self):
            raise PydanticCustomError("overwrite_digest_required", "SECRET-MESSAGE")

    registry = dispatch.ToolRegistry()

    async def handler(args):
        pytest.fail("Invalid calls must not execute")

    registry.register("custom", "Custom", CustomArgs, handler)
    result = (await registry.dispatch_batch([call("custom")], ledger(), Deadline(10)))[0]
    assert result.error["issues"] == [{"type": "overwrite_digest_required"}]
    assert "SECRET-MESSAGE" not in json.dumps(result.as_dict())
