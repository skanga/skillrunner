import importlib
import importlib.util
import json
import struct

import pytest
from pydantic import ValidationError

from skillrunner.config.models import DirectModel
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.context import RunContext


def media_api():
    assert importlib.util.find_spec("skillrunner.model.media") is not None, (
        "PNG media not implemented"
    )
    return importlib.import_module("skillrunner.model.media")


def test_image_accounting_is_explicit_and_default_off():
    assert getattr(DirectModel(), "image_accounting", "missing") is None
    configured = DirectModel(
        input_modalities=["text", "image"], image_accounting="openai-patch-high-v1"
    )
    assert configured.image_accounting == "openai-patch-high-v1"
    with pytest.raises(ValidationError):
        DirectModel(image_accounting="guess-from-model-name")


@pytest.mark.parametrize(
    "width,height,expected",
    [(1024, 1024, 1230), (2048, 2048, 3001), (32, 32, 3), (1, 2048, 78), (4096, 512, 616)],
)
def test_patch_accounting_includes_rounding_allowance(width, height, expected):
    assert media_api().image_tokens(width, height) == expected


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 2), (True, 10), (2**32, 1)])
def test_invalid_dimensions_fail(width, height):
    with pytest.raises(RunnerError):
        media_api().image_tokens(width, height)


def test_png_dimensions_requires_ihdr():
    api = media_api()
    header = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 1200, 1600)
        + b"\x08\x02\x00\x00\x00"
    )
    assert api.png_dimensions(header) == (1200, 1600)
    with pytest.raises(RunnerError):
        api.png_dimensions(b"not png")


def test_context_accounts_media_every_request_without_serializing_bytes_as_text():
    api = media_api()
    image = api.MediaAttachment(
        data=b"opaque-image-bytes", path="scratch/page.png", width=1024, height=1024
    )
    context = RunContext(runner_instructions="rules", prompt="task", catalog=[])
    context.append_media([image])
    request = context.messages()
    assert request[-1]["role"] == "user"
    assert request[-1]["content"][1]["image_url"]["detail"] == "high"
    assert request[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "b3BhcXVlLWltYWdlLWJ5dGVz" not in json.dumps(context.diagnostic_messages())
    assert context.image_token_estimate == 1230
    first = context.estimate([])
    assert context.estimate([]) == first
    context.append_media([image])
    assert context.image_token_estimate == 2460
    assert context.estimate([]) > first + 1230
    assert "opaque-image-bytes" not in repr(image)


@pytest.mark.parametrize("retry", [False, True])
async def test_media_batch_ordering_and_payload_never_enters_events(retry):
    from pydantic import BaseModel

    from skillrunner.model.protocol import ModelReply, ModelToolCall
    from skillrunner.runtime.agent import AgentLoop
    from skillrunner.runtime.budgets import Deadline, UsageLedger
    from skillrunner.tools.dispatch import ToolRegistry

    class Args(BaseModel):
        pass

    image = media_api().MediaAttachment(
        data=b"opaque-image-bytes", path="scratch/p.png", width=32, height=32
    )
    registry = ToolRegistry()

    async def read(_):
        return image

    async def text(_):
        return {"text": "second result"}

    async def finish(_):
        return {"report": "done"}

    registry.register("read_media", "image", Args, read)
    registry.register("read_text", "text", Args, text)
    registry.register("finish_run", "finish", Args, finish, kind="completion")
    calls = (
        ModelToolCall("image", "read_media", {}, "{}"),
        ModelToolCall("text", "read_text", {}, "{}"),
    )

    class Adapter:
        requests = []

        async def complete(self, messages, schemas, output_limit, request_deadline):
            self.requests.append(messages)
            if retry and len(self.requests) == 2:
                raise RunnerError("model_transport_error", "Retry", details={"retryable": True})
            tool_calls = (
                calls
                if len(self.requests) == 1
                else (ModelToolCall("done", "finish_run", {}, "{}"),)
            )
            return ModelReply(None, tool_calls, "tool_calls", None, None)

    adapter = Adapter()
    events = []
    loop = AgentLoop(
        adapter=adapter,
        context=RunContext(runner_instructions="rules", prompt="task", catalog=[]),
        dispatcher=registry,
        ledger=UsageLedger(max_steps=3, max_tool_calls=10, max_tokens=100000),
        deadline=Deadline(10),
        context_window=100000,
        max_output=1000,
        on_event=lambda n, p: events.append((n, p)),
    )
    assert await loop.run() == {"report": "done"}
    messages = adapter.requests[1]
    assert [m["role"] for m in messages[-4:]] == ["assistant", "tool", "tool", "user"]
    assert messages[-1]["content"][1]["type"] == "image_url"
    assert "b3BhcXVlLWltYWdlLWJ5dGVz" not in json.dumps(events)
    if retry:
        assert adapter.requests[1] == adapter.requests[2]
        assert loop.ledger.charged_tool_calls == 3
    assert events[-1][0] == "tool_completed"
    assert any(e[1].get("image_token_estimate") == 3 for e in events)


async def test_oversize_media_is_rejected_not_truncated():
    from pydantic import BaseModel

    from skillrunner.model.protocol import ModelToolCall
    from skillrunner.runtime.budgets import Deadline, UsageLedger
    from skillrunner.tools.dispatch import ToolRegistry

    class Args(BaseModel):
        pass

    registry = ToolRegistry(max_result_bytes=1000)

    async def read(_):
        return media_api().MediaAttachment(
            data=b"x" * 1000, path="scratch/x.png", width=32, height=32
        )

    registry.register("read_media", "image", Args, read)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        await registry.dispatch_batch(
            [ModelToolCall("i", "read_media", {}, "{}")],
            UsageLedger(max_steps=1, max_tool_calls=1, max_tokens=10000),
            Deadline(5),
        )
    assert registry.last_results[0].ok is False


def test_media_context_cannot_bypass_token_admission():
    from skillrunner.runtime.budgets import UsageLedger

    image = media_api().MediaAttachment(
        data=b"bytes", path="scratch/p.png", width=2048, height=2048
    )
    context = RunContext(runner_instructions="rules", prompt="task", catalog=[])
    context.append_media([image])
    with pytest.raises(RunnerError, match="context_capacity_exceeded"):
        UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=100000).reserve_model(
            input_tokens=context.estimate([]), context_window=2000, max_output=100
        )
    with pytest.raises(RunnerError, match="budget_exhausted"):
        UsageLedger(max_steps=2, max_tool_calls=2, max_tokens=2000).reserve_model(
            input_tokens=context.estimate([]), context_window=100000, max_output=100
        )


def test_media_binary_read_is_bounded_and_confined(tmp_path):
    from skillrunner.tools.files import FileTools

    root = tmp_path / "root"
    root.mkdir()
    (root / "large.png").write_bytes(b"x" * 101)
    files = FileTools(max_tool_output_bytes=100)
    files.register_root("input", root)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        files.read_binary_snapshot("input/large.png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    with pytest.raises(RunnerError, match="file_access_denied"):
        files.read_binary_snapshot(str(outside))
    with pytest.raises(RunnerError, match="file_access_denied"):
        files.read_binary_snapshot("input/../outside.png")


async def test_media_validation_cancellation_removes_copy(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from skillrunner.config.models import ExternalValidator, ModelProfile
    from skillrunner.runtime import media
    from skillrunner.runtime.budgets import Deadline
    from skillrunner.runtime.environment import build_child_environment
    from skillrunner.tools.files import FileTools

    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 1, 1) + b"\x08\x02\x00\x00\x00"
    )
    source = tmp_path / "source.png"
    source.write_bytes(data)
    staging = tmp_path / "staging"
    staging.mkdir()
    files = FileTools()
    files.register_root("input", tmp_path)

    async def cancelled(path, *args, **kwargs):
        assert path.read_bytes() == data
        assert path != source
        raise asyncio.CancelledError

    monkeypatch.setattr(media, "validate_external", cancelled)
    profile = ModelProfile(
        base_url="http://example.invalid/v1",
        model="arbitrary",
        input_modalities=["text", "image"],
        image_accounting="openai-patch-high-v1",
    )
    with pytest.raises(asyncio.CancelledError):
        await media.read_png(
            files,
            "input/source.png",
            "image",
            profile=profile,
            validator=ExternalValidator(command="validator", args=["{path}"]),
            supervisor=SimpleNamespace(),
            environment=build_child_environment({}, references={}),
            deadline=Deadline(2),
            staging=staging,
            monitor=lambda: None,
        )
    assert source.read_bytes() == data
    assert list(staging.iterdir()) == []
