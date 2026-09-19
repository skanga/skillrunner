import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillrunner.config.sources import resolve_settings
from skillrunner.domain.request import RunRequest
from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage


def api():
    assert importlib.util.find_spec("skillrunner.runtime.coordinator") is not None
    return importlib.import_module("skillrunner.runtime.coordinator")


def fixture(tmp_path, *, skill=True):
    if skill:
        package = tmp_path / "skills/writer"
        package.mkdir(parents=True)
        (package / "SKILL.md").write_text(
            "---\nname: writer\ndescription: Write a report.\n---\nUse precise prose."
        )
    (tmp_path / "skillrun.toml").write_text("""default_model = "test"
[models.test]
base_url = "http://example.invalid/v1"
model = "arbitrary"
auth_mode = "none"
context_window_tokens = 100000
max_output_tokens = 4000
[limits]
shutdown_grace = "0s"
""")
    return resolve_settings(tmp_path, {}, {})


class Adapter:
    def __init__(self, profile, calls):
        self.profile = profile
        self.calls = iter(calls)
        self.requests = []
        self.closed = False

    async def discover_capabilities(self, deadline):
        return SimpleNamespace(
            profile=self.profile, discovered={}, configured={}, sources={}, metadata_source=None
        )

    async def complete(self, messages, tool_schemas, output_limit, request_deadline):
        self.requests.append(messages)
        calls = next(self.calls)
        if callable(calls):
            calls = calls(messages)
        return ModelReply(None, tuple(calls), "tool_calls", ModelUsage(100, 30, 130), None)

    async def aclose(self):
        self.closed = True


def call(name, arguments, id="one"):
    return ModelToolCall(id, name, arguments, json.dumps(arguments))


def finish(outcome="succeeded", **kwargs):
    return [call("finish_run", {"outcome": outcome, "report": "Completed answer.", **kwargs})]


async def run(
    tmp_path,
    calls,
    *,
    skills=("writer",),
    skill_exists=True,
    output=None,
    overrides=None,
    prompt="Write an answer",
):
    settings = fixture(tmp_path, skill=skill_exists)
    if overrides:
        settings = resolve_settings(tmp_path, overrides, {})
    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, calls)
        adapters.append(adapter)
        return adapter

    receipt = await api().run_task(
        RunRequest(
            prompt=prompt,
            invocation_directory=tmp_path,
            required_skills=list(skills),
            output=output,
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )
    return receipt, adapters


async def test_no_catalog_finishes_locally_with_durable_failure(tmp_path):
    receipt, adapters = await run(tmp_path, [], skills=(), skill_exists=False)
    assert receipt["status"] == "no_matching_skill"
    assert receipt["exit_code"] == 3
    assert not adapters
    assert Path(receipt["manifest_path"]).is_file()


async def test_blank_text_completion_without_an_artifact_is_not_success(tmp_path):
    receipt, _ = await run(tmp_path, [finish(report=" \n ")])

    assert receipt["status"] == "failed"
    assert receipt["exit_code"] != 0
    assert receipt["primary_output"] is None
    report = Path(receipt["report_path"]).read_text()
    assert "answer" in report.lower()
    assert "Next action:" in report
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["status"] == "failed"


async def test_binary_completion_without_an_artifact_reports_required_artifact(tmp_path):
    receipt, _ = await run(
        tmp_path,
        [finish(report="")],
        output=tmp_path / "requested.pdf",
    )

    assert receipt["status"] == "failed"
    assert receipt["exit_code"] != 0
    assert receipt["errors"][0]["code"] == "artifact_invalid"
    assert "requires a generated artifact" in Path(receipt["report_path"]).read_text()


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        ("no_matching_skill", "no_matching_skill"),
        ("needs_input", "needs_input"),
        ("blocked", "blocked"),
        ("failed", "failed"),
    ],
)
async def test_model_declared_failure_has_nonzero_receipt_and_useful_report(
    tmp_path, outcome, expected_status
):
    explanation = "The requested prerequisite is unavailable."
    receipt, _ = await run(
        tmp_path,
        [finish(outcome=outcome, report=explanation)],
        skills=(),
    )

    assert receipt["status"] == expected_status
    assert receipt["exit_code"] != 0
    assert receipt["primary_output"] is None
    report = Path(receipt["report_path"]).read_text()
    assert explanation in report
    assert "Error (" in report
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["status"] == expected_status
    assert manifest["lifecycle"]["exit_code"] == receipt["exit_code"]


async def test_model_connection_failure_persists_action_and_unknown_outcome(tmp_path):
    import httpx

    from skillrunner.model.openai_compatible import OpenAICompatibleAdapter

    settings = fixture(tmp_path)
    attempts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        attempts.append(request)
        raise httpx.ConnectError("PRIVATE network detail", request=request)

    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: OpenAICompatibleAdapter(
            profile, api_key=key, transport=httpx.MockTransport(handler)
        ),
    )
    assert receipt["status"] == "failed" and len(attempts) == 2
    manifest_text = Path(receipt["manifest_path"]).read_text()
    manifest = json.loads(manifest_text)
    error = manifest["diagnostics"]["errors"][0]
    assert error == receipt["errors"][0]
    assert error["code"] == "model_transport_error"
    assert error["retryable"] is True and error["outcome_certainty"] == "unknown"
    assert "connectivity" in error["suggested_action"]
    report = Path(receipt["report_path"]).read_text()
    assert error["suggested_action"] in report
    assert "PRIVATE" not in report + manifest_text
    assert manifest["usage"]["measurement_quality"] == ["unknown"]
    assert manifest["usage"]["charged_tokens"] > 0


async def test_coordinator_error_preserves_known_actionable_context(tmp_path, monkeypatch):
    from skillrunner.domain.errors import RunnerError

    async def fail_prepare(self):
        raise RunnerError(
            "missing_dependency",
            "Converter is unavailable.",
            details={
                "skill": "writer",
                "tool": "run_command",
                "retryable": True,
                "suggested_action": "Install the converter and rerun.",
            },
        )

    monkeypatch.setattr(api().Coordinator, "prepare", fail_prepare)
    receipt, _ = await run(tmp_path, [])
    error = receipt["errors"][0]
    assert error["skill"] == "writer"
    assert error["tool"] == "run_command"
    assert error["retryable"] is True
    assert error["suggested_action"] == "Install the converter and rerun."


async def test_missing_required_skill_never_connects_model(tmp_path):
    receipt, adapters = await run(tmp_path, [], skills=("missing",))
    assert receipt["status"] == "invalid_request"
    assert not adapters


async def test_explicit_skill_text_completion_and_work_cleanup(tmp_path):
    receipt, adapters = await run(tmp_path, [finish()])
    assert receipt["status"] == "succeeded"
    assert receipt["primary_output"] == receipt["report_path"]
    assert "Completed answer." in Path(receipt["report_path"]).read_text()
    assert "Use precise prose." in json.dumps(adapters[0].requests[0])
    assert adapters[0].closed
    assert not (Path(receipt["manifest_path"]).parent / "work").exists()


@pytest.mark.parametrize("selection", ["default", "explicit", "direct"])
async def test_manifest_records_alias_only_for_named_model_selection(tmp_path, selection):
    settings = fixture(tmp_path)
    settings.models["chosen"] = settings.models["test"].model_copy(
        update={"model": "provider/model-id"}
    )
    expected_alias = "test"
    if selection == "explicit":
        settings.selected_model = "chosen"
        expected_alias = "chosen"
    elif selection == "direct":
        settings.base_url = "http://direct.invalid/custom/v1"
        # The literal ID deliberately collides with a configured alias.
        settings.selected_model = "chosen"
        settings.direct_model.auth_mode = "none"
        settings.direct_model.context_window_tokens = 100000
        settings.direct_model.max_output_tokens = 4000
        expected_alias = None

    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, [finish()])
        adapters.append(adapter)
        return adapter

    receipt = await api().run_task(
        RunRequest(
            prompt="Write an answer",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert receipt["status"] == "succeeded"
    model = json.loads(Path(receipt["manifest_path"]).read_text())["model"]
    assert model["alias"] == expected_alias
    expected_model = {"default": "arbitrary", "explicit": "provider/model-id", "direct": "chosen"}
    assert model["model"] == expected_model[selection] == adapters[0].profile.model
    assert model["base_url"] == adapters[0].profile.base_url


async def test_work_retention_is_explicit_and_outputs_are_independent(tmp_path):
    settings = fixture(tmp_path)
    original = tmp_path / "source.txt"
    original.write_text("private input marker", encoding="utf-8")
    request = RunRequest(
        prompt="Write a report",
        invocation_directory=tmp_path,
        required_skills=["writer"],
        inputs=[original],
    )
    assert settings.diagnostics.retain_work is False
    for keep_work in (False, True):
        settings.diagnostics.retain_work = keep_work
        receipt = await api().run_task(
            request,
            settings,
            environ={},
            adapter_factory=lambda profile, key: Adapter(profile, [*artifact_calls(), finish()]),
        )
        assert receipt["status"] == "succeeded" and receipt["exit_code"] == 0
        bundle = Path(receipt["manifest_path"]).parent
        manifest = json.loads(Path(receipt["manifest_path"]).read_text())
        assert manifest["lifecycle"]["cleanup"] == {
            "work_retained": keep_work,
            "owned_pids_remaining": [],
        }
        assert (bundle / "work").exists() == keep_work
        primary = Path(receipt["primary_output"])
        assert primary.parent == bundle / "artifacts"
        assert primary.read_text() == "generated output"
        if keep_work:
            assert (bundle / "work/skills/writer/SKILL.md").is_file()
            assert any(
                path.read_text() == "private input marker"
                for path in (bundle / "work/inputs").rglob("source.txt")
            )
            (bundle / "work/scratch/report.md").write_text("changed retained work")
            assert primary.read_text() == "generated output"
            assert "Work directory retained" in Path(receipt["report_path"]).read_text()
        else:
            assert not list(bundle.rglob("SKILL.md"))
            assert not list(bundle.rglob("source.txt"))
            for name in ("run.json", "result.md", "events.jsonl"):
                assert "private input marker" not in (bundle / name).read_text()
        assert original.read_text() == "private input marker"


async def test_automatic_activation_loads_complete_instructions_next_turn(tmp_path):
    receipt, adapters = await run(
        tmp_path,
        [
            [call("activate_skill", {"name": "writer", "reason": "Matches requested report."})],
            finish(),
        ],
        skills=(),
    )
    assert receipt["status"] == "succeeded"
    assert "Use precise prose." not in json.dumps(adapters[0].requests[0])
    assert "Use precise prose." in json.dumps(adapters[0].requests[1])


@pytest.mark.parametrize("tool_limit", [1, 5, 6])
async def test_mixed_batches_have_no_effects_and_corrections_keep_charges(tmp_path, tool_limit):
    def correction(messages):
        results = [json.loads(message["content"]) for message in messages[-2:]]
        assert all(result["error"]["code"] == "model_protocol_error" for result in results)
        assert all(result["executed"] is False for result in results)
        assert not list(tmp_path.rglob("marker.txt"))

    def activate_after_rejection(messages):
        correction(messages)
        resources = json.loads(messages[2]["content"].split("\n", 1)[1])
        assert resources["active_skills"] == []
        return [call("activate_skill", {"name": "writer", "reason": "Corrected activation"}, "3")]

    def mixed_completion(messages):
        resources = json.loads(messages[2]["content"].split("\n", 1)[1])
        assert len(resources["active_skills"]) == 1
        active = resources["active_skills"][0]
        assert active["instructions"] == (tmp_path / "skills/writer/SKILL.md").read_text()
        return [
            call("finish_run", {"outcome": "succeeded", "report": "Premature"}, "4"),
            call("write_file", {"path": "scratch/marker.txt", "content": "forbidden"}, "5"),
        ]

    def finish_after_rejection(messages):
        correction(messages)
        return [call("finish_run", {"outcome": "succeeded", "report": "Corrected finish"}, "6")]

    receipt, adapters = await run(
        tmp_path,
        [
            [
                call("activate_skill", {"name": "writer", "reason": "Premature"}, "1"),
                call("write_file", {"path": "scratch/marker.txt", "content": "forbidden"}, "2"),
            ],
            activate_after_rejection,
            mixed_completion,
            finish_after_rejection,
        ],
        skills=(),
        overrides={"max_tool_calls": tool_limit},
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert len(adapters[0].requests) == (1 if tool_limit == 1 else 4)
    assert manifest["usage"]["requested_tool_calls"] == (2 if tool_limit == 1 else 6)
    assert manifest["usage"]["charged_tool_calls"] == (0 if tool_limit == 1 else tool_limit)
    activated = manifest["provenance"]["activated_skills"]
    if tool_limit == 1:
        assert activated == []
    else:
        assert len(activated) == 1 and activated[0]["reason"] == "Corrected activation"
    assert receipt["status"] == ("succeeded" if tool_limit == 6 else "limit_exceeded")
    assert receipt["exit_code"] == (0 if tool_limit == 6 else 7)
    assert not list(tmp_path.rglob("marker.txt"))
    if tool_limit == 6:
        assert "Corrected finish" in Path(receipt["report_path"]).read_text()
    else:
        assert receipt["primary_output"] is None


async def test_model_cannot_succeed_without_activated_skill(tmp_path):
    output = tmp_path / "result.md"
    receipt, _ = await run(tmp_path, [finish()], skills=(), output=output)
    assert receipt["status"] != "succeeded"
    assert not output.exists()


async def test_external_text_output_is_independent_from_operational_report(tmp_path):
    output = tmp_path / "result.md"
    receipt, _ = await run(tmp_path, [finish()], output=output)
    assert receipt["status"] == "succeeded"
    assert receipt["primary_output"] == str(output)
    assert output.read_text() == "Completed answer."
    assert Path(receipt["report_path"]) != output
    assert receipt["artifact_paths"]


def artifact_calls():
    return [
        [call("write_file", {"path": "scratch/report.md", "content": "generated output"})],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/report.md",
                    "format": "md",
                    "role": "primary",
                    "description": "Generated report",
                },
            )
        ],
    ]


async def test_generated_primary_is_retained_validated_and_published(tmp_path):
    output = tmp_path / "published.md"
    receipt, _ = await run(tmp_path, [*artifact_calls(), finish()], output=output)
    assert receipt["status"] == "succeeded"
    assert output.read_text() == "generated output"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["outputs"]["artifacts"][0]["status"] == "validated"
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"


async def test_output_directory_uses_filename_requested_in_prompt(tmp_path):
    output = tmp_path / "published.json"
    output.mkdir()
    receipt, _ = await run(
        tmp_path, [*artifact_calls(), finish()], output=output, prompt="Write to report.md"
    )
    assert receipt["status"] == "succeeded"
    assert receipt["primary_output"] == str(output / "report.md")
    assert (output / "report.md").read_text() == "generated output"


async def test_output_directory_generates_unique_name_for_unnamed_artifact(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _ = await run(first_root, [*artifact_calls(), finish()], output=output)
    second, _ = await run(second_root, [*artifact_calls(), finish()], output=output)
    for receipt in (first, second):
        assert receipt["status"] == "succeeded"
        primary = Path(receipt["primary_output"])
        assert primary.parent == output
        assert primary.name.startswith("output-")
        assert primary.suffix == ".md"
        assert primary.read_text() == "generated output"
    assert first["primary_output"] != second["primary_output"]


async def test_output_directory_does_not_treat_input_filename_as_output_request(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    receipt, _ = await run(
        tmp_path,
        [*artifact_calls(), finish()],
        output=output,
        prompt="Read report.md as input and write a summary.",
    )
    assert receipt["status"] == "succeeded"
    assert Path(receipt["primary_output"]).name.startswith("output-")


async def test_output_directory_generates_unique_text_answer_filename(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _ = await run(first_root, [finish()], output=output)
    second, _ = await run(second_root, [finish()], output=output)
    assert first["status"] == second["status"] == "succeeded"
    for receipt in (first, second):
        primary = Path(receipt["primary_output"])
        assert primary.parent == output
        assert primary.suffix == ".md"
        assert primary.read_text() == "Completed answer."
    assert first["primary_output"] != second["primary_output"]


async def test_output_directory_preserves_existing_named_file_without_overwrite(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    existing = output / "report.md"
    existing.write_text("keep me")
    receipt, _ = await run(
        tmp_path, [*artifact_calls(), finish()], output=output, prompt="Write to report.md"
    )
    assert receipt["status"] == "failed"
    assert receipt["errors"][0]["code"] == "publication_failed"
    assert existing.read_text() == "keep me"


async def test_output_directory_replacement_during_run_is_rejected(tmp_path):
    output = tmp_path / "published"
    output.mkdir()

    def replace_directory(_messages):
        output.rename(tmp_path / "original-published")
        output.mkdir()
        return finish()

    receipt, _ = await run(tmp_path, [replace_directory], output=output)
    assert receipt["status"] == "failed"
    assert receipt["errors"][0]["code"] == "publication_failed"
    assert not list(output.iterdir())


async def test_failed_execution_preserves_registered_incomplete_artifact(tmp_path):
    output = tmp_path / "published.md"
    receipt, _ = await run(tmp_path, [*artifact_calls(), finish("failed")], output=output)
    assert receipt["status"] == "failed"
    assert not output.exists()
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["outputs"]["artifacts"][0]["status"] == "incomplete"


@pytest.mark.parametrize("log_failure", [False, True])
async def test_partial_recovery_records_registration_without_losing_output(
    tmp_path, monkeypatch, log_failure
):
    from skillrunner.recording.events import EventLog

    original_emit = EventLog.emit

    def emit(self, name, payload, **kwargs):
        if log_failure and name == "artifact_registered":
            raise OSError("injected registration log failure")
        return original_emit(self, name, payload, **kwargs)

    monkeypatch.setattr(EventLog, "emit", emit)
    receipt, _ = await run(tmp_path, [artifact_calls()[0], finish("needs_input")])
    assert receipt["status"] == "needs_input"
    assert receipt["exit_code"] == 5
    assert receipt["errors"][0]["code"] == "missing_decision"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "incomplete"
    assert Path(artifact["path"]).read_text() == "generated output"
    assert not (Path(receipt["manifest_path"]).parent / "work").exists()
    events = [
        json.loads(line)
        for line in (Path(receipt["manifest_path"]).parent / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    registered = [event for event in events if event["event_type"] == "artifact_registered"]
    if log_failure:
        assert not registered
        assert any(error["code"] == "reporting_failed" for error in receipt["errors"])
    else:
        assert len(registered) == 1
        assert registered[0]["payload"]["artifact_id"] == artifact["id"]
        assert "tool_name" not in registered[0]
        assert "call_id" not in registered[0]
    assert not any(event["event_type"] == "artifact_validated" for event in events)


async def test_non_success_completion_preserves_missing_requirements_as_action(tmp_path):
    receipt, _ = await run(
        tmp_path,
        [finish("blocked", missing_requirements=["Install the configured PDF validator."])],
    )
    error = receipt["errors"][0]
    assert error["details"]["missing_requirements"] == ["Install the configured PDF validator."]
    assert error["suggested_action"] == "Install the configured PDF validator."


@pytest.mark.parametrize("outcome,exit_code", [("succeeded", 0), ("needs_input", 5)])
async def test_completion_records_assumptions_in_manifest_and_report(tmp_path, outcome, exit_code):
    assumptions = ["Used UTC for dates without a timezone.", "Sorted equal values by name."]
    receipt, adapters = await run(tmp_path, [finish(outcome, assumptions=assumptions)])
    assert receipt["status"] == outcome and receipt["exit_code"] == exit_code
    assert len(adapters[0].requests) == 1
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["diagnostics"]["assumptions"] == assumptions
    report = Path(receipt["report_path"]).read_text()
    assert "## Assumptions" in report
    assert all(assumption in report for assumption in assumptions)


async def test_success_with_missing_requirements_is_blocked_and_preserves_details(tmp_path):
    missing = ["Configure the required parser.", "Supply the missing reference input."]
    output = tmp_path / "unpublished.md"
    receipt, _ = await run(
        tmp_path,
        [*artifact_calls(), finish("succeeded", missing_requirements=missing)],
        output=output,
    )
    assert receipt["status"] == "blocked" and receipt["exit_code"] == 4
    assert receipt["primary_output"] is None and not output.exists()
    error = receipt["errors"][0]
    assert error["details"].get("missing_requirements") == missing
    assert error["suggested_action"] == missing[0]
    assert missing[0] in Path(receipt["report_path"]).read_text()
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["diagnostics"]["errors"][0]["details"]["missing_requirements"] == missing
    artifact = manifest["outputs"]["artifacts"][0]
    assert artifact["status"] == "incomplete"
    assert Path(artifact["path"]).read_text() == "generated output"


async def test_run_command_defaults_to_unique_active_skill_snapshot(tmp_path):
    import sys

    executable = str(Path(sys.executable).resolve())
    settings = fixture(tmp_path)
    settings.policy.allowed_executables = [executable]
    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": ["-c", "from pathlib import Path; Path('made.txt').write_text('ok')"],
                },
            )
        ],
        finish(),
    ]
    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "succeeded", receipt["errors"]


async def test_run_command_explains_environment_reference_shape_and_can_retry(tmp_path):
    import os
    import sys

    executable = str(Path(sys.executable).resolve())
    settings = fixture(tmp_path)
    settings.policy.allowed_executables = [executable]
    executable_stat = Path(executable).stat()
    settings.policy.executable_identities[executable] = (
        executable_stat.st_dev,
        executable_stat.st_ino,
        executable_stat.st_size,
        executable_stat.st_mtime_ns,
    )
    settings.policy.command_env = {executable: {"PATH": "TEST_COMMAND_PATH"}}

    def retry(messages):
        response = json.dumps(messages)
        assert "child variable name" in response
        assert "omit env_refs" in response
        return [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": ["-c", "print('ok')"],
                },
                id="retry",
            )
        ]

    calls = [
        [
            call(
                "run_command",
                {"executable": executable, "argv": ["-V"], "env_refs": {executable: "PATH"}},
            )
        ],
        retry,
        finish(),
    ]
    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={"TEST_COMMAND_PATH": os.environ["PATH"]},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )

    assert receipt["status"] == "succeeded"
    events = [
        json.loads(line)
        for line in Path(receipt["report_path"]).with_name("events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event_type"] == "tool_completed" and event["payload"].get("call_id") == "retry"
        for event in events
    )


async def test_run_command_without_cwd_rejects_ambiguous_active_skills(tmp_path):
    import sys

    second = tmp_path / "skills/editor"
    second.mkdir(parents=True)
    (second / "SKILL.md").write_text(
        "---\nname: editor\ndescription: Edit a report.\n---\nEdit precisely."
    )
    executable = str(Path(sys.executable).resolve())
    marker = tmp_path / "launched"
    settings = fixture(tmp_path)
    settings.policy.allowed_executables = [executable]
    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": ["-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                },
            )
        ],
        lambda messages: (
            finish("failed")
            if "Provide an explicit cwd" in json.dumps(messages)
            else (_ for _ in ()).throw(AssertionError("missing actionable cwd diagnostic"))
        ),
    ]
    receipt = await api().run_task(
        RunRequest(
            prompt="Write", invocation_directory=tmp_path, required_skills=["writer", "editor"]
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "failed"
    assert not marker.exists()


async def test_exhausted_turn_budget_preserves_completed_artifact_work(tmp_path):
    receipt, _ = await run(tmp_path, artifact_calls(), overrides={"max_steps": 2})
    assert receipt["status"] == "limit_exceeded"
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"


async def test_input_snapshots_have_distinct_logical_roots(tmp_path):
    settings = fixture(tmp_path)
    sources = []
    for folder in ("a", "b"):
        path = tmp_path / folder / "same.txt"
        path.parent.mkdir()
        path.write_text(folder)
        sources.append(path)
    adapters = []

    def factory(profile, key):
        adapter = Adapter(profile, [finish()])
        adapters.append(adapter)
        return adapter

    receipt = await api().run_task(
        RunRequest(
            prompt="Summarize",
            invocation_directory=tmp_path,
            inputs=sources,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert receipt["status"] == "succeeded"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert [item["id"] for item in manifest["provenance"]["inputs"]] == ["input-1", "input-2"]
    assert [path.read_text() for path in sources] == ["a", "b"]


async def test_zero_tool_storage_fails_with_capacity_status_without_connecting(tmp_path):
    settings = fixture(tmp_path)
    settings.storage.max_tool_output_bytes = 0

    def no_connection(profile, key):
        raise AssertionError("Must not connect")

    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=no_connection,
    )
    assert receipt["status"] == "limit_exceeded"
    assert receipt["errors"][0]["code"] == "budget_exhausted"


async def test_unregistered_partial_output_is_preserved_after_limit(tmp_path):
    receipt, _ = await run(tmp_path, [artifact_calls()[0]], overrides={"max_steps": 1})
    assert receipt["status"] == "limit_exceeded"
    assert receipt["artifact_paths"]
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"


async def test_real_command_and_configured_external_validator(tmp_path):
    import sys

    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text()
        + "\n[policy]\nallowed_executables = ["
        + json.dumps(executable)
        + "]\n[artifacts.validators.pdf]\ncommand = "
        + json.dumps(executable)
        + '\nargs = ["-c", "import pathlib,sys; '
        "assert pathlib.Path(sys.argv[1]).read_bytes() == b'candidate'; "
        'print(\'accepted\')", "{path}"]\n'
    )
    settings = resolve_settings(tmp_path, {}, {})
    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": [
                        "-c",
                        "from pathlib import Path; Path('report.pdf').write_bytes(b'candidate')",
                    ],
                    "cwd": "scratch",
                },
            )
        ],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/report.pdf",
                    "format": "pdf",
                    "role": "primary",
                    "description": "Candidate",
                },
            )
        ],
        finish(),
    ]
    receipt = await api().run_task(
        RunRequest(
            prompt="Produce file",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="pdf",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "succeeded", receipt["errors"]
    assert Path(receipt["primary_output"]).read_bytes() == b"candidate"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["outputs"]["artifacts"][0]["validation_level"] == "external"
    assert manifest["outputs"]["artifacts"][0]["validator"]["command"] == executable


@pytest.mark.parametrize("expanded_bytes", [100, 101])
async def test_archive_expansion_limit_controls_publication_and_partial_retention(
    tmp_path, expanded_bytes
):
    import hashlib
    import sys
    import zipfile

    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text()
        + "\n[policy]\nallowed_executables = ["
        + json.dumps(executable)
        + "]\n[storage]\nmax_archive_expanded_bytes = 100\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    output = tmp_path / "published.zip"
    output.write_bytes(b"original destination")
    script = (
        "import sys,zipfile\n"
        "with zipfile.ZipFile('report.zip','w',compression=zipfile.ZIP_DEFLATED) as z:\n"
        " z.writestr('first.txt',b'A'*60)\n"
        " z.writestr('second.txt',b'B'*(int(sys.argv[1])-60))\n"
    )
    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": ["-c", script, str(expanded_bytes)],
                    "cwd": "scratch",
                },
            )
        ],
        [
            call(
                "register_artifact",
                {
                    "path": "scratch/report.zip",
                    "format": "zip",
                    "role": "primary",
                    "description": "Compressed output",
                },
            )
        ],
        finish(),
    ]
    receipt = await api().run_task(
        RunRequest(
            prompt="Produce an archive",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            output=output,
            overwrite=True,
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    root = Path(receipt["manifest_path"]).parent
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["controls"]["storage"]["max_archive_expanded_bytes"] == 100
    assert not manifest["lifecycle"]["cleanup"]["owned_pids_remaining"]
    assert not (root / "work").exists()
    artifact = manifest["outputs"]["artifacts"][0]
    retained = Path(artifact["path"])
    assert hashlib.sha256(retained.read_bytes()).hexdigest() == artifact["digest"]
    with zipfile.ZipFile(retained) as archive:
        assert sum(len(archive.read(name)) for name in archive.namelist()) == expanded_bytes
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    names = [event["event_type"] for event in events]
    assert names.count("process_stopped") == 1
    if expanded_bytes == 100:
        assert receipt["status"] == "succeeded" and receipt["exit_code"] == 0
        assert output.read_bytes() == retained.read_bytes()
        assert artifact["status"] == "validated" and artifact["validation_level"] == "container"
        assert manifest["outputs"]["publication"]["state"] == "committed"
        assert names.index("process_stopped") < names.index("artifact_validated")
    else:
        assert receipt["status"] == "limit_exceeded" and receipt["exit_code"] == 7
        assert receipt["errors"][0]["code"] == "budget_exhausted"
        assert output.read_bytes() == b"original destination"
        assert artifact["status"] == "incomplete" and artifact["validation_level"] == "none"
        assert manifest["outputs"]["primary_output"] is None
        assert "limit_reached" in names
        assert "artifact_validated" not in names and "publication_committed" not in names


async def test_text_materialization_does_not_collide_with_model_scratch(tmp_path):
    receipt, _ = await run(
        tmp_path,
        [
            [call("write_file", {"path": "scratch/answer.md", "content": "scratch notes"})],
            finish(),
        ],
        output=tmp_path / "published.md",
    )
    assert receipt["status"] == "succeeded"
    assert (tmp_path / "published.md").read_text() == "Completed answer."


async def test_failed_publication_records_failed_attempt(tmp_path, monkeypatch):
    from skillrunner.artifacts.publication import PublicationTarget
    from skillrunner.domain.errors import RunnerError

    def fail(*args, **kwargs):
        raise RunnerError("publication_failed", "Destination unavailable.")

    monkeypatch.setattr(PublicationTarget, "publish", fail)
    receipt, _ = await run(tmp_path, [finish()], output=tmp_path / "published.md")
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert receipt["status"] == "failed"
    assert manifest["outputs"]["publication"]["state"] == "failed"
    assert receipt["artifact_paths"]


@pytest.mark.parametrize("outcome", ["committed", "failed", "committed_cleanup_failed"])
async def test_publication_events_record_actual_commit_state(tmp_path, monkeypatch, outcome):
    from skillrunner.artifacts.publication import PublicationTarget
    from skillrunner.domain.errors import RunnerError

    output = tmp_path / "published.md"
    original = PublicationTarget.publish

    def publish(target, *args, **kwargs):
        if outcome == "failed":
            output.write_text("competing output")
        result = original(target, *args, **kwargs)
        if outcome == "committed_cleanup_failed":
            raise RunnerError(
                "publication_failed",
                "Staging cleanup failed.",
                details={"committed": True, "path": str(output)},
            )
        return result

    monkeypatch.setattr(PublicationTarget, "publish", publish)
    receipt, _ = await run(tmp_path, [finish()], output=output)
    events = [
        json.loads(line)
        for line in Path(receipt["manifest_path"])
        .with_name("events.jsonl")
        .read_text()
        .splitlines()
    ]
    publications = [event for event in events if event["event_type"].startswith("publication_")]
    assert [event["event_type"] for event in publications] == [
        "publication_failed" if outcome == "failed" else "publication_committed"
    ]
    assert publications[0]["payload"]["path"] == str(output)
    assert publications[0]["payload"]["state"] == ("failed" if outcome == "failed" else "committed")
    if outcome != "committed":
        assert publications[0]["payload"]["error"] == "publication_failed"
        assert receipt["status"] == "failed"
    else:
        assert receipt["status"] == "succeeded"
    assert publications[0]["sequence"] < events[-1]["sequence"]
    assert events[-1]["event_type"] == "run_finalized"
    assert output.read_text() == (
        "competing output" if outcome == "failed" else "Completed answer."
    )


async def test_post_publication_reporting_failure_keeps_output(tmp_path, monkeypatch):
    from skillrunner.recording.bundle import RunBundle

    original = RunBundle.save

    def fail_final(bundle):
        if bundle.state["lifecycle"]["phase"] == "finalizing":
            raise OSError("Cannot save report state")
        return original(bundle)

    monkeypatch.setattr(RunBundle, "save", fail_final)
    output = tmp_path / "published.md"
    receipt, _ = await run(tmp_path, [finish()], output=output)
    assert receipt["status"] == "failed"
    assert receipt["primary_output"] == str(output)
    assert output.read_text() == "Completed answer."
    assert receipt["errors"][-1]["code"] == "post_publication_reporting_failed"


@pytest.mark.parametrize("competing", [False, True])
async def test_publication_event_failure_preserves_original_outcome(
    tmp_path, monkeypatch, competing
):
    from skillrunner.domain.errors import RunnerError
    from skillrunner.recording.events import EventLog

    output = tmp_path / "published.md"
    original = EventLog.emit

    def fail_publication_event(log, event_type, *args, **kwargs):
        if event_type.startswith("publication_"):
            raise RunnerError("reporting_failed", "Could not persist the event log.")
        return original(log, event_type, *args, **kwargs)

    def respond(messages):
        if competing:
            output.write_text("competing output")
        return finish()

    monkeypatch.setattr(EventLog, "emit", fail_publication_event)
    receipt, _ = await run(tmp_path, [respond], output=output)
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 6
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    if competing:
        assert receipt["errors"][0]["code"] == "publication_failed"
        assert receipt["errors"][0]["details"]["reporting_errors"]
        assert output.read_text() == "competing output"
        assert manifest["outputs"]["publication"]["state"] == "failed"
    else:
        assert receipt["errors"][0]["code"] == "post_publication_reporting_failed"
        assert receipt["primary_output"] == str(output)
        assert output.read_text() == "Completed answer."
        assert manifest["outputs"]["publication"]["state"] == "committed"


async def test_surviving_writer_preserves_raw_work_without_freezing(tmp_path, monkeypatch):
    from skillrunner.artifacts.registry import ArtifactRegistry
    from skillrunner.domain.errors import RunnerError
    from skillrunner.runtime.processes import ProcessSupervisor

    async def cannot_close(owner):
        owner.active[123] = object()
        raise RunnerError("cleanup_failed", "Writer could not be stopped.")

    def must_not_freeze(*args, **kwargs):
        raise AssertionError("Cannot freeze while writers remain")

    monkeypatch.setattr(ProcessSupervisor, "aclose", cannot_close)
    monkeypatch.setattr(ArtifactRegistry, "freeze", must_not_freeze)
    receipt, _ = await run(tmp_path, [*artifact_calls(), finish()])
    assert receipt["status"] == "failed"
    bundle = Path(receipt["manifest_path"]).parent
    assert (bundle / "work/scratch/report.md").read_text() == "generated output"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["cleanup"]["work_retained"]
    assert not receipt["artifact_paths"]


async def test_running_command_is_stopped_when_scratch_limit_is_observed(tmp_path):
    import sys
    import time

    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text()
        + "\n[policy]\nallowed_executables = ["
        + json.dumps(executable)
        + "]\n[storage]\nmax_scratch_bytes = 4\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    calls = [
        [
            call(
                "run_command",
                {
                    "executable": executable,
                    "argv": [
                        "-c",
                        "from pathlib import Path; import time; "
                        "Path('partial.txt').write_text('x'*100); time.sleep(3)",
                    ],
                    "cwd": "scratch",
                },
            )
        ],
        finish(),
    ]
    started = time.monotonic()
    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "limit_exceeded"
    assert time.monotonic() - started < 2
    assert receipt["errors"][0]["code"] == "budget_exhausted"


async def test_activation_admission_includes_full_tool_result_before_marking_active(tmp_path):
    settings = fixture(tmp_path)
    settings.models["test"].context_window_tokens = 70000
    skill = tmp_path / "skills/writer/SKILL.md"
    skill.write_text(skill.read_text() + "\n" + "x" * 40000)
    calls = [[call("activate_skill", {"name": "writer", "reason": "Required for task."})], finish()]
    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "limit_exceeded"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert not manifest["provenance"]["activated_skills"]


async def test_explicit_activation_capacity_error_identifies_skill_and_tool(tmp_path):
    settings = fixture(tmp_path)
    settings.models["test"].context_window_tokens = 30000
    skill = tmp_path / "skills/writer/SKILL.md"
    skill.write_text(skill.read_text() + "\n" + "x" * 40000)
    receipt = await api().run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, []),
    )
    assert receipt["status"] == "limit_exceeded"
    assert receipt["errors"][0]["skill"] == "writer"
    assert receipt["errors"][0]["tool"] == "activate_skill"
    assert receipt["errors"][0]["suggested_action"] == (
        "Reduce the writer skill instructions or configure a larger model context, then rerun."
    )


async def test_model_content_logging_is_opt_in(tmp_path):
    settings = fixture(tmp_path)
    request = RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"])
    for enabled in (False, True):
        settings.diagnostics.log_content = enabled
        receipt = await api().run_task(
            request,
            settings,
            environ={},
            adapter_factory=lambda profile, key: Adapter(profile, [finish()]),
        )
        assert receipt["status"] == "succeeded"
        events = (Path(receipt["manifest_path"]).parent / "events.jsonl").read_text()
        assert ("Completed answer." in events) == enabled


async def test_frontmatter_cannot_grant_commands_or_package_write_access(tmp_path, monkeypatch):
    import asyncio
    import sys

    settings = fixture(tmp_path)
    settings.diagnostics.retain_work = True
    package = tmp_path / "skills/writer"
    instructions = (
        "---\nname: writer\ndescription: Write a report.\n"
        "allowed-tools: run_command write_file\n---\n"
        "This skill requires writing generated.txt inside its package.\n"
    )
    (package / "SKILL.md").write_text(instructions)

    async def forbidden_launch(*args, **kwargs):
        pytest.fail("Frontmatter must not authorize subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_launch)
    report = (
        "Incompatible skill: its required package write is read-only; command permission is absent."
    )

    def finish_after_denials(messages):
        observations = [message["content"] for message in messages if message["role"] == "tool"]
        assert len(observations) == 2
        assert "file_access_denied" in observations[0]
        assert "read-only" in observations[0]
        assert "command_not_allowed" in observations[1]
        return [
            call(
                "finish_run",
                {
                    "outcome": "blocked",
                    "report": report,
                    "missing_requirements": [
                        "Adapt the skill to write outputs in the run workspace."
                    ],
                },
            )
        ]

    calls = [
        [call("write_file", {"path": "skill-writer/generated.txt", "content": "forbidden"})],
        [call("run_command", {"executable": str(Path(sys.executable).resolve()), "argv": ["-V"]})],
        finish_after_denials,
    ]
    receipt = await api().run_task(
        RunRequest(prompt="Run writer", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, calls),
    )
    assert receipt["status"] == "blocked" and receipt["exit_code"] == 4
    assert report in Path(receipt["report_path"]).read_text()
    assert (package / "SKILL.md").read_text() == instructions
    snapshot = Path(receipt["manifest_path"]).parent / "work/skills/writer"
    assert (snapshot / "SKILL.md").read_text() == instructions
    assert not (package / "generated.txt").exists()
    assert not (snapshot / "generated.txt").exists()
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["controls"]["allowed_executables"] == []
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []


async def test_mcp_tools_share_dispatcher_and_keep_remote_identity(tmp_path, monkeypatch):
    from skillrunner.config.models import MCPConfig
    from skillrunner.mcp import MCPTool

    settings = fixture(tmp_path)
    settings.mcp = {"remote": MCPConfig(transport="streamable-http", url="http://unused/mcp")}
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    calls = []

    class Manager:
        tools = [MCPTool("mcp_test", "remote", "real-name", "Remote action", schema)]
        reverse_map = {"mcp_test": ("remote", "real-name")}
        diagnostics = []
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            calls.append("connect")

        async def invoke(self, name, arguments):
            calls.append((name, arguments))
            return {"content": [{"type": "text", "text": "done"}]}

        async def aclose(self):
            calls.append("close")

    class Model(Adapter):
        async def complete(self, messages, tool_schemas, *args):
            remote = next(item for item in tool_schemas if item["function"]["name"] == "mcp_test")
            assert remote["function"]["parameters"] == schema
            assert "remote" in remote["function"]["description"]
            assert "real-name" in remote["function"]["description"]
            return await super().complete(messages, tool_schemas, *args)

    monkeypatch.setattr(api(), "MCPManager", Manager, raising=False)
    receipt = await api().run_task(
        RunRequest(prompt="Use remote", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Model(
            profile, [[call("mcp_test", {"x": 42})], finish()]
        ),
    )
    assert receipt["status"] == "succeeded", receipt
    assert calls == ["connect", ("mcp_test", {"x": 42}), "close"]
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["provenance"]["mcp_tools"]["mcp_test"] == ["remote", "real-name"]
    events = [
        json.loads(line)
        for line in Path(receipt["manifest_path"])
        .with_name("events.jsonl")
        .read_text()
        .splitlines()
    ]
    starts = [e for e in events if e["event_type"] == "mcp_request_started"]
    assert starts[0]["payload"]["server"] == "remote"
    assert starts[0]["payload"]["tool"] == "real-name"
    assert starts[0]["tool_name"] == "mcp_test"
    assert starts[0]["call_id"] == "one"


async def test_mcp_unknown_outcome_stops_without_retry(tmp_path, monkeypatch):
    from skillrunner.config.models import MCPConfig
    from skillrunner.domain.errors import RunnerError
    from skillrunner.mcp import MCPTool

    settings = fixture(tmp_path)
    settings.mcp = {"remote": MCPConfig(transport="streamable-http", url="http://unused/mcp")}
    calls = []

    class Manager:
        tools = [MCPTool("remote_call", "remote", "write", "Write", {"type": "object"})]
        reverse_map = {"remote_call": ("remote", "write")}
        diagnostics = []

        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            pass

        async def invoke(self, name, arguments):
            calls.append(name)
            raise RunnerError(
                "external_outcome_unknown", "Disconnected", details={"outcome_unknown": True}
            )

        async def aclose(self):
            calls.append("close")

    monkeypatch.setattr(api(), "MCPManager", Manager, raising=False)
    receipt = await api().run_task(
        RunRequest(prompt="Use remote", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile, [[call("remote_call", {})], finish()]
        ),
    )
    assert receipt["status"] == "failed"
    assert calls == ["remote_call", "close"]
    assert receipt["errors"][0]["outcome_certainty"] == "unknown"


async def test_model_receives_absolute_workspace_and_safe_file_contract(tmp_path):
    receipt, adapters = await run(tmp_path, [finish()])
    instructions = adapters[0].requests[0][0]["content"]
    root = Path(receipt["manifest_path"]).parent
    assert json.dumps(str(root / "work/scratch")) in instructions
    assert json.dumps(str(root / "artifacts")) in instructions
    assert "scratch/answer.md" in instructions
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert len(manifest["controls"]["policy_digest"]) == 64
