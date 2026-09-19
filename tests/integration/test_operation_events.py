"""Operation records must reflect actual catalog and artifact boundaries."""

import json
import sys
from pathlib import Path

import pytest

from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError
from skillrunner.domain.request import RunRequest
from skillrunner.recording.events import EventLog
from skillrunner.runtime.coordinator import run_task
from tests.integration.test_coordinator import Adapter, artifact_calls, call, finish, fixture, run


def events_for(receipt):
    return [
        json.loads(line)
        for line in Path(receipt["manifest_path"])
        .with_name("events.jsonl")
        .read_text()
        .splitlines()
    ]


@pytest.mark.parametrize("generated", [False, True])
async def test_catalog_and_artifact_events_precede_publication(tmp_path, generated):
    calls = [*artifact_calls(), finish()] if generated else [finish()]
    receipt, _ = await run(tmp_path, calls, output=tmp_path / "published.md")
    assert receipt["status"] == "succeeded"
    events = events_for(receipt)
    names = [event["event_type"] for event in events]
    assert names.index("catalog_prepared") < names.index("skill_activated")
    assert names.index("artifact_registered") < names.index("artifact_validated")
    assert names.index("artifact_validated") < names.index("publication_committed")
    assert names.count("artifact_registered") == names.count("artifact_validated") == 1
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    artifact = manifest["outputs"]["artifacts"][0]
    registered = next(
        event["payload"] for event in events if event["event_type"] == "artifact_registered"
    )
    validated = next(
        event["payload"] for event in events if event["event_type"] == "artifact_validated"
    )
    assert registered["artifact_id"] == validated["artifact_id"] == artifact["id"]
    registered_event = next(
        event for event in events if event["event_type"] == "artifact_registered"
    )
    if generated:
        assert registered_event["tool_name"] == "register_artifact"
        assert registered_event["call_id"] == "one"
    else:
        assert "tool_name" not in registered_event
        assert "call_id" not in registered_event
    assert validated["digest"] == artifact["digest"]
    assert validated["validation_level"] == artifact["validation_level"]
    assert "generated output" not in json.dumps(events)
    assert "Completed answer." not in json.dumps(events)


async def test_invalid_artifact_has_registration_but_no_validation_success_event(tmp_path):
    settings = fixture(tmp_path)
    receipt = await run_task(
        RunRequest(
            prompt="Generate JSON",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            format="json",
            output=tmp_path / "out.json",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, [finish(report="not JSON")]),
    )
    assert receipt["exit_code"] == 6
    assert receipt["errors"][0]["code"] == "artifact_invalid"
    names = [event["event_type"] for event in events_for(receipt)]
    assert "artifact_registered" in names
    assert "artifact_validated" not in names
    assert "publication_committed" not in names


async def test_empty_catalog_event_exists_without_model_connection(tmp_path):
    settings = fixture(tmp_path, skill=False)
    settings.skills_dir.mkdir()

    def forbidden_factory(profile, key):
        pytest.fail("Empty catalog must not connect to a model")

    receipt = await run_task(
        RunRequest(prompt="Produce report", invocation_directory=tmp_path),
        settings,
        environ={},
        adapter_factory=forbidden_factory,
    )
    assert receipt["exit_code"] == 3
    catalog = [event for event in events_for(receipt) if event["event_type"] == "catalog_prepared"]
    assert len(catalog) == 1
    assert catalog[0]["payload"] == {"valid_packages": 0, "rejected_packages": 0}


@pytest.mark.parametrize("event_type", ["artifact_registered", "artifact_validated"])
async def test_artifact_event_write_failure_stops_and_preserves_outputs(
    tmp_path, monkeypatch, event_type
):
    original = EventLog.emit

    def fail_selected(log, name, *args, **kwargs):
        if name == event_type:
            raise RunnerError("reporting_failed", "Could not persist the event log.")
        return original(log, name, *args, **kwargs)

    monkeypatch.setattr(EventLog, "emit", fail_selected)
    receipt, adapters = await run(
        tmp_path, [*artifact_calls(), finish()], output=tmp_path / "out.md"
    )
    assert receipt["exit_code"] == 6
    assert receipt["errors"][0]["code"] == "reporting_failed"
    assert len(adapters[0].requests) == (2 if event_type == "artifact_registered" else 3)
    assert not (tmp_path / "out.md").exists()
    assert receipt["artifact_paths"]
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"


@pytest.mark.parametrize("logging_failure", [False, True])
async def test_failed_model_event_matches_retained_run_accounting(
    tmp_path, monkeypatch, logging_failure
):
    settings = fixture(tmp_path)
    original = EventLog.emit

    def emit(log, name, *args, **kwargs):
        if logging_failure and name == "model_request_failed":
            raise RunnerError("reporting_failed", "private logging detail")
        return original(log, name, *args, **kwargs)

    class DisconnectingAdapter(Adapter):
        async def complete(self, *args):
            if len(self.requests) == 1:
                raise RunnerError("model_connection_failed", "Model connection lost.")
            return await super().complete(*args)

    monkeypatch.setattr(EventLog, "emit", emit)
    receipt = await run_task(
        RunRequest(
            prompt="Produce report",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            output=tmp_path / "out.md",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: DisconnectingAdapter(profile, artifact_calls()),
    )
    assert receipt["exit_code"] == 6
    assert receipt["errors"][0]["code"] == "model_connection_failed"
    assert not (tmp_path / "out.md").exists()
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["usage"]["model_attempts"] == 2
    assert manifest["usage"]["measurement_quality"] == ["reported", "unknown"]
    assert manifest["outputs"]["artifacts"][0]["status"] == "incomplete"
    failures = [
        event for event in events_for(receipt) if event["event_type"] == "model_request_failed"
    ]
    if logging_failure:
        assert not failures
        assert receipt["errors"][0]["details"]["reporting_errors"]
        assert "private logging detail" not in json.dumps(manifest)
    else:
        assert len(failures) == 1
        payload = failures[0]["payload"]
        assert payload["attempt"] == 2
        assert payload["measurement_quality"] == "unknown"
        assert payload["actual_usage"] is None
        assert payload["budget_charge"] > 4000
        assert payload["budget_charge"] + 130 == manifest["usage"]["charged_tokens"]
        assert payload["error_code"] == "model_connection_failed"


async def test_tool_attempts_have_ordered_start_and_outcome_with_call_identity(tmp_path):
    receipt, _ = await run(
        tmp_path,
        [
            [
                call("write_file", {"path": 42, "content": "private argument"}, "invalid"),
                call(
                    "write_file", {"path": "outside/file", "content": "private argument"}, "denied"
                ),
                call("write_file", {"path": "scratch/file.txt", "content": "retained"}, "written"),
            ],
            finish(),
        ],
    )
    assert receipt["status"] == "succeeded"
    events = events_for(receipt)
    attempts = [event for event in events if event["event_type"].startswith("tool_")]
    assert [event["event_type"] for event in attempts] == [
        "tool_started",
        "tool_failed",
        "tool_started",
        "tool_failed",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
    ]
    for start, end in zip(attempts[::2], attempts[1::2], strict=True):
        assert start["call_id"] == end["call_id"] == start["payload"]["call_id"]
        assert start["tool_name"] == end["tool_name"] == start["payload"]["name"]
        assert start["sequence"] < end["sequence"]
        assert start["payload"]["executed"] is False
    assert attempts[1]["payload"]["error_code"] == "invalid_arguments"
    assert attempts[3]["payload"]["error_code"] == "file_access_denied"
    assert attempts[1]["payload"]["executed"] is False
    assert attempts[3]["payload"]["executed"] is False
    assert attempts[5]["payload"]["executed"] is True
    assert "private argument" not in json.dumps(events)
    activated = next(event for event in events if event["event_type"] == "skill_activated")
    assert activated["skill_id"] == "writer"


async def test_rejected_tool_budget_batch_emits_no_tool_started_event(tmp_path):
    receipt, _ = await run(
        tmp_path,
        [
            [
                call("write_file", {"path": "scratch/a", "content": "x"}, "a"),
                call("write_file", {"path": "scratch/b", "content": "x"}, "b"),
            ]
        ],
        overrides={"max_tool_calls": 1},
    )
    assert receipt["exit_code"] == 7
    assert not any(event["event_type"].startswith("tool_") for event in events_for(receipt))
    assert not receipt["artifact_paths"]


async def test_tool_start_log_failure_prevents_handler_effect(tmp_path, monkeypatch):
    original = EventLog.emit

    def fail_start(log, name, *args, **kwargs):
        if name == "tool_started":
            raise RunnerError("reporting_failed", "Cannot write start record.")
        return original(log, name, *args, **kwargs)

    monkeypatch.setattr(EventLog, "emit", fail_start)
    receipt, adapters = await run(tmp_path, [*artifact_calls(), finish()])
    assert receipt["exit_code"] == 6
    assert receipt["errors"][0]["code"] == "reporting_failed"
    assert len(adapters[0].requests) == 1
    assert receipt["errors"][0]["tool"] == "write_file"
    assert not receipt["artifact_paths"]
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["usage"]["charged_tool_calls"] == 1
    assert not (Path(receipt["manifest_path"]).parent / "work").exists()


@pytest.mark.parametrize("limit", ["turns", "context", "scratch"])
async def test_limit_event_matches_stop_and_affected_operation(tmp_path, limit):
    settings = fixture(tmp_path)
    if limit == "turns":
        settings.limits.max_steps = 1
    elif limit == "context":
        settings.models["test"].context_window_tokens = 1
    else:
        settings.storage.max_scratch_bytes = 1
    receipt = await run_task(
        RunRequest(
            prompt="Produce report",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            output=tmp_path / "out.md",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, artifact_calls()),
    )
    assert receipt["status"] == "limit_exceeded"
    assert receipt["exit_code"] == 7
    expected = "context_capacity_exceeded" if limit == "context" else "budget_exhausted"
    assert receipt["errors"][0]["code"] == expected
    events = events_for(receipt)
    limits = [event for event in events if event["event_type"] == "limit_reached"]
    assert len(limits) == 1
    assert limits[0]["payload"]["error_code"] == expected
    assert limits[0]["payload"]["stage"] == receipt["errors"][0]["stage"]
    assert limits[0]["sequence"] < events[-1]["sequence"]
    if limit == "context":
        assert limits[0]["tool_name"] == "activate_skill"
        assert limits[0]["skill_id"] == "writer"
    elif limit == "scratch":
        assert limits[0]["tool_name"] == "write_file"
    else:
        assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"
    assert not (tmp_path / "out.md").exists()


@pytest.mark.parametrize("logging", ["saturated", "failed"])
async def test_limit_reporting_survives_event_log_unavailability(tmp_path, monkeypatch, logging):
    settings = fixture(tmp_path)
    settings.limits.max_steps = 1
    if logging == "saturated":
        settings.storage.max_event_log_bytes = 0
    else:
        original = EventLog.emit

        def fail_limit(log, name, *args, **kwargs):
            if name == "limit_reached":
                raise RunnerError("reporting_failed", "private storage detail")
            return original(log, name, *args, **kwargs)

        monkeypatch.setattr(EventLog, "emit", fail_limit)
    receipt = await run_task(
        RunRequest(
            prompt="Produce report",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            output=tmp_path / "out.md",
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, artifact_calls()),
    )
    assert receipt["exit_code"] == 7
    assert receipt["errors"][0]["code"] == "budget_exhausted"
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    report = Path(receipt["report_path"]).read_text()
    assert manifest["lifecycle"]["stop_reason"] == "budget_exhausted"
    assert "budget_exhausted" in report
    assert not (tmp_path / "out.md").exists()
    assert Path(receipt["artifact_paths"][0]).read_text() == "generated output"
    if logging == "failed":
        assert receipt["errors"][0]["details"]["reporting_errors"]
        assert "Could not persist the limit event." in report
        assert "private storage detail" not in json.dumps(manifest)
    else:
        assert not events_for(receipt)
        assert manifest["diagnostics"]["logging"]["omitted_events"] > 0


@pytest.mark.parametrize("logging_failure", [False, True])
async def test_process_stop_event_and_failure_preserve_cleanup(
    tmp_path, monkeypatch, logging_failure
):
    fixture(tmp_path)
    executable = str(Path(sys.executable).resolve())
    config = tmp_path / "skillrun.toml"
    config.write_text(
        config.read_text() + "\n[policy]\nallowed_executables = [" + json.dumps(executable) + "]\n"
    )
    settings = resolve_settings(tmp_path, {}, {})
    original = EventLog.emit

    def emit(log, name, *args, **kwargs):
        if logging_failure and name == "process_stopped":
            raise RunnerError("reporting_failed", "private log error")
        return original(log, name, *args, **kwargs)

    monkeypatch.setattr(EventLog, "emit", emit)
    receipt = await run_task(
        RunRequest(
            prompt="Run the permitted command",
            invocation_directory=tmp_path,
            required_skills=["writer"],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile,
            [
                [
                    call(
                        "run_command",
                        {"executable": executable, "argv": ["-c", "print('private child text')"]},
                        "child",
                    )
                ],
                finish(),
            ],
        ),
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []
    events = events_for(receipt)
    stopped = [event for event in events if event["event_type"] == "process_stopped"]
    if logging_failure:
        assert receipt["exit_code"] == 6
        assert receipt["errors"][0]["code"] == "reporting_failed"
        assert manifest["usage"]["model_attempts"] == 1
        assert not stopped
    else:
        assert receipt["exit_code"] == 0
        assert len(stopped) == 1
        assert stopped[0]["payload"]["returncode"] == 0
        assert stopped[0]["payload"]["pid"] > 0
        assert stopped[0]["call_id"] == "child"
        assert stopped[0]["tool_name"] == "run_command"
    assert "private log error" not in json.dumps(manifest)
    assert "private child text" not in json.dumps(events)
