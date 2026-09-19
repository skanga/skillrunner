import importlib
import json

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.recording") is not None, (
        "Bounded redacted event recording is not implemented"
    )
    return importlib.import_module("skillrunner.recording.events")


def test_events_redact_secrets_and_omit_content_by_default(tmp_path):
    recorder = api().EventLog(
        tmp_path / "events.jsonl", run_id="run1", max_bytes=10000, secrets=["private-key"]
    )
    recorder.emit(
        "tool_started",
        {"message": "using private-key", "Authorization": "Bearer other"},
        content={"prompt": "private task text"},
    )
    recorder.close()
    encoded = (tmp_path / "events.jsonl").read_text()
    assert "private-key" not in encoded
    assert "Bearer other" not in encoded
    assert "private task text" not in encoded
    event = json.loads(encoded)
    assert event["sequence"] == 1
    assert event["schema_version"] == 1
    assert event["run_id"] == "run1"
    assert event["event_type"] == "tool_started"


def test_opt_in_content_still_redacts_and_excludes_private_reasoning(tmp_path):
    recorder = api().EventLog(
        tmp_path / "events.jsonl",
        run_id="run1",
        max_bytes=10000,
        secrets=["secret"],
        log_content=True,
    )
    recorder.emit(
        "model_completed",
        {},
        content={"public_text": "secret output", "reasoning": "private chain"},
    )
    recorder.close()
    encoded = (tmp_path / "events.jsonl").read_text()
    assert "secret" not in encoded
    assert "private chain" not in encoded
    assert "output" in encoded


def test_log_saturation_is_bounded_and_recorded(tmp_path):
    recorder = api().EventLog(tmp_path / "events.jsonl", run_id="run1", max_bytes=1000)
    for index in range(50):
        recorder.emit("tool_completed", {"index": index, "metadata": "x" * 100})
    recorder.close()
    encoded = (tmp_path / "events.jsonl").read_bytes()
    assert len(encoded) <= 1000
    events = [json.loads(line) for line in encoded.splitlines()]
    assert any(event["event_type"] == "log_truncated" for event in events)
    assert recorder.summary()["omitted_events"] > 0
    assert recorder.summary()["omitted_bytes"] > 0
    assert [event["sequence"] for event in events] == sorted(event["sequence"] for event in events)


def test_zero_log_limit_exposes_omissions_without_preventing_finalization(tmp_path):
    recorder = api().EventLog(tmp_path / "events.jsonl", run_id="run1", max_bytes=0)
    assert not recorder.emit("run_initialized", {})
    recorder.close()
    assert (tmp_path / "events.jsonl").stat().st_size == 0
    assert recorder.summary()["omitted_events"] == 1
    assert recorder.summary()["saturated"]


@pytest.mark.parametrize("scheme", ["https", "HTTPS", "HtTpS"])
def test_embedded_url_credentials_and_query_are_redacted(tmp_path, scheme):
    recorder = api().EventLog(tmp_path / "events.jsonl", run_id="run1", max_bytes=10000)
    recorder.emit(
        "failure",
        {"message": f"Request {scheme}://user:password@example.com/path?token=value failed"},
    )
    recorder.close()
    encoded = (tmp_path / "events.jsonl").read_text()
    assert "password" not in encoded
    assert "token=value" not in encoded
    assert "example.com/path" in encoded


def test_event_records_operation_identifiers(tmp_path):
    recorder = api().EventLog(tmp_path / "events.jsonl", run_id="run1", max_bytes=10000)
    recorder.emit("tool_started", {}, skill_id="sample", tool_name="read_text", call_id="call1")
    recorder.close()
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["skill_id"] == "sample"
    assert event["tool_name"] == "read_text"
    assert event["call_id"] == "call1"


@pytest.mark.parametrize(
    "marker", [{"type": "reasoning"}, {"channel": "analysis"}, {"type": "thinking"}]
)
def test_typed_private_reasoning_is_never_logged(tmp_path, marker):
    recorder = api().EventLog(
        tmp_path / "events.jsonl", run_id="run1", max_bytes=10000, log_content=True
    )
    recorder.emit("model_completed", {}, content={"blocks": [{**marker, "text": "private chain"}]})
    recorder.close()
    assert "private chain" not in (tmp_path / "events.jsonl").read_text()


@pytest.mark.parametrize("role", ["tool", "assistant"])
def test_invalid_protocol_json_is_omitted_without_mutating_content(tmp_path, role):
    raw = '{"Authorization": "opaque-marker", "reasoning": "private-marker"'
    message = (
        {"role": "tool", "content": raw}
        if role == "tool"
        else {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "sample", "arguments": raw}}],
        }
    )
    content = {"messages": [message]}
    before = json.dumps(content)
    recorder = api().EventLog(
        tmp_path / "events.jsonl", run_id="run", max_bytes=10000, log_content=True
    )
    recorder.emit("model_request_started", {}, content=content)
    recorder.close()
    encoded = (tmp_path / "events.jsonl").read_text()
    assert "INVALID_PROTOCOL_JSON_OMITTED" in encoded
    assert "opaque-marker" not in encoded
    assert "private-marker" not in encoded
    assert json.dumps(content) == before


def test_surrogateescaped_filename_does_not_break_recording(tmp_path):
    recorder = api().EventLog(tmp_path / "events.jsonl", run_id="run1", max_bytes=10000)
    recorder.emit("file_observed", {"filename": "file-\udcff"})
    recorder.close()
    assert (
        json.loads((tmp_path / "events.jsonl").read_text())["payload"]["filename"] == "file-\udcff"
    )
