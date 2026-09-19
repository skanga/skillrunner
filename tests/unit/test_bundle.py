import importlib
import json

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.recording.bundle") is not None, (
        "Durable run bundles are not implemented"
    )
    return importlib.import_module("skillrunner.recording.bundle")


def test_initial_bundle_is_durable_and_nonterminal(tmp_path):
    bundle = api().RunBundle.create(tmp_path / "outputs", invocation_directory=tmp_path)
    manifest = json.loads((bundle.root / "run.json").read_text())
    assert manifest["lifecycle"]["status"] == "initializing"
    assert manifest["lifecycle"]["exit_code"] is None
    for relative in ["artifacts", "work/inputs", "work/skills", "work/scratch", "work/staging"]:
        assert (bundle.root / relative).is_dir()
    assert (bundle.root / "result.md").is_file()
    assert (bundle.root / "events.jsonl").is_file()


def test_identical_invocations_get_distinct_bundles(tmp_path):
    first = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    second = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    assert first.root != second.root


def test_finalization_redacts_and_returns_absolute_receipt(tmp_path):
    bundle = api().RunBundle.create(
        tmp_path, invocation_directory=tmp_path, secrets=["hidden-token"]
    )
    bundle.state["request"]["prompt"] = "use hidden-token"
    receipt = bundle.finalize(status="failed", exit_code=6, answer="Partial answer hidden-token")
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 6
    assert "# Run failed" in (bundle.root / "result.md").read_text()
    assert receipt["report_path"] == str(bundle.root / "result.md")
    assert "hidden-token" not in (bundle.root / "run.json").read_text()
    assert "hidden-token" not in (bundle.root / "result.md").read_text()


def test_report_includes_assumptions_secondary_outputs_and_incomplete_work(tmp_path):
    bundle = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    bundle.state["diagnostics"]["assumptions"] = ["Assumed UTC timezone"]
    bundle.state["diagnostics"]["incomplete_work"] = ["Chart needs source data"]
    bundle.state["diagnostics"]["warnings"] = ["Image validator unavailable"]
    bundle.state["outputs"]["artifacts"] = [
        {
            "path": str(bundle.root / "artifacts/chart.csv"),
            "description": "Partial chart",
            "status": "incomplete",
        }
    ]
    bundle.finalize(status="failed", exit_code=6, answer="Partial result")
    report = (bundle.root / "result.md").read_text()
    assert "[Partial chart](artifacts/chart.csv)" in report
    for text in [
        "Assumed UTC timezone",
        "Chart needs source data",
        "Image validator unavailable",
        "Partial chart",
        "incomplete",
        "chart.csv",
    ]:
        assert text in report


def test_failure_after_publication_preserves_published_output(tmp_path, monkeypatch):
    module = api()
    bundle = module.RunBundle.create(tmp_path / "outputs", invocation_directory=tmp_path)
    published = tmp_path / "published.txt"
    published.write_text("validated output")
    bundle.state["outputs"]["publication"] = {"state": "committed", "path": str(published)}
    bundle.state["outputs"]["primary_output"] = str(published)
    original = module.atomic_write
    elapsed = [2.0]

    def fail_report(path, content):
        if path.name == "result.md":
            elapsed[0] += 0.5
            raise OSError("disk full")
        return original(path, content)

    monkeypatch.setattr(module, "atomic_write", fail_report)
    receipt = bundle.finalize(
        status="succeeded", exit_code=0, answer="Done", elapsed_seconds=lambda: elapsed[0]
    )
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 6
    assert receipt["primary_output"] == str(published)
    assert receipt["errors"][0]["code"] == "post_publication_reporting_failed"
    assert published.read_text() == "validated output"
    assert bundle.state["outputs"]["publication"]["state"] == "committed"
    assert bundle.state["identity"]["finished_at"] is not None
    assert bundle.state["usage"]["elapsed_seconds"] == 3.0


def test_failed_atomic_replace_preserves_previous_manifest(tmp_path, monkeypatch):
    module = api()
    destination = tmp_path / "run.json"
    destination.write_text("old content")

    def fail(*args, **kwargs):
        raise OSError("failed install")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        module.atomic_write(destination, b"new content")
    assert destination.read_text() == "old content"
    assert list(tmp_path.iterdir()) == [destination]


def test_finalization_is_idempotent(tmp_path):
    bundle = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    first = bundle.finalize(status="failed", exit_code=6, answer="Failed")
    second = bundle.finalize(status="succeeded", exit_code=0, answer="Ignored")
    assert first == second
    assert bundle.state["lifecycle"]["status"] == "failed"


def test_event_close_failure_still_returns_nonzero_receipt(tmp_path, monkeypatch):
    bundle = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    original = bundle.events.close

    def fail_close():
        original()
        raise OSError("close failure")

    monkeypatch.setattr(bundle.events, "close", fail_close)
    elapsed = iter([1.0, 2.0, 3.0])
    receipt = bundle.finalize(
        status="succeeded", exit_code=0, answer="Done", elapsed_seconds=lambda: next(elapsed)
    )
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 6
    assert "# Run failed" in (bundle.root / "result.md").read_text()
    manifest = json.loads((bundle.root / "run.json").read_text())
    assert manifest["identity"]["finished_at"] is not None
    assert manifest["usage"]["elapsed_seconds"] == 3.0


def test_error_report_includes_the_rerun_action(tmp_path):
    bundle = api().RunBundle.create(tmp_path, invocation_directory=tmp_path)
    bundle.state["diagnostics"]["errors"] = [
        {
            "code": "missing_dependency",
            "message": "Format validator unavailable.",
            "suggested_action": "Configure an allowlisted PDF validator and rerun.",
        }
    ]
    bundle.finalize(status="blocked", exit_code=4, answer="Cannot validate the generated file.")
    assert (
        "Configure an allowlisted PDF validator and rerun."
        in (bundle.root / "result.md").read_text()
    )


def test_finalization_records_post_report_timing_and_renders_elapsed_time(tmp_path, monkeypatch):
    module = api()
    bundle = module.RunBundle.create(tmp_path, invocation_directory=tmp_path)
    bundle.state["usage"].update(
        execution_elapsed_seconds=2.0,
        post_execution_elapsed_seconds=0.5,
    )
    writes = []
    elapsed = [2.5]
    original = module.atomic_write

    def record_write(path, content):
        if path.name == "result.md":
            writes.append(content.decode())
            elapsed[0] += 0.75
        return original(path, content)

    monkeypatch.setattr(module, "atomic_write", record_write)
    monkeypatch.setattr(
        module,
        "_now",
        lambda: "finished-after-report" if writes else pytest.fail("timestamped before report"),
    )
    bundle.finalize(
        status="failed",
        exit_code=6,
        answer="Partial",
        elapsed_seconds=lambda: elapsed[0],
    )

    manifest = json.loads((bundle.root / "run.json").read_text())
    assert len(writes) == 2
    assert "Execution elapsed: 2.000 seconds" in writes[-1]
    assert "Post-execution work (cleanup, validation, publication): 0.500 seconds" in writes[-1]
    assert "Reporting elapsed at sample: 0.750 seconds" in writes[-1]
    assert "Overall elapsed at report sample: 3.250 seconds" in writes[-1]
    assert "excludes the final report write" in writes[-1]
    assert manifest["usage"]["elapsed_seconds"] == 4.0
    assert manifest["usage"]["overall_elapsed_seconds_at_report"] == 3.25
    assert manifest["usage"]["reporting_elapsed_seconds_at_report"] == 0.75
    assert manifest["identity"]["finished_at"] == "finished-after-report"
