import argparse
import hashlib
import json
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from skillrunner.qualification import run


@pytest.mark.parametrize("corpus_track", [None, "explicit", "automatic"])
@pytest.mark.parametrize("budget_usd", ["5", "25"])
async def test_driver_keeps_all_repetitions_ungraded_and_never_replaces_evidence(
    tmp_path, monkeypatch, corpus_track, budget_usd
):
    config = tmp_path / "config.toml"
    config.write_text("""default_model = "luna"
[models.luna]
model = "gpt-5.6-luna"
base_url = "http://unused.invalid/v1"
auth_mode = "none"
context_window_tokens = 1050000
max_output_tokens = 128000
""")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps({"cases": [{"id": "text", "prompt": "Write", "expected_skills": ["writer"]}]})
    )
    ledger = tmp_path / "spending.json"
    ledger.write_text(
        json.dumps(
            {
                "budget_usd": budget_usd,
                "charged_usd": "0.0015080",
                "requests": [{"charge_usd": "0.0015080"}],
            }
        )
    )
    args = argparse.Namespace(
        config=config,
        cases=cases,
        skills_dir=tmp_path / "skills",
        output_dir=tmp_path / "outputs",
        model="luna",
        case=["text"],
        repeats=3,
        ledger=ledger,
        result=tmp_path / "evidence.json",
        max_steps=8,
        max_tool_calls=20,
        max_tokens=200000,
        max_output=2048,
        timeout="9m",
    )
    if corpus_track:
        document = json.loads(cases.read_text())
        document["corpus_version"] = "test"
        document["cases"][0].update(
            inputs=[],
            prerequisites=["Runtime available"],
            selection={
                "explicit_required_skills": ["writer"],
                "automatic_expected_skills": ["writer"],
            },
        )
        cases.write_text(json.dumps(document))
        source_hash = hashlib.sha256(cases.read_bytes()).hexdigest()
        args.prepared_inputs = tmp_path / "prepared.json"
        args.prepared_inputs.write_text(
            json.dumps(
                {
                    "corpus_version": "test",
                    "source_manifest_sha256": source_hash,
                    "cases": [{"id": "text", "input_paths": []}],
                }
            )
        )
        args.preflight = tmp_path / "preflight.json"
        args.preflight.write_text(
            json.dumps(
                {
                    "source_manifest_sha256": source_hash,
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "prepared_inputs_sha256": hashlib.sha256(
                        args.prepared_inputs.read_bytes()
                    ).hexdigest(),
                    "cases": [
                        {
                            "id": "text",
                            "status": "ready",
                            "prerequisites": [
                                {
                                    "requirement": "Runtime available",
                                    "evidence": "local test fixture",
                                }
                            ],
                        }
                    ],
                }
            )
        )
        args.track = corpus_track
    invocations = []
    observed_timeouts = []
    observed_output_caps = []

    async def task(request, settings, **kwargs):
        invocations.append(request)
        observed_timeouts.append(settings.limits.timeout)
        observed_output_caps.append(settings.models["luna"].max_output_tokens)
        manifest = tmp_path / f"manifest-{len(invocations)}.json"
        manifest.write_text(json.dumps({"provenance": {"activated_skills": [{"name": "writer"}]}}))
        output = tmp_path / f"answer-{len(invocations)}.md"
        output.write_text("Answer")
        return {
            "status": "succeeded",
            "manifest_path": str(manifest),
            "primary_output": str(output),
        }

    monkeypatch.setattr(run, "run_with_signals", task)
    result = await run.evaluate(args)
    assert len(invocations) == 3
    assert observed_timeouts == [540.0] * 3
    assert observed_output_caps == [2048] * 3
    assert invocations[0].required_skills == (["writer"] if corpus_track == "explicit" else [])
    if corpus_track:
        assert len(result["corpus"]["population"]) == 1
        assert result["corpus"]["track"] == corpus_track
    assert all(item["selection_matches"] and item["primary_exists"] for item in result["runs"])
    assert all(item["semantic_grade"] is None for item in result["runs"])
    assert result["release_qualified"] is False
    assert result["charged_usd"] == "0.0015080"
    with pytest.raises(FileExistsError):
        await run.evaluate(args)
    assert len(json.loads(args.result.read_text())["runs"]) == 3
    if corpus_track:
        args.result = tmp_path / "must-not-exist.json"
        args.prepared_inputs.write_text(args.prepared_inputs.read_text() + "\n")
        with pytest.raises(ValueError, match="bound"):
            await run.evaluate(args)
        assert not args.result.exists()


def test_no_match_selection_requires_an_actual_terminal_decision():
    assert run.selection_matches([], [], "failed") is False
    assert run.selection_matches([], [], "no_matching_skill") is True
    assert run.selection_matches(["writer"], ["writer"], "failed") is True


async def test_missing_command_environment_fails_before_paid_qualification(tmp_path, monkeypatch):
    reference = "SKILLRUN_TEST_MISSING_COMMAND_REFERENCE"
    monkeypatch.delenv(reference, raising=False)
    config = tmp_path / "config.toml"
    executable = json.dumps(sys.executable)
    config.write_text(
        f'''default_model = "luna"
[models.luna]
model = "gpt-5.6-luna"
base_url = "http://unused.invalid/v1"
auth_mode = "none"
context_window_tokens = 1050000
max_output_tokens = 128000
[policy]
allowed_executables = [{executable}]
allowed_env = ["{reference}"]
[policy.command_env.{executable}]
UV_OFFLINE = "{reference}"
'''
    )
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [{"id": "text", "prompt": "Write"}]}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"budget_usd": "5", "charged_usd": "0", "requests": []}))
    args = argparse.Namespace(
        config=config,
        cases=cases,
        skills_dir=tmp_path / "skills",
        output_dir=tmp_path / "outputs",
        model="luna",
        case=["text"],
        repeats=1,
        ledger=ledger,
        result=tmp_path / "evidence.json",
        max_steps=8,
        max_tool_calls=20,
        max_tokens=200000,
        max_output=2048,
        timeout="9m",
    )

    invocations = []

    async def task(*positional, **kwargs):
        invocations.append(True)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"provenance": {"activated_skills": []}}))
        return {"status": "succeeded", "manifest_path": str(manifest), "primary_output": None}

    monkeypatch.setattr(run, "run_with_signals", task)
    with pytest.raises(ValueError, match=reference):
        await run.evaluate(args)
    assert not args.result.exists()
    assert not invocations
    assert json.loads(ledger.read_text())["charged_usd"] == "0"


@pytest.mark.parametrize("manifest_content", [None, "invalid JSON", "{}"])
async def test_unreadable_manifest_is_recorded_and_halts_remaining_trials(
    tmp_path, monkeypatch, manifest_content
):
    config = tmp_path / "config.toml"
    config.write_text("""default_model = "luna"
[models.luna]
model = "gpt-5.6-luna"
base_url = "http://unused.invalid/v1"
auth_mode = "none"
context_window_tokens = 1050000
max_output_tokens = 128000
""")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [{"id": "text", "prompt": "Write"}]}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"budget_usd": "5", "charged_usd": "0.25", "requests": [{"charge_usd": "0.25"}]})
    )
    args = argparse.Namespace(
        config=config,
        cases=cases,
        skills_dir=tmp_path / "skills",
        output_dir=tmp_path / "outputs",
        model="luna",
        case=["text"],
        repeats=3,
        ledger=ledger,
        result=tmp_path / "evidence.json",
        max_steps=8,
        max_tool_calls=20,
        max_tokens=200000,
        max_output=2048,
    )
    calls = []
    if manifest_content is not None:
        (tmp_path / "missing.json").write_text(manifest_content)

    async def task(*positional, **kwargs):
        calls.append(True)
        return {
            "status": "succeeded",
            "manifest_path": str(tmp_path / "missing.json"),
            "primary_output": None,
        }

    monkeypatch.setattr(run, "run_with_signals", task)
    result = await run.evaluate(args)
    assert len(calls) == 1
    assert result["evidence_status"] == "incomplete"
    assert result["charged_usd"] == "0.25"
    assert result["runs"][0]["evidence_error"] == "Could not read the returned run manifest."
    assert result["runs"][0]["selection_matches"] is None
    assert result["release_qualified"] is False
    assert json.loads(args.result.read_text()) == result


def test_unbilled_qualification_rates_are_scoped_to_approved_endpoint():
    profile = SimpleNamespace(
        model="hosted_vllm/openai/gpt-oss-120b", base_url="https://ehl.infra.adobe.net/v1"
    )
    rates, basis = run.qualification_rates(profile)
    assert rates.cost(131072, 131072) == Decimal("0")
    assert "unbilled" in basis
    profile.base_url = "https://different.invalid/v1"
    with pytest.raises(ValueError, match="pricing"):
        run.qualification_rates(profile)


def test_cli_reports_nonzero_when_qualification_evidence_is_incomplete(monkeypatch):
    arguments = ["qualification"]
    for name in ("config", "skills-dir", "output-dir", "cases", "ledger", "result"):
        arguments.extend(["--" + name, "unused"])
    arguments.extend(["--case", "text"])
    monkeypatch.setattr("sys.argv", arguments)

    async def incomplete(args):
        return {"evidence_status": "incomplete"}

    monkeypatch.setattr(run, "evaluate", incomplete)
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 1
