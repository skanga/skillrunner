"""First-run, output, and context feedback contracts."""

import json
from pathlib import Path

from typer.testing import CliRunner

from skillrunner.cli.app import app
from skillrunner.config.sources import resolve_settings, select_model
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.coordinator import run_task
from tests.integration.test_coordinator import Adapter, call, finish, fixture


async def test_default_answer_is_separate_from_report(tmp_path):
    settings = fixture(tmp_path)
    receipt = await run_task(
        RunRequest(prompt="Write", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(profile, [finish()]),
    )
    assert receipt["exit_code"] == 0
    assert receipt["primary_output"] != receipt["report_path"]
    assert Path(receipt["primary_output"]).suffix == ".md"
    assert Path(receipt["primary_output"]).read_text() == "Completed answer."


async def test_reads_preserve_capacity_for_next_turn(tmp_path):
    settings = fixture(tmp_path)
    settings.models["test"].context_window_tokens = 32768
    source = tmp_path / "notes.txt"
    source.write_text("word " * 4096)
    seen = []

    def next_turn(messages):
        result = json.loads(messages[-1]["content"])["value"]
        seen.append(result)
        assert result["truncated"] is True
        assert result["next_offset"] > 0
        return finish()[0:1]

    receipt = await run_task(
        RunRequest(
            prompt="Read a preview",
            invocation_directory=tmp_path,
            required_skills=["writer"],
            inputs=[source],
        ),
        settings,
        environ={},
        adapter_factory=lambda profile, key: Adapter(
            profile, [[call("read_text", {"path": "input-1/notes.txt"})], next_turn]
        ),
    )
    assert receipt["exit_code"] == 0, receipt["errors"]
    assert seen[0]["context_remaining_estimate"] > 0


def test_explicit_output_directory_can_be_new(tmp_path):
    request = RunRequest(
        prompt="Write",
        invocation_directory=tmp_path,
        output=Path("new-folder"),
        output_kind="directory",
    )
    assert request.output_is_directory
    assert not (tmp_path / "new-folder").exists()


def test_explicit_prompt_filename_infers_format(tmp_path):
    request = RunRequest(prompt="Write the report to report.pdf", invocation_directory=tmp_path)
    assert request.format == "pdf"


def test_initializer_is_complete_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["init", "--model", "my-model", "--context-window", "32768", "--max-output", "4096"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    settings = resolve_settings(tmp_path, {}, {})
    assert select_model(settings).model == "my-model"
    assert (tmp_path / "skills/summarize/SKILL.md").is_file()
    config = (tmp_path / "skillrun.toml").read_bytes()
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 2
    assert (tmp_path / "skillrun.toml").read_bytes() == config


def test_explicit_alias_ignores_ambient_endpoint(tmp_path):
    fixture(tmp_path)
    settings = resolve_settings(
        tmp_path, {"model_alias": "test"}, {"OPENAI_BASE_URL": "http://other.invalid/v1"}
    )
    assert select_model(settings).model == "arbitrary"
    assert settings.base_url is None


def test_runtime_progress_and_quiet(tmp_path, monkeypatch):
    from skillrunner.runtime import signals

    fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        signals, "OpenAICompatibleAdapter", lambda profile, api_key: Adapter(profile, [finish()])
    )
    args = ["run", "Write", "-s", "writer", "-j"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "executing" in result.stderr
    assert "model" in result.stderr
    assert len(result.stdout.splitlines()) == 1
    quiet = CliRunner().invoke(app, [*args, "-q"])
    assert quiet.exit_code == 0
    assert quiet.stderr == ""
