"""Keep shipped setup examples and documented switches executable."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillrunner.cli.app import app
from skillrunner.config.sources import resolve_settings


@pytest.mark.parametrize("name", ["skillrun.toml", "skillrun-minimal.toml"])
def test_shipped_configuration_examples_validate(tmp_path, name):
    path = Path(__file__).resolve().parents[2] / "examples" / name
    settings = resolve_settings(tmp_path, {"config": path}, {})
    assert settings.default_model in settings.models
    assert settings.policy.allowed_executables == []


@pytest.mark.parametrize(
    "command,options",
    [
        (
            ["run"],
            ["--output-file", "--output-directory", "--exclude", "--model-alias", "--log-content"],
        ),
        (["doctor"], ["--network", "--model-alias"]),
        (["init"], ["--template", "--context-window", "--max-output", "--api-key-env"]),
        (["runs", "clean"], ["--yes", "--output-dir"]),
        (["inputs", "preview"], ["--exclude", "--input"]),
    ],
)
def test_documented_switches_exist(command, options):
    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0
    assert all(option in result.stdout for option in options)


def test_content_logging_cli_override(tmp_path, monkeypatch):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)
    seen = []

    async def task(request, settings, *, environ):
        seen.append(settings.diagnostics.log_content)
        return {"status": "succeeded", "exit_code": 0}

    monkeypatch.setattr(execution, "run_task", task)
    for switch in ["--log-content", "--no-log-content"]:
        result = CliRunner().invoke(app, ["run", "Write", switch, "-j", "-q"])
        assert result.exit_code == 0
    assert seen == [True, False]
