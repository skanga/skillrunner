"""User-facing setup and diagnostics regressions; no live network."""

import json

import pytest
from typer.testing import CliRunner

from skillrunner.cli.app import app
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError


@pytest.mark.parametrize(
    "text,field",
    [
        ('"broken', "TOML"),
        ("[limits]\nmax_stpes=4", "limits.max_stpes"),
        ('[limits]\ntimeout="private-invalid-value"', "limits.timeout"),
    ],
)
def test_configuration_identifies_location_without_values(tmp_path, text, field):
    (tmp_path / "skillrun.toml").write_text(text)
    with pytest.raises(RunnerError) as caught:
        resolve_settings(tmp_path, {}, {})
    error = caught.value
    assert field in error.message + json.dumps(error.details)
    assert "private-invalid-value" not in str(error) + json.dumps(error.details)
    assert error.details["suggested_action"]


def test_invalid_environment_limit_identifies_variable(tmp_path):
    with pytest.raises(RunnerError) as caught:
        resolve_settings(tmp_path, {}, {"SKILLRUN_MAX_STEPS": "private-value"})
    assert "SKILLRUN_MAX_STEPS" in caught.value.message
    assert "private-value" not in str(caught.value)
    assert caught.value.details["suggested_action"]


def test_missing_config_identified(tmp_path):
    with pytest.raises(RunnerError) as caught:
        resolve_settings(tmp_path, {"config": "missing.toml"}, {})
    assert "missing.toml" in caught.value.message
    assert "not found" in caught.value.message.lower()


@pytest.mark.parametrize(
    "args",
    [
        ["run", "task", "--max-steps", "private-value", "-j"],
        ["run", "task", "--unknown-private-value", "-j"],
        ["run", "task", "--max-steps", "private-value", "-qj"],
    ],
)
def test_parser_errors_honor_json(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "invalid_request"
    assert receipt["errors"][0]["suggested_action"]
    assert "private-value" not in result.stdout + result.stderr
    assert not (tmp_path / "outputs").exists()


def test_early_receipt_keeps_safe_details(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run", "task", "--timeout", "2", "-j"])
    error = json.loads(result.stdout)["errors"][0]
    assert error["details"]["issues"][0]["field"] == "limits.timeout"
    assert error["suggested_action"]
    assert "Next action:" in result.stderr


def test_empty_catalog_has_helpful_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "No skills found" in result.stdout
    assert str(tmp_path / "skills") in result.stdout
    assert "init" in result.stdout


def test_version_needs_no_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skillrun.toml").write_text("invalid TOML")
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_config_show_reports_effective_model_and_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skillrun.toml").write_text("""default_model="local"
[models.local]
base_url="http://localhost:8000/v1"
model="actual-model"
auth_mode="none"
context_window_tokens=32768
max_output_tokens=4096
""")
    result = CliRunner().invoke(app, ["config", "show", "-j"])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.stdout)
    assert shown["model"]["model"] == "actual-model"
    assert shown["sources"]["models.local.model"] == "file"
    assert "md" in shown["supported_formats"]


def test_obsolete_environment_policy_explains_migration(tmp_path):
    (tmp_path / "skillrun.toml").write_text('[policy]\nallowed_env=["TOKEN"]')
    with pytest.raises(RunnerError) as caught:
        resolve_settings(tmp_path, {}, {})
    assert "command_env" in str(caught.value)
