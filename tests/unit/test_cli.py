import json

import pytest
from typer.testing import CliRunner


def cli():
    from skillrunner.cli.app import app

    return app


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["hello"],
        ["run"],
        ["run", " "],
        ["run", "x", "--prompt-stdin"],
        ["run", "x", "--max-ste", "2"],
    ],
)
def test_invalid_invocations_never_start(arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli(), arguments)
    assert result.exit_code == 2
    assert not (tmp_path / "outputs").exists()


def test_mixed_aliases_and_last_scalar(tmp_path, monkeypatch):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)
    seen = {}

    async def fake(request, settings, *, environ):
        seen.update(request=request, settings=settings)
        return {"status": "succeeded", "exit_code": 0, "run_id": "sample"}

    monkeypatch.setattr(execution, "run_task", fake)
    result = CliRunner().invoke(
        cli(),
        [
            "run",
            "doctor",
            "-i",
            "a",
            "--input",
            "b",
            "-s",
            "one",
            "--skill",
            "two",
            "-m",
            "first",
            "--model",
            "last",
            "-t",
            "2s",
            "--max-steps",
            "3",
            "--shutdown-grace",
            "0s",
            "-j",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "succeeded"
    assert seen["request"].inputs == [tmp_path / "a", tmp_path / "b"]
    assert seen["request"].required_skills == ["one", "two"]
    assert seen["settings"].selected_model == "last"
    assert seen["settings"].limits.max_steps == 3


def test_cli_accepts_input_and_output_directories_without_changing_defaults(tmp_path, monkeypatch):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "published").mkdir()
    seen = []

    async def fake(request, settings, *, environ):
        seen.append((request, settings))
        return {"status": "succeeded", "exit_code": 0}

    monkeypatch.setattr(execution, "run_task", fake)
    result = CliRunner().invoke(cli(), ["run", "Summarize", "-i", "source", "-o", "published"])
    assert result.exit_code == 0, result.output
    request, settings = seen[0]
    assert request.inputs == [tmp_path / "source"]
    assert request.output == tmp_path / "published"
    assert request.output_is_directory
    assert request.format == "md"
    assert settings.output_dir == tmp_path / "outputs"
    omitted = CliRunner().invoke(cli(), ["run", "Summarize"])
    assert omitted.exit_code == 0, omitted.output
    default_request, default_settings = seen[1]
    assert default_request.inputs == []
    assert default_request.output is None
    assert default_settings.output_dir == tmp_path / "outputs"


@pytest.mark.parametrize("source", ["file", "stdin"])
def test_utf8_prompt_sources(source, tmp_path, monkeypatch):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompt.txt").write_text("Résumé ✨", encoding="utf-8")

    async def fake(request, settings, *, environ):
        assert request.prompt == "Résumé ✨"
        assert request.prompt_source == source
        return {"status": "blocked", "exit_code": 4}

    monkeypatch.setattr(execution, "run_task", fake)
    args = ["-p", "prompt.txt"] if source == "file" else ["--prompt-stdin"]
    result = CliRunner().invoke(cli(), ["run", *args, "-j"], input="Résumé ✨")
    assert result.exit_code == 4, result.output
    assert json.loads(result.stdout)["status"] == "blocked"


def test_help_aliases_no_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli(), ["run", "-h"])
    assert result.exit_code == 0
    for alias in [
        "--prompt-file",
        "--skills-dir",
        "--shutdown-grace",
        "--max-tool-calls",
        "--base-url",
    ]:
        assert alias in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_inspection_rejected_packages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package = tmp_path / "skills" / "bad"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("not valid")
    listed = CliRunner().invoke(cli(), ["skills", "list", "-j"])
    assert listed.exit_code == 0
    assert len(json.loads(listed.stdout)["rejected"]) == 1
    validated = CliRunner().invoke(cli(), ["skills", "validate", "-j"])
    assert validated.exit_code == 2
    assert not (tmp_path / "outputs").exists()


def test_doctor_always_checks_connectivity(tmp_path, monkeypatch):
    from skillrunner.cli import inspection

    monkeypatch.chdir(tmp_path)
    called = []

    async def check(settings, environ):
        called.append(True)
        return {"status": "ok", "model": "test"}

    monkeypatch.setattr(inspection, "check_model", check)
    result = CliRunner().invoke(cli(), ["doctor", "-j"])
    assert result.exit_code == 0, result.output
    assert called == [True]
    assert json.loads(result.stdout)["model"]["status"] == "ok"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--timeout", "2"),
        ("--timeout", "0s"),
        ("--shutdown-grace", "-1s"),
        ("--max-steps", "0"),
        ("--max-tokens", "-1"),
        ("--max-tool-calls", "0"),
    ],
)
def test_invalid_limits_before_bundle(option, value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli(), ["run", "task", option, value, "-j"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "invalid_request"
    assert not (tmp_path / "outputs").exists()


def test_task_chatter_is_only_stderr(tmp_path, monkeypatch):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)

    async def fake(request, settings, *, environ):
        print("tool chatter")
        return {"status": "succeeded", "exit_code": 0}

    monkeypatch.setattr(execution, "run_task", fake)
    result = CliRunner().invoke(cli(), ["run", "task", "-j"])
    assert json.loads(result.stdout)["status"] == "succeeded"
    assert "tool chatter" in result.stderr


def test_missing_explicit_validation_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli(), ["skills", "validate", "missing", "-j"])
    assert result.exit_code == 2


def test_invalid_utf8_stdin_returns_argument_error_before_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli(), ["run", "--prompt-stdin", "-j"], input=b"\xff")
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "invalid_request"
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("quiet_flag", ["-q", "--quiet"])
def test_quiet_suppresses_task_progress_but_keeps_diagnostics_and_receipt(
    tmp_path, monkeypatch, quiet_flag
):
    import sys

    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)

    async def fake(request, settings, *, environ):
        print("progress-marker")
        print("diagnostic-marker", file=sys.stderr)
        return {
            "status": "blocked",
            "exit_code": 4,
            "errors": [{"code": "missing_dependency", "message": "Install the required runtime."}],
        }

    monkeypatch.setattr(execution, "run_task", fake)
    result = CliRunner().invoke(cli(), ["run", "task", quiet_flag, "-j"])
    assert result.exit_code == 4
    assert json.loads(result.stdout)["status"] == "blocked"
    assert "progress-marker" not in result.stderr
    assert "diagnostic-marker" in result.stderr
    assert "missing_dependency: Install the required runtime." in result.stderr
    assert len(result.stdout.splitlines()) == 1


@pytest.mark.parametrize("short", [True, False])
def test_path_model_and_output_aliases_forward_resolved_controls(tmp_path, monkeypatch, short):
    from skillrunner.cli import execution

    monkeypatch.chdir(tmp_path)
    (tmp_path / "team").mkdir()
    (tmp_path / "team/config.toml").write_text('skills_dir="ignored"\noutput_dir="ignored"\n')
    (tmp_path / "team/policy.toml").write_text("allowed_executables=[]\nallowed_env=[]\n")
    observed = []

    async def fake(request, settings, *, environ):
        observed.append((request, settings))
        return {"status": "succeeded", "exit_code": 0}

    monkeypatch.setattr(execution, "run_task", fake)
    choices = [
        ("-S", "--skills-dir", "catalog"),
        ("-O", "--output-dir", "bundles"),
        ("-o", "--output", "answer.json"),
        ("-f", "--format", "json"),
        ("-m", "--model", "literal-model"),
        ("-b", "--base-url", "https://example.invalid/custom/v1"),
        ("-c", "--config", "team/config.toml"),
    ]
    arguments = ["run", "Produce JSON", "--overwrite", "--policy", "team/policy.toml", "-j"]
    for alias, long_name, value in choices:
        arguments.extend([alias if short else long_name, value])
    result = CliRunner().invoke(cli(), arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "succeeded"
    ((request, settings),) = observed
    assert request.output == tmp_path / "answer.json"
    assert request.format == "json" and request.overwrite is True
    assert settings.skills_dir == tmp_path / "catalog"
    assert settings.output_dir == tmp_path / "bundles"
    assert settings.selected_model == "literal-model"
    assert settings.base_url == "https://example.invalid/custom/v1"
    assert settings.policy.allowed_executables == []
    assert settings.sources["skills_dir"] == settings.sources["output_dir"] == "cli"
