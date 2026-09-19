"""Configuration contracts: real files, isolated environments, no model calls."""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def api():
    # Fail at the missing feature, rather than aborting test collection on import.
    assert importlib.util.find_spec("skillrunner.config.sources") is not None, (
        "Configuration resolution has not been implemented"
    )
    return importlib.import_module("skillrunner.config.sources")


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults(api, tmp_path):
    settings = api.resolve_settings(tmp_path, {}, {})
    assert settings.skills_dir == tmp_path / "skills"
    assert settings.output_dir == tmp_path / "outputs"
    assert settings.limits.timeout == 600
    assert settings.limits.max_steps == 40
    assert settings.limits.max_tool_calls == 100
    assert settings.limits.max_tokens == 100_000
    assert settings.limits.shutdown_grace == 5
    assert settings.policy.allowed_executables == []
    assert settings.storage.model_dump() == {
        "max_input_files": 10_000,
        "max_input_bytes": 1_073_741_824,
        "max_package_files": 20_000,
        "max_package_bytes": 536_870_912,
        "max_artifact_bytes": 2_147_483_648,
        "max_scratch_bytes": 4_294_967_296,
        "max_tool_output_bytes": 1_048_576,
        "max_event_log_bytes": 10_485_760,
        "max_read_bytes": 65_536,
        "max_archive_expanded_bytes": 268_435_456,
    }


@pytest.mark.parametrize(
    "field", ["timeout", "max_steps", "max_tool_calls", "max_tokens", "shutdown_grace"]
)
def test_all_five_limits_obey_precedence(api, tmp_path, field):
    duration = field in {"timeout", "shutdown_grace"}
    a, b, c = ('"1s"', "2s", "3s") if duration else ("1", "2", 3)
    write_config(tmp_path / "skillrun.toml", f"[limits]\n{field} = {a}\n")
    env = {f"SKILLRUN_{field.upper()}": b}
    assert getattr(api.resolve_settings(tmp_path, {}, {}).limits, field) == 1
    assert getattr(api.resolve_settings(tmp_path, {}, env).limits, field) == 2
    result = api.resolve_settings(tmp_path, {field: c}, env)
    assert getattr(result.limits, field) == 3
    assert result.sources[f"limits.{field}"] == "cli"


@pytest.mark.parametrize("value,seconds", [("500ms", 0.5), ("1.5m", 90), ("1h", 3600)])
def test_duration_units(api, tmp_path, value, seconds):
    assert api.resolve_settings(tmp_path, {"timeout": value}, {}).limits.timeout == seconds
    assert api.resolve_settings(tmp_path, {"shutdown_grace": "0s"}, {}).limits.shutdown_grace == 0


@pytest.mark.parametrize("value", ["0s", "5", 5, "-1s", "NaN", "inf", "1e99h", True])
def test_invalid_duration_rejected(api, tmp_path, value):
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {"timeout": value}, {})


@pytest.mark.parametrize("value", ['"1 GiB"', "1.5", "true", "-1"])
def test_storage_requires_nonnegative_integer_bytes(api, tmp_path, value):
    write_config(tmp_path / "skillrun.toml", f"[storage]\nmax_input_bytes = {value}\n")
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {}, {})


def test_path_sources_and_explicit_config_replacement(api, tmp_path):
    write_config(tmp_path / "skillrun.toml", "[limits]\nmax_steps = 3\n")
    custom = write_config(
        tmp_path / "team/config.toml", 'skills_dir = "packages"\noutput_dir = "results"\n'
    )
    resolved = api.resolve_settings(tmp_path, {"config": custom}, {})
    assert resolved.skills_dir == tmp_path / "team/packages"
    assert resolved.output_dir == tmp_path / "team/results"
    assert resolved.limits.max_steps == 40
    resolved = api.resolve_settings(
        tmp_path, {"config": custom, "skills_dir": "cli"}, {"SKILLRUN_OUTPUT_DIR": "env"}
    )
    assert resolved.skills_dir == tmp_path / "cli"
    assert resolved.output_dir == tmp_path / "env"


def test_no_parent_or_dotenv_discovery(api, tmp_path):
    write_config(tmp_path / "skillrun.toml", "[limits]\nmax_steps = 3\n")
    child = tmp_path / "child"
    child.mkdir()
    write_config(child / ".env", "SKILLRUN_MAX_STEPS=9\n")
    assert api.resolve_settings(child, {}, {}).limits.max_steps == 40


@pytest.mark.parametrize(
    "text",
    [
        "nonsense = 1",
        "[limits]\nmax_tokens = 0",
        "schema_version = 2",
        "[limits]\nmax_steps = true",
        "[broken",
    ],
)
def test_bad_configuration_rejected(api, tmp_path, text):
    write_config(tmp_path / "skillrun.toml", text)
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {}, {})


def test_missing_explicit_config_rejected(api, tmp_path):
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {"config": "absent.toml"}, {})


def test_policy_replaces_all_embedded_fields(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        f"[policy]\nallowed_executables = {json.dumps([sys.executable])}\n"
        'allowed_env = ["EXAMPLE_TOKEN"]\n',
    )
    replacement = write_config(tmp_path / "policy.toml", "allowed_executables = []\n")
    settings = api.resolve_settings(tmp_path, {"policy": replacement}, {})
    assert settings.policy.allowed_executables == []
    assert settings.policy.allowed_env == []


def test_policy_relative_executables_use_policy_parent(api, tmp_path):
    executable = tmp_path / "team/bin/script"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    policy = write_config(tmp_path / "team/policy.toml", 'allowed_executables = ["./bin/script"]\n')
    settings = api.resolve_settings(tmp_path, {"policy": policy}, {})
    assert settings.policy.allowed_executables == [str(executable.resolve())]


def test_model_alias_selection_and_explicit_no_auth(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        """default_model = "local"
[models.local]
base_url = "http://localhost:9999/custom/v1"
model = "arbitrary/id"
auth_mode = "none"
context_window_tokens = 32000
max_output_tokens = 4000
""",
    )
    settings = api.resolve_settings(tmp_path, {}, {"OPENAI_API_KEY": "do-not-use"})
    profile = api.select_model(settings)
    assert profile.model == "arbitrary/id"
    assert profile.base_url == "http://localhost:9999/custom/v1"
    secrets = importlib.import_module("skillrunner.config.secrets")
    assert secrets.resolve_api_key(profile, {"OPENAI_API_KEY": "do-not-use"}) is None
    assert "do-not-use" not in repr(settings)


def test_direct_model_does_not_borrow_alias_credentials(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        """default_model = "same"
[models.same]
base_url = "https://named.example/v1"
model = "remote"
api_key_env = "NAMED_KEY"
[direct_model]
auth_mode = "none"
context_window_tokens = 4000
max_output_tokens = 500
""",
    )
    profile = api.select_model(
        api.resolve_settings(
            tmp_path, {"base_url": "http://local.test/prefix", "model": "same"}, {}
        )
    )
    assert profile.model == "same"
    assert profile.base_url == "http://local.test/prefix"
    assert profile.api_key_env is None
    assert profile.auth_mode == "none"


def test_direct_model_requires_literal_cli_model(api, tmp_path):
    settings = api.resolve_settings(tmp_path, {}, {"OPENAI_BASE_URL": "http://local.test/v1"})
    with pytest.raises(ValueError, match="invalid_arguments"):
        api.select_model(settings)


@pytest.mark.parametrize(
    "option",
    [
        "model",
        "messages",
        "tools",
        "api_key",
        "base_url",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "extra_body",
        "extra_headers",
    ],
)
def test_inference_options_cannot_override_contract(api, tmp_path, option):
    write_config(
        tmp_path / "skillrun.toml",
        f"""[models.bad]
base_url = "http://localhost/v1"
model = "bad"
auth_mode = "none"
[models.bad.request_options]
{option} = "forbidden"
""",
    )
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {}, {})


def test_secret_errors_do_not_include_secret_values(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        '[models.bad]\nbase_url = "ftp://user:secret@example.org"\nmodel = "x"\n',
    )
    with pytest.raises(ValueError) as error:
        api.resolve_settings(tmp_path, {}, {})
    assert "secret" not in str(error.value)


def test_missing_named_secret_does_not_use_openai_fallback(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        """default_model = "one"
[models.one]
base_url = "https://example.org/v1"
model = "one"
api_key_env = "SPECIFIC_KEY"
""",
    )
    profile = api.select_model(api.resolve_settings(tmp_path, {}, {}))
    secrets = importlib.import_module("skillrunner.config.secrets")
    with pytest.raises(ValueError, match="missing_credential") as caught:
        secrets.resolve_api_key(profile, {"OPENAI_API_KEY": "must-not-fallback"})
    assert caught.value.details["credential_reference"] == "SPECIFIC_KEY"
    assert caught.value.details["suggested_action"] == "Set SPECIFIC_KEY and rerun."
    assert "must-not-fallback" not in str(caught.value)


def test_executable_identity_changes_block_launch(api, tmp_path):
    executable = tmp_path / "tool"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    write_config(tmp_path / "skillrun.toml", '[policy]\nallowed_executables = ["./tool"]\n')
    settings = api.resolve_settings(tmp_path, {}, {})
    assert api.verify_executable(settings.policy, str(executable)) == str(executable)
    executable.write_text("#!/bin/sh\necho changed\n")
    with pytest.raises(ValueError, match="missing_dependency"):
        api.verify_executable(settings.policy, str(executable))


def test_missing_executable_deferred_until_required(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml", '[policy]\nallowed_executables = ["definitely-absent-tool"]\n'
    )
    settings = api.resolve_settings(tmp_path, {}, {"PATH": ""})
    assert settings.policy.allowed_executables == []
    with pytest.raises(ValueError, match="missing_dependency"):
        api.verify_executable(settings.policy, "definitely-absent-tool")


@pytest.mark.parametrize("value", ["true", "1.0"])
def test_schema_version_requires_exact_integer(api, tmp_path, value):
    write_config(tmp_path / "skillrun.toml", f"schema_version = {value}\n")
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.resolve_settings(tmp_path, {}, {})


def test_ambient_environment_is_never_implicitly_read(api, tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLRUN_MAX_STEPS", "999")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://ambient.invalid")
    settings = api.resolve_settings(tmp_path, {}, {})
    assert settings.limits.max_steps == 40
    assert settings.base_url is None


def test_mcp_path_is_relative_to_config_and_does_not_grant_command(api, tmp_path):
    config = write_config(
        tmp_path / "team/config.toml",
        """[mcp.local]
transport = "stdio"
command = "missing-server"
cwd = "working"
allowed_tools = ["lookup"]
""",
    )
    settings = api.resolve_settings(tmp_path, {"config": config}, {})
    assert settings.mcp["local"].cwd == tmp_path / "team/working"
    assert settings.policy.allowed_executables == []


@pytest.mark.parametrize(
    "text",
    [
        '[mcp.bad]\ntransport="stdio"\nurl="http://example.org"',
        '[mcp.bad]\ntransport="streamable-http"\nurl="http://example.org"\ncommand="bad"',
        '[artifacts.validators.pdf]\ncommand="check"\nargs=[]',
        '[models.bad]\nbase_url="https://user:private@example.org"\nmodel="x"',
    ],
)
def test_connection_and_validator_schema_rejects_invalid_combinations(api, tmp_path, text):
    write_config(tmp_path / "skillrun.toml", text)
    with pytest.raises(ValueError, match="invalid_configuration") as error:
        api.resolve_settings(tmp_path, {}, {})
    assert "private" not in str(error.value)


def test_secret_reference_fallback_and_masked_representation(api, tmp_path):
    write_config(
        tmp_path / "skillrun.toml",
        """default_model="a"
[models.a]
base_url="https://example.org"
model="a"
""",
    )
    profile = api.select_model(api.resolve_settings(tmp_path, {}, {}))
    secrets = importlib.import_module("skillrunner.config.secrets")
    key = secrets.resolve_api_key(profile, {"OPENAI_API_KEY": "private-value"})
    assert key.get_secret_value() == "private-value"
    assert "private-value" not in repr(key)


@pytest.mark.parametrize("path_entry", ["bin", ""])
def test_path_search_uses_invocation_directory(api, tmp_path, path_entry):
    name = "tool.exe" if os.name == "nt" else "tool"
    executable = tmp_path / path_entry / name
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    config = write_config(
        tmp_path / "team/config.toml", f'[policy]\nallowed_executables=["{name}"]\n'
    )
    settings = api.resolve_settings(tmp_path, {"config": config}, {"PATH": path_entry})
    assert settings.policy.allowed_executables == [str(executable)]


@pytest.mark.parametrize(
    "section",
    [
        '[mcp.remote]\ntransport="streamable-http"\nurl="https://example.org"\n[mcp.remote.headers]\nAuthorization',
        '[mcp.local]\ntransport="stdio"\ncommand="server"\n[mcp.local.env]\nTOKEN',
        "[policy.command_env.tool]\nTOKEN",
    ],
)
def test_child_and_mcp_secret_fields_accept_only_references(api, tmp_path, section):
    write_config(tmp_path / "skillrun.toml", section + '="Bearer private-token"\n')
    with pytest.raises(ValueError, match="invalid_configuration") as error:
        api.resolve_settings(tmp_path, {}, {})
    assert "private-token" not in str(error.value)
