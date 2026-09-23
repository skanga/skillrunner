"""Explicit CLI > environment > one TOML file > defaults resolution."""

import os
import shutil
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from skillrunner.config.models import (
    FileSettings,
    ModelProfile,
    Policy,
    ResolvedSettings,
    endpoint,
)
from skillrunner.domain.errors import RunnerError

LIMITS = {
    "timeout",
    "shutdown_grace",
    "max_steps",
    "max_tool_calls",
    "max_tokens",
    "model_transport_retries",
}
ENVIRONMENT = {
    **{f"SKILLRUN_{name.upper()}": name for name in LIMITS},
    "SKILLRUN_SKILLS_DIR": "skills_dir",
    "SKILLRUN_OUTPUT_DIR": "output_dir",
    "OPENAI_BASE_URL": "base_url",
}


class ExplicitEnvironment(BaseSettings):
    """Only supplied, documented environment values; no process or dotenv reads."""

    model_config = SettingsConfigDict(extra="forbid")
    values: dict[str, str]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (init_settings,)


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _origins(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_origins(value, name))
        else:
            result[name] = "file"
    return result


def _command(value: str, base: Path, environ: Mapping[str, str], cwd: Path) -> str | None:
    if "/" in value or "\\" in value or Path(value).is_absolute():
        candidate = _path(value, base)
    else:
        # PATH entries belong to the invocation environment, not the config file.
        search_path = os.pathsep.join(
            str(_path(entry, cwd)) for entry in environ.get("PATH", os.defpath).split(os.pathsep)
        )
        found = shutil.which(value, path=search_path)
        if found is None:
            return None
        candidate = Path(found).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return None
    return str(candidate)


def resolve_settings(
    cwd: Path, overrides: dict[str, Any], environ: Mapping[str, str]
) -> ResolvedSettings:
    """Resolve once, recording file origins and immutable executable identities."""
    try:
        return _resolve(cwd.resolve(), overrides, environ)
    except RunnerError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        # Pydantic and parser messages can contain credential-bearing input values.
        raise RunnerError(
            "invalid_configuration",
            "Invalid configuration; check known keys, value types, paths, and model settings.",
        ) from None


def _resolve(cwd: Path, overrides: dict[str, Any], environ: Mapping[str, str]) -> ResolvedSettings:
    supported = LIMITS | {"config", "policy", "skills_dir", "output_dir", "model", "base_url"}
    if overrides.keys() - supported:
        raise ValueError("Unknown override")
    explicit = overrides.get("config")
    config_path = _path(explicit, cwd) if explicit is not None else cwd / "skillrun.toml"
    exists = config_path.exists()
    if explicit is not None and not exists:
        raise ValueError("Missing config")
    data = _read(config_path) if exists else {}
    base = config_path.parent if exists else cwd
    sources = _origins(data)
    # Check the original document even when a later source replaces a bad value.
    initial = dict(data)
    _prepare_mcp_paths(initial, base)
    FileSettings.model_validate(initial)
    policy_base = base
    if overrides.get("policy") is not None:
        policy_path = _path(overrides["policy"], cwd)
        data["policy"] = _read(policy_path)
        policy_base = policy_path.parent
        sources = {k: v for k, v in sources.items() if not k.startswith("policy.")}
        sources.update({k: "policy" for k in _origins(data["policy"], "policy")})
    selected_model = overrides.get("model")
    base_url: str | None = None
    values = ExplicitEnvironment(
        values={key: value for key, value in environ.items() if key in ENVIRONMENT}
    ).values
    for origin, entries in (
        ("env", {ENVIRONMENT[key]: value for key, value in values.items()}),
        ("cli", {key: value for key, value in overrides.items() if value is not None}),
    ):
        for name, value in entries.items():
            if name in LIMITS:
                if origin == "env" and name not in {"timeout", "shutdown_grace"}:
                    if not value.isascii() or not value.isdecimal():
                        raise ValueError("Invalid integer environment limit")
                    value = int(value)
                data.setdefault("limits", {})[name] = value
                sources[f"limits.{name}"] = origin
            elif name in {"skills_dir", "output_dir"}:
                data[name] = _path(value, cwd)
                sources[name] = origin
            elif name == "base_url":
                base_url = endpoint(value)
                sources[name] = origin
    for name, default in (("skills_dir", "skills"), ("output_dir", "outputs")):
        data[name] = _path(data.get(name, default), base if name in sources else cwd)
    _prepare_mcp_paths(data, base)
    settings = ResolvedSettings.model_validate(
        {
            **data,
            "config_path": config_path if exists else None,
            "selected_model": selected_model,
            "base_url": base_url,
            "sources": sources,
        }
    )
    policy = settings.policy
    if policy.executable_identities or policy.unresolved_executables:
        raise ValueError("Executable identities are internally computed")
    canonical: list[str] = []
    for command in policy.allowed_executables:
        resolved = _command(command, policy_base, environ, cwd)
        if resolved is None:
            policy.unresolved_executables.append(command)
            continue
        stat = Path(resolved).stat()
        policy.executable_identities[resolved] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )
        if resolved not in canonical:
            canonical.append(resolved)
    policy.allowed_executables = canonical
    policy.command_env = {
        _command(command, policy_base, environ, cwd) or command: references
        for command, references in policy.command_env.items()
    }
    for server in settings.mcp.values():
        if server.command:
            server.command = _command(server.command, base, environ, cwd) or server.command
    for validator in (
        *settings.artifacts.validators.values(),
        *settings.acceptance.checks.values(),
    ):
        validator.command = _command(validator.command, base, environ, cwd) or validator.command
    defaults = FileSettings().model_dump()
    for key in _origins(defaults):
        settings.sources.setdefault(key, "default")
    return settings


def _prepare_mcp_paths(data: dict[str, Any], base: Path) -> None:
    servers = data.get("mcp", {})
    if isinstance(servers, dict):
        for server in servers.values():
            if isinstance(server, dict) and server.get("cwd") is not None:
                server["cwd"] = _path(server["cwd"], base)


def select_model(settings: ResolvedSettings) -> ModelProfile:
    if settings.base_url is not None:
        if not settings.selected_model:
            raise RunnerError(
                "invalid_arguments", "Direct endpoints require --model with a literal model ID."
            )
        try:
            return ModelProfile.model_validate(
                {
                    **settings.direct_model.model_dump(),
                    "base_url": settings.base_url,
                    "model": settings.selected_model,
                }
            )
        except ValidationError:
            raise RunnerError("invalid_configuration", "Invalid direct model settings.") from None
    alias = settings.selected_model or settings.default_model
    if alias is None or alias not in settings.models:
        raise RunnerError(
            "invalid_arguments", "Select a configured model alias using --model or default_model."
        )
    return settings.models[alias]


def verify_executable(policy: Policy, command: str) -> str:
    """Check a previously resolved identity immediately before process creation."""
    expected = policy.executable_identities.get(command)
    if expected is None:
        if command in policy.unresolved_executables:
            raise RunnerError(
                "missing_dependency", "Install the configured executable and restart the run."
            )
        raise RunnerError(
            "command_not_allowed", "Configure this executable in the command allowlist."
        )
    try:
        stat = Path(command).stat()
        observed = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if observed != expected or not os.access(command, os.X_OK):
            raise OSError("Changed executable")
    except OSError:
        raise RunnerError(
            "missing_dependency",
            "The allowed executable changed or disappeared; restart after verifying it.",
        ) from None
    return command
