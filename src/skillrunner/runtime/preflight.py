"""Offline capability checks shared by diagnostics and task admission."""

from collections.abc import Mapping
from typing import Any

from skillrunner.artifacts.validation import MEDIA_TYPES
from skillrunner.config.models import ResolvedSettings
from skillrunner.config.network import tls_verify
from skillrunner.config.secrets import resolve_api_key
from skillrunner.config.sources import verify_executable
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.environment import build_child_environment


def check_format(settings: ResolvedSettings, format_name: str, environ: Mapping[str, str]) -> None:
    if format_name in MEDIA_TYPES:
        return
    validator = settings.artifacts.validators.get(format_name)
    if validator is None:
        raise RunnerError(
            "unsupported_capability",
            f"Output format {format_name} requires a configured validator.",
            details={
                "format": format_name,
                "suggested_action": (
                    f"Configure artifacts.validators.{format_name} and allow its executable, "
                    "or choose a supported format with --format."
                ),
            },
        )
    verify_executable(settings.policy, validator.command)
    build_child_environment(
        environ, references=settings.policy.command_env.get(validator.command, {})
    )


def check_dependencies(settings: ResolvedSettings, environ: Mapping[str, str]) -> None:
    for checker in settings.acceptance.checks.values():
        verify_executable(settings.policy, checker.command)
        build_child_environment(
            environ, references=settings.policy.command_env.get(checker.command, {})
        )
    if settings.image_inspector is not None:
        check_format(settings, "png", environ)
        profile = settings.models[settings.image_inspector]
        resolve_api_key(profile, environ)
        tls_verify(profile.ca_bundle)
    for server in settings.mcp.values():
        tls_verify(server.ca_bundle)
        if server.command:
            verify_executable(settings.policy, server.command)
            build_child_environment(environ, references=server.env)
        for reference in server.headers.values():
            if not environ.get(reference):
                raise RunnerError(
                    "missing_credential",
                    f"Required MCP environment reference {reference} is missing.",
                    details={"suggested_action": f"Set {reference} and retry."},
                )


def error_record(error: RunnerError) -> dict[str, Any]:
    return {
        "code": error.code,
        "message": error.message,
        "details": error.details,
        "suggested_action": error.details.get(
            "suggested_action", "Inspect configuration with skillrun config show."
        ),
    }
