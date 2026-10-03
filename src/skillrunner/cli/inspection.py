"""Read-only package inspection and explicit model connectivity diagnostics."""

import asyncio
import json
import os
import platform
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from skillrunner.artifacts.validation import MEDIA_TYPES
from skillrunner.catalog.discovery import discover
from skillrunner.cli.receipts import fail
from skillrunner.config.models import ResolvedSettings
from skillrunner.config.network import tls_verify
from skillrunner.config.secrets import resolve_api_key
from skillrunner.config.sources import resolve_settings, select_model
from skillrunner.domain.errors import RunnerError
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.recording.redaction import Redactor
from skillrunner.runtime.budgets import UsageLedger, estimate_text_tokens
from skillrunner.runtime.preflight import check_dependencies, check_format, error_record


def catalog_command(
    path: Path | None,
    skills_dir: Path | None,
    config: Path | None,
    json_mode: bool,
    *,
    validate: bool,
) -> None:
    try:
        settings = resolve_settings(
            Path.cwd(), {"skills_dir": skills_dir, "config": config}, dict(os.environ)
        )
        root = (Path.cwd() / path).resolve() if path is not None else settings.skills_dir
        if validate and not root.is_dir():
            raise RunnerError(
                "invalid_arguments", "Choose an existing skill package or catalog directory."
            )
        catalog = discover(root, max_instruction_bytes=settings.storage.max_package_bytes)
        result = {
            "status": "invalid_request" if validate and catalog.rejected else "succeeded",
            "skills": [
                {"name": item.name, "description": item.description, "path": str(item.root)}
                for item in catalog.skills.values()
            ],
            "rejected": [{**asdict(item), "path": str(item.path)} for item in catalog.rejected],
        }
    except RunnerError as error:
        fail(error, json_mode)
    if json_mode:
        typer.echo(json.dumps(result, ensure_ascii=False))
    else:
        if not catalog.skills:
            typer.echo(f"No skills found in {root}. Run skillrun init or use --skills-dir.")
        for skill in catalog.skills.values():
            typer.echo(f"{skill.name}: {skill.description}")
        for rejected in catalog.rejected:
            typer.echo(f"Rejected {rejected.path}: {rejected.message}", err=True)
        if validate:
            typer.echo(
                f"Validated {len(catalog.skills)} package(s); {len(catalog.rejected)} rejected."
            )
    raise typer.Exit(2 if validate and catalog.rejected else 0)


def config_command(overrides: dict[str, Any], json_mode: bool) -> None:
    try:
        settings = resolve_settings(Path.cwd(), overrides, dict(os.environ))
        try:
            profile = select_model(settings)
            model: dict[str, Any] = profile.model_dump(mode="json")
        except RunnerError as error:
            model = {"status": "unconfigured", "message": error.message}
        result = Redactor().clean(
            {
                "config_path": str(settings.config_path) if settings.config_path else None,
                "skills_dir": str(settings.skills_dir),
                "output_dir": str(settings.output_dir),
                "model": model,
                "limits": settings.limits.model_dump(),
                "storage": settings.storage.model_dump(),
                "policy": settings.policy.model_dump(),
                "diagnostics": settings.diagnostics.model_dump(),
                "acceptance": settings.acceptance.model_dump(),
                "artifacts": settings.artifacts.model_dump(),
                "image_inspector": settings.image_inspector,
                "mcp": {
                    name: server.model_dump(mode="json") for name, server in settings.mcp.items()
                },
                "sources": settings.sources,
                "warnings": settings.warnings,
                "selection_mode": "direct" if settings.base_url else "alias",
                "supported_formats": sorted(
                    set(MEDIA_TYPES) | settings.artifacts.validators.keys()
                ),
                "note": (
                    "Credentials are references, never resolved here. Token estimates use UTF-8 "
                    "bytes. Prompts and outputs persist even with content logging disabled."
                ),
            }
        )
    except RunnerError as error:
        fail(error, json_mode)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=None if json_mode else 2))


async def check_model(settings: ResolvedSettings, environ: Mapping[str, str]) -> dict[str, Any]:
    """Discovery never substitutes for a real, token-budgeted completion request."""
    profile = select_model(settings)
    key = resolve_api_key(profile, environ)
    adapter = OpenAICompatibleAdapter(profile, api_key=key)
    deadline = asyncio.get_running_loop().time() + min(settings.limits.timeout, 30.0)
    try:
        resolved = await adapter.discover_capabilities(deadline)
        profile = resolved.profile
        adapter.profile = profile
        if profile.context_window_tokens is None or profile.max_output_tokens is None:
            raise RunnerError(
                "unsupported_capability",
                "Configure or discover model context and output token capacities.",
            )
        messages = [
            {
                "role": "user",
                "content": "Call skillrun_probe exactly once with ok=true. Do not reply with text.",
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "skillrun_probe",
                    "description": "Confirm tool-calling support; no side effects.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        ledger = UsageLedger(
            max_steps=settings.limits.max_steps,
            max_tool_calls=settings.limits.max_tool_calls,
            max_tokens=settings.limits.max_tokens,
        )
        reservation = ledger.reserve_model(
            input_tokens=estimate_text_tokens(messages, tools),
            context_window=profile.context_window_tokens,
            max_output=min(profile.max_output_tokens, 1024),
        )
        reply = await adapter.complete(messages, tools, reservation.output_limit, deadline)
        if (
            reply.finish_reason != "tool_calls"
            or len(reply.tool_calls) != 1
            or reply.tool_calls[0].name != "skillrun_probe"
            or reply.tool_calls[0].arguments != {"ok": True}
            or type(reply.tool_calls[0].arguments.get("ok")) is not bool
        ):
            raise RunnerError(
                "unsupported_capability",
                "Connectivity works, but the tool-calling probe did not complete correctly.",
                details={
                    "suggested_action": (
                        "Use a tool-capable model; check its output allowance and request options."
                    )
                },
            )
        return {
            "status": "ok",
            "model": profile.model,
            "authentication": profile.auth_mode,
            "connectivity": "verified",
            "tool_calling": "verified",
        }
    finally:
        await adapter.aclose()


def doctor_command(overrides: dict[str, Any], json_mode: bool, *, network: bool = False) -> None:
    environ = dict(os.environ)
    try:
        settings = resolve_settings(Path.cwd(), overrides, environ)
        # Presence checks do not execute a binary and do not require an allowlist entry.
        runtimes = {}
        for name in ("python", "python3", "node", "sh"):
            found = shutil.which(name, path=environ.get("PATH", os.defpath))
            resolved = str(Path(found).resolve()) if found else None
            runtimes[name] = {
                "path": resolved,
                "allowed": resolved in settings.policy.allowed_executables,
            }
        checks: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        failures: list[RunnerError] = []

        def check(name: str, operation: Any) -> Any:
            try:
                value = operation()
                checks.append({"name": name, "status": "ok"})
                return value
            except RunnerError as error:
                checks.append({"name": name, "status": "failed", **error_record(error)})
                errors.append(error_record(error))
                failures.append(error)
                return None

        def catalog_check() -> None:
            catalog = discover(
                settings.skills_dir, max_instruction_bytes=settings.storage.max_package_bytes
            )
            if not catalog.skills or catalog.rejected:
                raise RunnerError(
                    "invalid_arguments",
                    f"Catalog {settings.skills_dir}: {len(catalog.skills)} valid, "
                    f"{len(catalog.rejected)} rejected.",
                    details={"suggested_action": "Run skillrun skills validate or skillrun init."},
                )

        def output_check() -> None:
            parent = settings.output_dir
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK):
                raise RunnerError(
                    "invalid_arguments",
                    "Bundle parent is not a writable directory.",
                    details={
                        "suggested_action": "Choose a writable --output-dir outside input trees."
                    },
                )

        def capacities_check() -> None:
            if profile.discovery is None and (
                profile.context_window_tokens is None or profile.max_output_tokens is None
            ):
                raise RunnerError(
                    "invalid_configuration",
                    "Configure context_window_tokens and max_output_tokens, or explicit discovery.",
                )
            if (
                profile.context_window_tokens
                and profile.max_output_tokens
                and profile.max_output_tokens > profile.context_window_tokens
            ):
                raise RunnerError(
                    "invalid_configuration", "Maximum output exceeds context capacity."
                )
            tls_verify(profile.ca_bundle)

        check("catalog", catalog_check)
        check("output_directory", output_check)
        profile = check("model_configuration", lambda: select_model(settings))
        if profile is not None:
            check("credentials", lambda: resolve_api_key(profile, environ))
            check("model_capacities_and_tls", capacities_check)
        check("dependencies", lambda: check_dependencies(settings, environ))
        for format_name in settings.artifacts.validators:
            check(
                f"validator.{format_name}", lambda f=format_name: check_format(settings, f, environ)
            )
        model = {
            "status": "not_checked",
            "connectivity": "not_checked",
            "tool_calling": "not_checked",
        }
        if network:
            model = (
                check("model_network", lambda: asyncio.run(check_model(settings, environ))) or model
            )
        result: dict[str, Any] = {
            "status": failures[0].status if failures else "succeeded",
            "exit_code": failures[0].exit_code if failures else 0,
            "checks": checks,
            "errors": errors,
            "warnings": settings.warnings,
            "supported_formats": sorted(set(MEDIA_TYPES) | settings.artifacts.validators.keys()),
            "model": model,
            "environment": {
                "platform": platform.system(),
                "coordinator_python": platform.python_version(),
                "runtimes": runtimes,
                "unresolved_allowed_executables": settings.policy.unresolved_executables,
            },
            "note": (
                "Only runtimes required by a selected skill are mandatory. "
                "Host execution is not sandboxed."
            ),
        }
    except RunnerError as error:
        fail(error, json_mode)
    if json_mode:
        typer.echo(json.dumps(result, ensure_ascii=False))
    else:
        for item in checks:
            typer.echo(f"{item['name']}: {item['status']}")
        typer.echo(f"Model connectivity: {model['status']} (use --network to probe tools)")
        for name, info in runtimes.items():
            typer.echo(f"{name}: {info['path'] or 'not found'}; allowed={info['allowed']}")
        for name in settings.policy.unresolved_executables:
            typer.echo(f"Warning: configured executable not found: {name}", err=True)
        for warning in settings.warnings:
            typer.echo(f"Warning: {warning}", err=True)
        typer.echo(result["note"])
    for item_error in errors:
        typer.echo(
            f"{item_error['code']}: {item_error['message']}\n"
            f"Next action: {item_error['suggested_action']}",
            err=True,
        )
    raise typer.Exit(int(result["exit_code"]))
