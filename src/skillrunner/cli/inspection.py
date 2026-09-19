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

from skillrunner.catalog.discovery import discover
from skillrunner.cli.receipts import fail
from skillrunner.config.models import ResolvedSettings
from skillrunner.config.secrets import resolve_api_key
from skillrunner.config.sources import resolve_settings, select_model
from skillrunner.domain.errors import RunnerError
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.runtime.budgets import UsageLedger, estimate_text_tokens


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
        for skill in catalog.skills.values():
            typer.echo(f"{skill.name}: {skill.description}")
        for rejected in catalog.rejected:
            typer.echo(f"Rejected {rejected.path}: {rejected.message}", err=True)
        if validate:
            typer.echo(
                f"Validated {len(catalog.skills)} package(s); {len(catalog.rejected)} rejected."
            )
    raise typer.Exit(2 if validate and catalog.rejected else 0)


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
        messages = [{"role": "user", "content": "Reply OK."}]
        ledger = UsageLedger(
            max_steps=settings.limits.max_steps,
            max_tool_calls=settings.limits.max_tool_calls,
            max_tokens=settings.limits.max_tokens,
        )
        reservation = ledger.reserve_model(
            input_tokens=estimate_text_tokens(messages, []),
            context_window=profile.context_window_tokens,
            max_output=min(profile.max_output_tokens, 8),
        )
        await adapter.complete(messages, [], reservation.output_limit, deadline)
        return {
            "status": "ok",
            "model": profile.model,
            "authentication": profile.auth_mode,
            "connectivity": "verified",
        }
    finally:
        await adapter.aclose()


def doctor_command(overrides: dict[str, Any], json_mode: bool) -> None:
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
        model = asyncio.run(check_model(settings, environ))
        result = {
            "status": "succeeded",
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
        typer.echo(f"Model connectivity: {model['status']}")
        for name, info in runtimes.items():
            typer.echo(f"{name}: {info['path'] or 'not found'}; allowed={info['allowed']}")
        typer.echo(result["note"])
