"""Small, packaged starter templates; no dependencies, downloads, or overwrites."""

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from skillrunner.cli.receipts import fail
from skillrunner.config.models import ModelProfile
from skillrunner.domain.errors import RunnerError

SKILL = """---
name: summarize
description: Summarize supplied notes into a concise Markdown document.
---
Read the supplied notes completely, following pagination when necessary.
Write a concise summary preserving important facts. No host commands are required.
"""


def initialize(
    model: str,
    base_url: str | None,
    context_window: int,
    max_output: int,
    template: str,
    api_key_env: str,
) -> None:
    try:
        if template not in {"local", "authenticated"}:
            raise RunnerError("invalid_arguments", "Choose --template local or authenticated.")
        if template == "authenticated" and not base_url:
            raise RunnerError(
                "invalid_arguments", "The authenticated template requires --base-url."
            )
        profile = ModelProfile(
            base_url=base_url or "http://localhost:8000/v1",
            model=model,
            auth_mode="none" if template == "local" else "bearer",
            api_key_env=api_key_env if template == "authenticated" else None,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
        )
        if max_output > context_window:
            raise RunnerError("invalid_arguments", "--max-output must not exceed --context-window.")
        config = Path.cwd() / "skillrun.toml"
        skill = Path.cwd() / "skills/summarize/SKILL.md"
        if any(path.exists() or path.is_symlink() for path in (config, skill)):
            raise RunnerError(
                "invalid_arguments",
                "Starter files already exist; init never overwrites them.",
                details={
                    "suggested_action": "Use an empty directory or edit the existing configuration."
                },
            )
        lines = [
            "schema_version = 1",
            'skills_dir = "./skills"',
            'output_dir = "./outputs"',
            'default_model = "default"',
            "",
            "[models.default]",
        ]
        for key in (
            "base_url",
            "model",
            "auth_mode",
            "api_key_env",
            "context_window_tokens",
            "max_output_tokens",
        ):
            value = getattr(profile, key)
            if value is not None:
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines += [
            "",
            "# Commands are disabled. Add only trusted executables when needed.",
            "[policy]",
            "allowed_executables = []",
            "",
            "[diagnostics]",
            "log_content = false",
            "retain_work = false",
            "",
        ]
        skill.parent.mkdir(parents=True, exist_ok=True)
        with skill.open("x", encoding="utf-8") as output:
            output.write(SKILL)
        with config.open("x", encoding="utf-8") as output:
            output.write("\n".join(lines))
    except ValidationError:
        fail(
            RunnerError(
                "invalid_arguments",
                "Use a valid endpoint, model ID, credential reference and positive capacities.",
            )
        )
    except OSError:
        fail(
            RunnerError(
                "invalid_arguments",
                "Cannot create starter files; existing files were not overwritten. "
                "Inspect the directory before retrying.",
            )
        )
    except RunnerError as error:
        fail(error)
    typer.echo(f"Created {config} and {skill}.")
    if template == "authenticated":
        typer.echo(f"Set the {api_key_env} environment variable to your credential.")
    typer.echo(
        "Next: skillrun doctor\nThen: skillrun doctor --network\n"
        'Run: skillrun run "Summarize these notes" -i notes.txt --output-file summary.md'
    )
