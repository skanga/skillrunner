"""Translate parsed task arguments into the application request."""

import asyncio
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from skillrunner.cli.receipts import fail, render
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError
from skillrunner.domain.request import RunRequest
from skillrunner.runtime.progress import reporting
from skillrunner.runtime.signals import run_with_signals as run_task


def execute(
    *,
    prompt: str | None,
    prompt_file: Path | None,
    prompt_stdin: bool,
    inputs: list[Path],
    skills: list[str],
    output: Path | None,
    format: str | None,
    overwrite: bool,
    json_mode: bool,
    quiet: bool,
    overrides: dict[str, Any],
    output_file: Path | None = None,
    output_directory: Path | None = None,
    input_excludes: list[str] | None = None,
) -> None:
    cwd = Path.cwd()
    environ = dict(os.environ)
    try:
        if sum((prompt is not None, prompt_file is not None, prompt_stdin)) != 1:
            raise RunnerError(
                "invalid_arguments",
                "Supply exactly one prompt source: PROMPT, --prompt-file, or --prompt-stdin.",
            )
        source = "argument"
        if prompt_file is not None:
            source = "file"
            try:
                prompt = (cwd / prompt_file).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                raise RunnerError(
                    "invalid_arguments", "Cannot read --prompt-file as UTF-8."
                ) from None
        elif prompt_stdin:
            source = "stdin"
            try:
                binary = getattr(sys.stdin, "buffer", None)
                prompt = binary.read().decode("utf-8") if binary is not None else sys.stdin.read()
                prompt.encode("utf-8", errors="strict")
            except (OSError, UnicodeError):
                raise RunnerError(
                    "invalid_arguments", "Cannot read --prompt-stdin as UTF-8."
                ) from None
        if sum(item is not None for item in (output, output_file, output_directory)) > 1:
            raise RunnerError(
                "invalid_arguments",
                "Choose only one of --output, --output-file, or --output-directory.",
            )
        try:
            request = RunRequest.model_validate(
                {
                    "prompt": prompt,
                    "prompt_source": source,
                    "invocation_directory": cwd,
                    "inputs": inputs,
                    "input_excludes": input_excludes or [],
                    "required_skills": skills,
                    "output": output_file or output_directory or output,
                    "output_kind": "file"
                    if output_file
                    else "directory"
                    if output_directory
                    else "auto",
                    "format": format,
                    "overwrite": overwrite,
                }
            )
        except ValidationError:
            raise RunnerError(
                "invalid_arguments",
                "Provide a nonempty prompt and compatible output format and extension.",
            ) from None
        settings = resolve_settings(cwd, overrides, environ)
        for warning in settings.warnings:
            typer.echo(f"Warning: {warning}", err=True)
        if not quiet:
            from skillrunner.config.sources import select_model

            try:
                selected = select_model(settings)
                typer.echo(
                    f"Model: {selected.model} at {selected.base_url}; "
                    f"auth={selected.auth_mode}; format={request.format}",
                    err=True,
                )
                typer.echo(
                    f"Bundles: {settings.output_dir}; output: {request.output or 'inside bundle'}",
                    err=True,
                )
            except RunnerError:
                pass  # Runtime preserves its durable diagnostic and exit status.
        # Suppress task progress in quiet mode; stderr diagnostics remain visible.
        with (
            (
                open(os.devnull, "w", encoding="utf-8") if quiet else nullcontext(sys.stderr)
            ) as progress,
            redirect_stdout(progress),
            reporting(not quiet),
        ):
            receipt = asyncio.run(run_task(request, settings, environ=environ))
    except RunnerError as error:
        fail(error, json_mode)
    render(receipt, json_mode)
    raise typer.Exit(int(receipt["exit_code"]))
