"""Explicit Typer command routing and option aliases."""

from pathlib import Path
from typing import Annotated

import typer

from skillrunner.cli import execution, inspection

HELP = {"help_option_names": ["-h", "--help"]}
app = typer.Typer(
    no_args_is_help=True, context_settings=HELP, add_completion=False, rich_markup_mode=None
)
skills_app = typer.Typer(no_args_is_help=True, context_settings=HELP)
app.add_typer(skills_app, name="skills", help="Inspect skill packages without running them.")

Config = Annotated[
    Path | None,
    typer.Option("-c", "--config", help="Use this TOML file instead of ./skillrun.toml."),
]
SkillsDir = Annotated[
    Path | None, typer.Option("-S", "--skills-dir", help="Catalog root (default ./skills).")
]
Json = Annotated[bool, typer.Option("-j", "--json", help="Print one JSON receipt on stdout.")]
Model = Annotated[
    str | None,
    typer.Option("-m", "--model", help="Configured alias, or literal ID with --base-url."),
]
BaseURL = Annotated[
    str | None, typer.Option("-b", "--base-url", help="OpenAI-compatible endpoint URL.")
]
Policy = Annotated[
    Path | None, typer.Option("--policy", help="Replace the entire configured policy.")
]


@app.command(
    "run",
    context_settings=HELP,
    epilog=(
        'Examples: skillrun run "Summarize notes" -i notes.txt -o summary.md; '
        "skillrun run -p task.txt -s writer -j"
    ),
)
def run(
    prompt: Annotated[
        str | None, typer.Argument(help="Task text; exactly one prompt source is required.")
    ] = None,
    prompt_file: Annotated[
        Path | None, typer.Option("-p", "--prompt-file", help="Read UTF-8 prompt text.")
    ] = None,
    prompt_stdin: Annotated[
        bool, typer.Option("--prompt-stdin", help="Read prompt from stdin until EOF.")
    ] = False,
    skills_dir: SkillsDir = None,
    skill: Annotated[
        list[str] | None, typer.Option("-s", "--skill", help="Require a skill; repeatable.")
    ] = None,
    input: Annotated[
        list[Path] | None,
        typer.Option("-i", "--input", help="Read-only input file or directory; repeatable."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Publish to a file or existing directory."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("-O", "--output-dir", help="Run bundle parent (default ./outputs)."),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option(
            "-f",
            "--format",
            help="Primary output format; inferred from output, otherwise Markdown.",
        ),
    ] = None,
    model: Model = None,
    base_url: BaseURL = None,
    config: Config = None,
    policy: Policy = None,
    timeout: Annotated[
        str | None,
        typer.Option(
            "-t", "--timeout", help="Positive duration with ms/s/m/h unit, e.g. 1.5m (default 10m)."
        ),
    ] = None,
    max_steps: Annotated[
        int | None, typer.Option("--max-steps", help="Maximum model turns (default 40).")
    ] = None,
    max_tool_calls: Annotated[
        int | None, typer.Option("--max-tool-calls", help="Maximum tool calls (default 100).")
    ] = None,
    max_tokens: Annotated[
        int | None, typer.Option("--max-tokens", help="Aggregate token budget (default 100000).")
    ] = None,
    model_transport_retries: Annotated[
        int | None,
        typer.Option(
            "--model-transport-retries",
            help="Retries for transient model transport failures (default 1).",
        ),
    ] = None,
    shutdown_grace: Annotated[
        str | None,
        typer.Option(
            "--shutdown-grace", help="Duration with ms/s/m/h unit; permits 0s (default 5s)."
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Permit replacing the requested primary output.")
    ] = False,
    json_mode: Json = False,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Suppress progress; retain receipt and errors.")
    ] = False,
) -> None:
    """Execute one task and persist its outputs and execution report."""
    execution.execute(
        prompt=prompt,
        prompt_file=prompt_file,
        prompt_stdin=prompt_stdin,
        inputs=input or [],
        skills=skill or [],
        output=output,
        format=format,
        overwrite=overwrite,
        json_mode=json_mode,
        quiet=quiet,
        overrides={
            "skills_dir": skills_dir,
            "output_dir": output_dir,
            "model": model,
            "base_url": base_url,
            "config": config,
            "policy": policy,
            "timeout": timeout,
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
            "max_tokens": max_tokens,
            "model_transport_retries": model_transport_retries,
            "shutdown_grace": shutdown_grace,
        },
    )


@skills_app.command("list", context_settings=HELP)
def list_skills(
    skills_dir: SkillsDir = None, config: Config = None, json_mode: Json = False
) -> None:
    """Show valid skill names, descriptions, and rejected packages."""
    inspection.catalog_command(None, skills_dir, config, json_mode, validate=False)


@skills_app.command("validate", context_settings=HELP)
def validate_skills(
    path: Annotated[Path | None, typer.Argument(help="Package or catalog to validate.")] = None,
    skills_dir: SkillsDir = None,
    config: Config = None,
    json_mode: Json = False,
) -> None:
    """Validate package structure without model calls or script execution."""
    inspection.catalog_command(path, skills_dir, config, json_mode, validate=True)


@app.command(context_settings=HELP)
def doctor(
    config: Config = None,
    model: Model = None,
    base_url: BaseURL = None,
    policy: Policy = None,
    skills_dir: SkillsDir = None,
    json_mode: Json = False,
) -> None:
    """Check configuration, credentials, model connectivity, and host readiness.

    Always contacts the selected model: requires network access and may consume provider resources.
    """
    inspection.doctor_command(
        {
            "config": config,
            "model": model,
            "base_url": base_url,
            "policy": policy,
            "skills_dir": skills_dir,
        },
        json_mode,
    )


def main() -> None:
    app()
