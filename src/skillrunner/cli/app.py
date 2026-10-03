"""Explicit Typer command routing and option aliases."""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from skillrunner.cli import execution, inspection
from skillrunner.cli.routing import ReceiptGroup

HELP = {"help_option_names": ["-h", "--help"]}
app = typer.Typer(
    no_args_is_help=True,
    context_settings=HELP,
    add_completion=False,
    rich_markup_mode=None,
    cls=ReceiptGroup,
)
skills_app = typer.Typer(no_args_is_help=True, context_settings=HELP)
app.add_typer(skills_app, name="skills", help="Inspect skill packages without running them.")
config_app = typer.Typer(no_args_is_help=True, context_settings=HELP)
app.add_typer(config_app, name="config", help="Inspect effective configuration offline.")
inputs_app = typer.Typer(no_args_is_help=True, context_settings=HELP)
runs_app = typer.Typer(no_args_is_help=True, context_settings=HELP)
app.add_typer(inputs_app, name="inputs", help="Preview explicit input selection offline.")
app.add_typer(runs_app, name="runs", help="Inspect disk usage and safely clean completed bundles.")


def show_version(value: bool) -> None:
    if value:
        typer.echo(f"skillrunner {version('skillrunner')}")
        raise typer.Exit()


@app.callback()
def root(
    version_flag: Annotated[
        bool,
        typer.Option(
            "--version", callback=show_version, is_eager=True, help="Show the installed version."
        ),
    ] = False,
) -> None:
    """Run trusted skills with explicit permissions and persistent results."""


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
ModelAlias = Annotated[
    str | None,
    typer.Option("--model-alias", help="Select a named profile, ignoring OPENAI_BASE_URL."),
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
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            help="Explicit input-relative glob to omit; repeatable. Preview with inputs preview.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Publish to a file or existing directory."),
    ] = None,
    output_file: Annotated[
        Path | None, typer.Option("--output-file", help="Publish at this exact file path.")
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-directory", help="Publish inside a directory; create it if missing."
        ),
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
            help="Primary format; inferred from destination filename, otherwise Markdown.",
        ),
    ] = None,
    model: Model = None,
    model_alias: ModelAlias = None,
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
            help="Per-turn retries for transport failures and empty completions (default 1).",
        ),
    ] = None,
    shutdown_grace: Annotated[
        str | None,
        typer.Option(
            "--shutdown-grace", help="Duration with ms/s/m/h unit; permits 0s (default 5s)."
        ),
    ] = None,
    log_content: Annotated[
        bool | None,
        typer.Option(
            "--log-content/--no-log-content",
            help="Opt into detailed event content; may retain sensitive task data.",
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
        input_excludes=exclude or [],
        skills=skill or [],
        output=output,
        output_file=output_file,
        output_directory=output_directory,
        format=format,
        overwrite=overwrite,
        json_mode=json_mode,
        quiet=quiet,
        overrides={
            "skills_dir": skills_dir,
            "output_dir": output_dir,
            "model": model,
            "model_alias": model_alias,
            "log_content": log_content,
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
    model_alias: ModelAlias = None,
    base_url: BaseURL = None,
    policy: Policy = None,
    skills_dir: SkillsDir = None,
    json_mode: Json = False,
    network: Annotated[
        bool,
        typer.Option(
            "--network",
            help="Contact the model to verify tool calling; may consume provider resources.",
        ),
    ] = False,
) -> None:
    """Check local readiness offline; opt into a model tool-calling probe with --network."""
    inspection.doctor_command(
        {
            "config": config,
            "model": model,
            "model_alias": model_alias,
            "base_url": base_url,
            "policy": policy,
            "skills_dir": skills_dir,
        },
        json_mode,
        network=network,
    )


@config_app.command("show", context_settings=HELP)
def show_config(
    config: Config = None,
    model: Model = None,
    base_url: BaseURL = None,
    model_alias: ModelAlias = None,
    policy: Policy = None,
    json_mode: Json = False,
) -> None:
    """Show resolved paths, model, limits, formats and setting origins; no network."""
    inspection.config_command(
        {
            "config": config,
            "model": model,
            "base_url": base_url,
            "policy": policy,
            "model_alias": model_alias,
        },
        json_mode,
    )


@app.command("init", context_settings=HELP)
def init(
    model: Annotated[str, typer.Option("--model", help="Literal model ID.")],
    context_window: Annotated[
        int, typer.Option("--context-window", min=1, help="Documented context capacity.")
    ],
    max_output: Annotated[
        int, typer.Option("--max-output", min=1, help="Documented per-response output capacity.")
    ],
    base_url: BaseURL = None,
    template: Annotated[str, typer.Option("--template", help="local or authenticated.")] = "local",
    api_key_env: Annotated[
        str,
        typer.Option(
            "--api-key-env", help="Credential environment variable name, never the value."
        ),
    ] = "OPENAI_API_KEY",
) -> None:
    """Create a minimal config and text-only starter skill without overwriting files."""
    from skillrunner.cli.setup import initialize

    initialize(model, base_url, context_window, max_output, template, api_key_env)


@inputs_app.command("preview", context_settings=HELP)
def preview_inputs(
    input: Annotated[
        list[Path], typer.Option("-i", "--input", help="Input file or directory; repeatable.")
    ],
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", help="Explicit relative glob; repeatable.")
    ] = None,
    config: Config = None,
    json_mode: Json = False,
) -> None:
    """List selected files and sizes without copying contents or contacting a model."""
    from skillrunner.cli.operations import preview_inputs as preview

    preview(input, exclude or [], config, json_mode)


@runs_app.command("list", context_settings=HELP)
def list_runs(
    output_dir: Annotated[Path | None, typer.Option("-O", "--output-dir")] = None,
    config: Config = None,
    json_mode: Json = False,
) -> None:
    """List recognized bundles, completion status and disk usage."""
    from skillrunner.cli.operations import runs_command

    runs_command(output_dir, config, json_mode)


@runs_app.command("clean", context_settings=HELP)
def clean_runs(
    output_dir: Annotated[Path | None, typer.Option("-O", "--output-dir")] = None,
    config: Config = None,
    json_mode: Json = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Delete completed bundles; otherwise only preview.")
    ] = False,
) -> None:
    """Preview deletion of completed bundles; --yes confirms it. Active/unknown runs stay."""
    from skillrunner.cli.operations import runs_command

    runs_command(output_dir, config, json_mode, clean=True, yes=yes)


def main() -> None:
    app()
