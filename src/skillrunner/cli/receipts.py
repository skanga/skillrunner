"""Terminal output stays separate from task and diagnostic output."""

import json
from typing import Any, NoReturn

import typer

from skillrunner.domain.errors import RunnerError


def render(receipt: dict[str, Any], json_mode: bool) -> None:
    if json_mode:
        typer.echo(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    else:
        typer.echo(f"Status: {receipt['status']}")
        for field in ("primary_output", "report_path", "manifest_path"):
            if receipt.get(field):
                typer.echo(f"{field}: {receipt[field]}")
    for error in receipt.get("errors", []):
        typer.echo(f"{error.get('code', 'error')}: {error.get('message', '')}", err=True)


def fail(error: RunnerError, json_mode: bool = False) -> NoReturn:
    render(
        {
            "schema_version": "1",
            "run_id": None,
            "status": error.status,
            "exit_code": error.exit_code,
            "primary_output": None,
            "report_path": None,
            "manifest_path": None,
            "artifact_paths": [],
            "errors": [{"code": error.code, "message": error.message}],
        },
        json_mode,
    )
    raise typer.Exit(error.exit_code)
