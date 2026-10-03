"""Uniform machine-readable parser failures without echoing argument values."""

import sys
from collections.abc import Sequence
from typing import Any

from typer import Exit
from typer.core import TyperGroup

try:  # Recent Typer versions vendor Click.
    from typer._click.exceptions import ClickException
except ImportError:
    from click.exceptions import ClickException  # type: ignore[assignment]

from skillrunner.cli.receipts import fail
from skillrunner.domain.errors import RunnerError


class ReceiptGroup(TyperGroup):
    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        kwargs = dict(
            prog_name=prog_name,
            complete_var=complete_var,
            windows_expand_args=windows_expand_args,
            **extra,
        )
        arguments = list(sys.argv[1:] if args is None else args)
        options = arguments[: arguments.index("--")] if "--" in arguments else arguments
        if not any(
            arg in {"-j", "--json"}
            or (arg.startswith("-") and "j" in arg[1:] and set(arg[1:]) <= {"q", "j"})
            for arg in options
        ):
            return super().main(args=arguments, standalone_mode=standalone_mode, **kwargs)
        standalone = standalone_mode
        try:
            result = super().main(args=arguments, standalone_mode=False, **kwargs)
        except ClickException as error:
            parameter = getattr(error, "param", None)
            names = getattr(parameter, "opts", [])
            hint = names[-1] if names else None
            message = (
                f"Invalid value for {hint}; check its type and range."
                if hint
                else ("Invalid command-line arguments; check option names and required values.")
            )
            try:
                fail(
                    RunnerError(
                        "invalid_arguments",
                        message,
                        details={
                            "suggested_action": (
                                "Check option names and value types with skillrun run --help."
                            )
                        },
                    ),
                    True,
                )
            except Exit as exit_error:
                if standalone:
                    raise SystemExit(exit_error.exit_code) from None
                raise
        if standalone:
            raise SystemExit(result if isinstance(result, int) else 0)
        return result
