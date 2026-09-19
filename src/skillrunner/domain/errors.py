"""Stable, safe public errors shared by domain and adapters."""

from typing import Any

_ERROR_GROUPS = {
    "invalid_request": (
        2,
        {
            "invalid_configuration",
            "invalid_arguments",
            "duplicate_skill_name",
            "required_skill_unavailable",
        },
    ),
    "no_matching_skill": (3, {"no_matching_skill"}),
    "blocked": (
        4,
        {
            "missing_dependency",
            "missing_credential",
            "command_not_allowed",
            "mcp_tool_not_allowed",
            "unsupported_capability",
        },
    ),
    "needs_input": (5, {"missing_decision"}),
    "limit_exceeded": (7, {"context_capacity_exceeded", "budget_exhausted"}),
}


class RunnerError(ValueError):
    """Messages and details must be safe for public rendering, never raw inputs."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        status: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        default_status, default_exit = "failed", 6
        for candidate, (number, codes) in _ERROR_GROUPS.items():
            if code in codes:
                default_status, default_exit = candidate, number
                break
        if code in {"cancelled", "terminated"}:
            default_status = "cancelled"
            default_exit = 130 if code == "cancelled" else 143
        self.code = code
        self.message = message
        self.status = status or default_status
        self.exit_code = default_exit if exit_code is None else exit_code
        self.details = details or {}
        super().__init__(f"{code}: {message}")
