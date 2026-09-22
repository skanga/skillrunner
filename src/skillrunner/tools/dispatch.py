"""Sequential, whole-batch-admitted tool dispatch with bounded public results."""

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from skillrunner.domain.errors import RunnerError
from skillrunner.model.media import MediaAttachment
from skillrunner.model.protocol import ModelToolCall
from skillrunner.runtime.budgets import Deadline, UsageLedger
from skillrunner.tools.schemas import WriteFileArgs

ToolKind = Literal["action", "activation", "completion"]
Handler = Callable[[Any], Awaitable[Any]]
_TERMINAL = {
    "reporting_failed",
    "budget_exhausted",
    "context_capacity_exceeded",
    "cancelled",
    "terminated",
    "external_outcome_unknown",
}
_DENIED = {"command_not_allowed", "mcp_tool_not_allowed", "file_access_denied", "invalid_arguments"}


def _normalize(value: Any) -> Any:
    """Explicit JSON conversion; arbitrary objects never leak through repr/default=str."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    raise TypeError("Unsupported tool result type")


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    executed: bool
    ok: bool
    value: Any = None
    error: dict[str, Any] | None = None
    _public: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    attachment: MediaAttachment | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        if self._public is not None:
            return dict(self._public)
        return {
            "call_id": self.call_id,
            "name": self.name,
            "executed": self.executed,
            "ok": self.ok,
            "value": _normalize(self.value),
            "error": _normalize(self.error),
        }


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    kind: ToolKind
    parameters_schema: dict[str, Any] | None
    terminal_errors: frozenset[str]


class ToolRegistry:
    def __init__(
        self,
        *,
        max_result_bytes: int = 1_048_576,
        activation_is_new: Callable[[ModelToolCall], bool] | None = None,
        on_start: Callable[[ModelToolCall], None] | None = None,
    ) -> None:
        if type(max_result_bytes) is not int or max_result_bytes < 2:
            raise ValueError("Result byte limit must fit a JSON object (at least 2 bytes)")
        self.max_result_bytes = max_result_bytes
        self.activation_is_new = activation_is_new
        self.on_start = on_start
        self._tools: dict[str, _Tool] = {}
        self.last_results: list[ToolResult] = []
        self.current_call: ModelToolCall | None = None

    def register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Handler,
        *,
        kind: ToolKind = "action",
        parameters_schema: dict[str, Any] | None = None,
        terminal_errors: frozenset[str] = frozenset(),
    ) -> None:
        if not name or name in self._tools or kind not in {"action", "activation", "completion"}:
            raise ValueError("Tool names must be nonempty and unique, with a valid kind")
        self._tools[name] = _Tool(
            name,
            description,
            args_model,
            handler,
            kind,
            copy.deepcopy(parameters_schema),
            terminal_errors,
        )

    def model_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": copy.deepcopy(tool.parameters_schema)
                    if tool.parameters_schema is not None
                    else tool.args_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def _bounded(self, value: Any, limit: int) -> Any:
        serialized = _serialize(value)
        size = len(serialized.encode("utf-8"))
        if size <= limit:
            return value
        result = {"truncated": True, "original_bytes": size, "omitted_bytes": size, "preview": ""}
        if len(_serialize(result)) > limit:
            raise RunnerError(
                "budget_exhausted", "Tool result limit cannot fit truncation metadata."
            )
        low, high = 0, min(len(serialized), limit)
        while low < high:
            middle = (low + high + 1) // 2
            result.update(preview=serialized[:middle], omitted_bytes=size - middle)
            if len(_serialize(result)) <= limit:
                low = middle
            else:
                high = middle - 1
        result.update(preview=serialized[:low], omitted_bytes=size - low)
        return result

    def _bound_result(self, result: ToolResult) -> tuple[ToolResult, RunnerError | None]:
        public = result.as_dict()
        if len(_serialize(public)) <= self.max_result_bytes:
            return result, None
        key = "error" if result.error is not None else "value"
        original = public[key]
        public[key] = None
        allowance = self.max_result_bytes - len(_serialize(public)) + 4
        try:
            if key == "error":
                bounded = {
                    "code": original.get("code", "internal_error"),
                    "truncated": True,
                    "original_bytes": len(_serialize(original)),
                }
                if len(_serialize(bounded)) > allowance:
                    raise RunnerError("budget_exhausted", "Tool result envelope exceeds its limit.")
                return replace(result, error=bounded), None
            return replace(result, value=self._bounded(original, allowance)), None
        except RunnerError as error:
            # Identity stays on the in-memory outcome, never silently abbreviated.
            # No model turn follows this terminal budget failure.
            return replace(
                result, value=None, error={"code": "budget_exhausted"}, _public={}
            ), error

    def _batch_kind(self, call: ModelToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return "action"
        if tool.kind == "activation" and self.activation_is_new is not None:
            try:
                if not self.activation_is_new(call):
                    return "action"
            except Exception:
                pass  # Failed classification must not permit action side effects.
        return tool.kind

    async def dispatch_batch(
        self,
        calls: Sequence[ModelToolCall],
        ledger: UsageLedger,
        deadline: Deadline,
        *,
        on_result: Callable[[ToolResult], Awaitable[None]] | None = None,
    ) -> list[ToolResult]:
        self.last_results = []
        batch = ledger.admit_tools(len(calls))
        kinds = {self._batch_kind(call) for call in calls}
        mixed = ("activation" in kinds and len(kinds) > 1) or (
            "completion" in kinds and len(calls) > 1
        )
        try:
            for call in calls:
                self.current_call = call
                await asyncio.sleep(0)
                deadline.check()
                ledger.charge_tool(batch)
                if self.on_start is not None:
                    try:
                        self.on_start(call)
                    except RunnerError as error:
                        error.details.setdefault("tool", call.name)
                        raise
                tool = self._tools.get(call.name)
                terminal: BaseException | None = None
                executed = False
                timeout_scope: asyncio.Timeout | None = None
                try:
                    if mixed:
                        raise RunnerError(
                            "model_protocol_error",
                            "Activation and completion require separate tool responses.",
                        )
                    tool = self._tools.get(call.name)
                    if tool is None:
                        raise RunnerError("unknown_tool", "Requested tool is not registered.")
                    args = tool.args_model.model_validate(call.arguments)
                    await asyncio.sleep(0)
                    deadline.check()
                    executed = True
                    timeout_scope = asyncio.timeout(deadline.remaining)
                    async with timeout_scope:
                        value = await tool.handler(args)
                    attachment = value if isinstance(value, MediaAttachment) else None
                    value = attachment.metadata() if attachment else _normalize(value)
                    _serialize(value)
                    result = ToolResult(
                        call.id, call.name, True, True, value, attachment=attachment
                    )
                    if attachment is not None:
                        encoded = _serialize(result.as_dict()) + _serialize(
                            {"role": "user", "content": attachment.content()}
                        )
                        if len(encoded.encode("utf-8")) > self.max_result_bytes:
                            raise RunnerError(
                                "budget_exhausted", "Encoded image exceeds tool output limit."
                            )
                except ValidationError as error:
                    # Locations, validator messages and input values can contain secrets.
                    details = {
                        "code": "invalid_arguments",
                        "message": "Arguments failed validation.",
                        "issues": [
                            {
                                "type": item["type"],
                                **(
                                    {
                                        "field": "expected_sha256",
                                        "guidance": (
                                            "Read the current file with read_text and copy its "
                                            "full SHA-256 into expected_sha256. Use overwrite=true "
                                            "only for intentional replacement."
                                        ),
                                    }
                                    if not executed
                                    and tool is not None
                                    and tool.args_model is WriteFileArgs
                                    and item["type"] == "overwrite_digest_required"
                                    else {}
                                ),
                                **(
                                    {"field": item["loc"][0]}
                                    if item["loc"]
                                    and tool is not None
                                    and item["loc"][0] in tool.args_model.model_fields
                                    else {}
                                ),
                            }
                            for item in error.errors(
                                include_input=False, include_context=False, include_url=False
                            )
                        ],
                    }
                    result = ToolResult(call.id, call.name, False, False, error=details)
                except (RunnerError, TimeoutError, asyncio.CancelledError) as error:
                    if isinstance(error, asyncio.CancelledError):
                        terminal = error
                        public = RunnerError(
                            "cancelled",
                            "Tool execution was cancelled.",
                            details={
                                "outcome_unknown": bool(getattr(error, "outcome_unknown", False)),
                                "cleanup_errors": list(getattr(error, "__notes__", [])),
                            },
                        )
                    elif isinstance(error, TimeoutError):
                        if timeout_scope is not None and timeout_scope.expired():
                            public = RunnerError(
                                "budget_exhausted",
                                "Execution timeout reached.",
                                details={
                                    "outcome_unknown": bool(
                                        getattr(error.__cause__, "outcome_unknown", False)
                                    ),
                                    "cleanup_errors": list(
                                        getattr(error.__cause__, "__notes__", [])
                                    ),
                                },
                            )
                            terminal = public
                        else:
                            public = RunnerError(
                                "internal_error", "Tool execution failed unexpectedly."
                            )
                    else:
                        public = error
                        if public.code in _TERMINAL or (
                            tool is not None and public.code in tool.terminal_errors
                        ):
                            terminal = public
                    public.details.setdefault("tool", call.name)
                    details = {
                        "code": public.code,
                        "message": public.message,
                        "details": _normalize(public.details),
                    }
                    result = ToolResult(
                        call.id,
                        call.name,
                        executed and public.code not in _DENIED,
                        False,
                        error=details,
                    )
                except Exception:
                    result = ToolResult(
                        call.id,
                        call.name,
                        executed,
                        False,
                        error={
                            "code": "internal_error",
                            "message": "Tool execution failed unexpectedly.",
                        },
                    )
                tool = self._tools.get(call.name)
                exempt = (
                    result.ok and tool is not None and tool.kind in {"activation", "completion"}
                )
                if not exempt:
                    result, bound_error = self._bound_result(result)
                    if terminal is None and bound_error is not None:
                        terminal = bound_error
                self.last_results.append(result)
                if on_result is not None:
                    try:
                        await on_result(result)
                    except BaseException:
                        if terminal is not None:
                            raise terminal from None
                        raise
                if terminal is not None:
                    raise terminal
            return list(self.last_results)
        finally:
            self.current_call = None
            ledger.close_tools(batch)
