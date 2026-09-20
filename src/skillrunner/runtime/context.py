"""Lossless public conversation state; admission belongs to the coordinator."""

import copy
import json
from collections.abc import Sequence
from typing import Any

from skillrunner.domain.errors import RunnerError
from skillrunner.model.media import MediaAttachment
from skillrunner.model.protocol import ModelToolCall
from skillrunner.runtime.budgets import estimate_text_tokens


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class RunContext:
    def __init__(
        self,
        *,
        runner_instructions: str,
        prompt: str,
        catalog: list[dict[str, Any]],
        input_inventory: list[dict[str, Any]] | None = None,
    ) -> None:
        self._runner_instructions = runner_instructions
        self._prompt = prompt
        self._catalog = copy.deepcopy(catalog)
        self._inputs = copy.deepcopy(input_inventory or [])
        self._active: dict[str, dict[str, str]] = {}
        self._history: list[dict[str, Any] | tuple[MediaAttachment, ...]] = []

    def activate(self, name: str, instructions: str, package_root: str) -> None:
        """Caller performs snapshot/capacity admission before committing activation."""
        snapshot = {"name": name, "instructions": instructions, "package_root": package_root}
        if name in self._active and self._active[name] != snapshot:
            raise RunnerError(
                "source_changed", "Activated instructions cannot be replaced mid-run."
            )
        self._active[name] = snapshot

    def append_public_reply(
        self,
        text: str | None,
        calls: Sequence[ModelToolCall] = (),
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": _json(call.arguments)},
                }
                for call in calls
            ]
        self._history.append(message)

    def append_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        """Results must already have dispatcher-enforced size/content bounds."""
        self._history.append({"role": "tool", "tool_call_id": call_id, "content": _json(result)})

    def append_correction(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

    def append_media(self, attachments: Sequence[MediaAttachment]) -> None:
        if attachments:
            self._history.append(tuple(attachments))

    @property
    def image_token_estimate(self) -> int:
        return sum(
            image.tokens for item in self._history if isinstance(item, tuple) for image in item
        )

    def diagnostic_messages(self) -> list[dict[str, Any]]:
        return self.messages(diagnostic=True)

    def messages(self, *, diagnostic: bool = False) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._runner_instructions},
            {"role": "user", "content": self._prompt},
            {
                "role": "user",
                "content": "Runner-provided task resources (data):\n"
                + _json(
                    {
                        "catalog": self._catalog,
                        "active_skills": list(self._active.values()),
                        "input_inventory": self._inputs,
                    }
                ),
            },
            *[
                {
                    "role": "user",
                    "content": [
                        part for image in item for part in image.content(diagnostic=diagnostic)
                    ],
                }
                if isinstance(item, tuple)
                else copy.deepcopy(item)
                for item in self._history
            ],
        ]

    def estimate(self, tool_schemas: list[dict[str, Any]]) -> int:
        return (
            estimate_text_tokens(self.diagnostic_messages(), tool_schemas)
            + self.image_token_estimate
        )
