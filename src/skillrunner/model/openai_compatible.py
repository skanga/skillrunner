"""Generic Chat Completions transport with explicit credentials and bounded discovery.

Discovery mappings use dot-separated object keys (for example ``limits.context``)
relative to the selected model object; mapped capacity values must be integer token
counts. A list response must contain exactly one record with the configured model
ID. Without a configured mapping no capacity schema is assumed.
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote, unquote

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, Omit
from pydantic import SecretStr

from skillrunner.config.models import ModelProfile
from skillrunner.domain.errors import RunnerError
from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage

_METADATA_BYTES = 1_048_576


class _ExplicitHeadersClient(AsyncOpenAI):
    @property
    def default_headers(self) -> dict[str, str | Omit]:
        # AsyncOpenAI merges OPENAI_CUSTOM_HEADERS in __init__. Do not use its
        # default headers: neither those nor ambient org/project may cross here.
        headers: dict[str, str | Omit] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


@dataclass(frozen=True)
class ResolvedCapabilities:
    profile: ModelProfile
    discovered: dict[str, Any]
    configured: dict[str, Any]
    sources: dict[str, str]
    metadata_source: str


def _protocol_error() -> RunnerError:
    return RunnerError("model_protocol_error", "The endpoint returned an invalid model response.")


def _status_error(status: int, code: str | None = None) -> RunnerError:
    details: dict[str, Any] = {"http_status": status, "retryable": False}
    if status == 400 and code == "context_length_exceeded":
        details["suggested_action"] = (
            "Reduce the request context or configure a larger-capacity model."
        )
        return RunnerError(
            "context_capacity_exceeded", "The endpoint rejected the context size.", details=details
        )
    if status in {401, 403}:
        details["suggested_action"] = (
            "Check the configured credential reference and endpoint access."
        )
        return RunnerError(
            "missing_credential",
            "The endpoint rejected the configured credentials.",
            details=details,
        )
    if status in {400, 404, 405, 415, 422}:
        details["suggested_action"] = (
            "Check the endpoint URL, model ID and configured request options."
        )
        return RunnerError(
            "unsupported_capability", "The endpoint does not support this request.", details=details
        )
    if status == 429 and code == "insufficient_quota":
        details["suggested_action"] = (
            "Check the provider quota and billing allowance before rerunning."
        )
    else:
        details["retryable"] = status == 429 or 500 <= status <= 599
        details["suggested_action"] = (
            "Check endpoint availability and rate limits, then rerun within the remaining budget."
            if details["retryable"]
            else "Check the endpoint service configuration and HTTP status before rerunning."
        )
    return RunnerError("model_transport_error", "The endpoint request failed.", details=details)


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise RunnerError("model_timeout", "The model request deadline has elapsed.")
    return remaining


def _usage(value: Any) -> ModelUsage | None:
    if not isinstance(value, dict):
        return None
    counts: list[Any] = [
        value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    if any(type(n) is not int or n < 0 for n in counts):
        return None
    if counts[0] + counts[1] != counts[2]:
        return None
    return ModelUsage(*counts)


def _reject_constant(value: str) -> Any:
    raise ValueError("Non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _normalize(payload: Any, request_id: str | None) -> ModelReply:
    try:
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        finish = choice["finish_reason"]
        if finish not in {"stop", "length", "tool_calls", "content_filter"}:
            raise ValueError
        message = choice["message"]
        if message.get("role") != "assistant" or message.get("function_call") is not None:
            raise ValueError
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise ValueError
        normalized = []
        ids: set[str] = set()
        for call in calls:
            call_id = call["id"]
            function = call["function"]
            name = function["name"]
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or any(ord(c) < 33 for c in call_id)
                or call_id in ids
                or call.get("type") != "function"
                or not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            ):
                raise ValueError
            raw = function["arguments"]
            if not isinstance(raw, str):
                raise ValueError
            ids.add(call_id)
            if finish == "length":
                # Truncated arguments are not an executable action batch.
                continue
            arguments = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
            if not isinstance(arguments, dict):
                raise ValueError
            normalized.append(ModelToolCall(call_id, name, arguments, raw))
        if finish != "length" and (
            (bool(calls) != (finish == "tool_calls")) or (not calls and not (content or "").strip())
        ):
            raise ValueError
        return ModelReply(
            content, tuple(normalized), finish, _usage(payload.get("usage")), request_id
        )
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise _protocol_error() from None


class OpenAICompatibleAdapter:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        api_key: SecretStr | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.profile = profile
        if profile.auth_mode == "bearer" and (
            api_key is None or not api_key.get_secret_value().strip()
        ):
            raise RunnerError("missing_credential", "Provide the configured model credential.")
        key = api_key.get_secret_value() if api_key and profile.auth_mode == "bearer" else ""
        self._http = httpx.AsyncClient(transport=transport, follow_redirects=False, trust_env=False)
        self._client = _ExplicitHeadersClient(
            base_url=profile.base_url,
            api_key=key,
            admin_api_key="",
            organization="",
            project="",
            webhook_secret="",
            max_retries=0,
            http_client=self._http,
            _enforce_credentials=False,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        output_limit: int,
        request_deadline: float,
    ) -> ModelReply:
        if type(output_limit) is not int or output_limit <= 0:
            raise RunnerError("invalid_arguments", "Output allowance must be a positive integer.")
        if self.profile.max_output_tokens and output_limit > self.profile.max_output_tokens:
            raise RunnerError(
                "context_capacity_exceeded", "Output allowance exceeds model capacity."
            )
        remaining = _remaining(request_deadline)
        options: dict[str, Any] = {
            **self.profile.request_options,
            "model": self.profile.model,
            "messages": messages,
            "stream": False,
            self.profile.output_token_parameter: output_limit,
            "timeout": remaining,
        }
        if self.profile.auth_mode == "none":
            options["extra_headers"] = {"Authorization": Omit()}
        if tool_schemas:
            options["tools"] = tool_schemas
        try:
            async with asyncio.timeout(remaining):
                response = await self._client.chat.completions.with_raw_response.create(**options)
                try:
                    payload = response.http_response.json()
                except (ValueError, UnicodeError, RecursionError):
                    raise _protocol_error() from None
                return _normalize(payload, response.http_response.headers.get("x-request-id"))
        except APIStatusError as error:
            raise _status_error(error.status_code, error.code) from None
        except (TimeoutError, APITimeoutError):
            raise RunnerError(
                "model_timeout",
                "The model request deadline has elapsed.",
                details={
                    "retryable": False,
                    "outcome_certainty": "unknown",
                    "suggested_action": (
                        "Check endpoint latency and the run timeout before rerunning; "
                        "the request may already have consumed provider resources."
                    ),
                },
            ) from None
        except APIConnectionError:
            raise RunnerError(
                "model_transport_error",
                "Could not complete the model request.",
                details={
                    "retryable": True,
                    "outcome_certainty": "unknown",
                    "suggested_action": (
                        "Check endpoint connectivity before rerunning within the remaining budget; "
                        "the request may already have consumed provider resources."
                    ),
                },
            ) from None
        except (json.JSONDecodeError, UnicodeError, RecursionError):
            raise _protocol_error() from None

    async def discover_capabilities(self, request_deadline: float) -> ResolvedCapabilities:
        discovery = self.profile.discovery
        path = (
            discovery.path
            if discovery
            else "models/" + quote(self.profile.model, safe="").replace(".", "%2E")
        )
        decoded = unquote(path)
        if discovery and (
            decoded.startswith("/")
            or any(c in decoded for c in ("\\", "?", "#", ":"))
            or ".." in decoded.split("/")
        ):
            raise RunnerError(
                "invalid_configuration", "Discovery must use an endpoint-relative path."
            )
        url = self.profile.base_url.rstrip("/") + "/" + path
        remaining = min(_remaining(request_deadline), 10.0)
        payload: Any = None
        try:
            async with asyncio.timeout(remaining):
                async with self._http.stream(
                    "GET",
                    url,
                    headers=cast(dict[str, str], self._client.default_headers),
                    timeout=remaining,
                ) as response:
                    if response.status_code not in {404, 405, 501}:
                        if response.status_code != 200:
                            raise _status_error(response.status_code)
                        body = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=65_536):
                            body.extend(chunk)
                            if len(body) > _METADATA_BYTES:
                                raise _protocol_error()
                        payload = json.loads(body)
        except (TimeoutError, httpx.TimeoutException):
            raise RunnerError("model_timeout", "Model metadata discovery timed out.") from None
        except httpx.HTTPError:
            raise RunnerError("model_transport_error", "Model metadata discovery failed.") from None
        except (ValueError, UnicodeError, RecursionError) as error:
            if isinstance(error, RunnerError):
                raise
            raise _protocol_error() from None
        discovered: dict[str, Any] = {}
        record = _metadata_record(payload, self.profile.model)
        if discovery and record is not None:
            for key, field in (
                ("context_window_tokens", discovery.context_window_field),
                ("max_output_tokens", discovery.max_output_field),
                ("input_modalities", discovery.input_modalities_field),
            ):
                value: Any = record
                if field:
                    for part in field.split("."):
                        value = value.get(part) if isinstance(value, dict) else None
                    if value is not None:
                        discovered[key] = value
        configured = {
            key: getattr(self.profile, key)
            for key in ("context_window_tokens", "max_output_tokens", "input_modalities")
            if key in self.profile.model_fields_set and getattr(self.profile, key) is not None
        }
        resolved = {**discovered, **configured}
        missing = [
            key for key in ("context_window_tokens", "max_output_tokens") if key not in resolved
        ]
        if missing:
            raise RunnerError(
                "unsupported_capability",
                "Configure missing model capacity fields.",
                details={"required_keys": missing},
            )
        for key in ("context_window_tokens", "max_output_tokens"):
            if type(resolved[key]) is not int or resolved[key] <= 0:
                raise RunnerError(
                    "unsupported_capability",
                    "Model capacities require positive token counts.",
                    details={"required_keys": [key]},
                )
        if resolved["max_output_tokens"] > resolved["context_window_tokens"]:
            raise RunnerError("unsupported_capability", "Maximum output exceeds context capacity.")
        modalities = resolved.get("input_modalities", ["text"])
        if (
            not isinstance(modalities, list)
            or not modalities
            or any(not isinstance(item, str) or not item.strip() for item in modalities)
        ):
            raise RunnerError("unsupported_capability", "Model input modalities are invalid.")
        resolved["input_modalities"] = modalities
        sources = {key: "configured" if key in configured else "discovered" for key in resolved}
        if "input_modalities" not in configured and "input_modalities" not in discovered:
            sources["input_modalities"] = "default"
        return ResolvedCapabilities(
            self.profile.model_copy(update=resolved), discovered, configured, sources, url
        )


def _metadata_record(payload: Any, model: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if "data" in payload:
        records = payload["data"]
        if not isinstance(records, list):
            return None
        matches = [item for item in records if isinstance(item, dict) and item.get("id") == model]
        return matches[0] if len(matches) == 1 else None
    if payload.get("id", model) != model:
        return None
    return payload
