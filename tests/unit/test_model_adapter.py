import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from skillrunner.config.models import Discovery, ModelProfile
from skillrunner.domain.errors import RunnerError


def adapter(profile: ModelProfile, handler: Any, key: SecretStr | None = None) -> Any:
    from skillrunner import model

    assert hasattr(model, "OpenAICompatibleAdapter"), "Model adapter is not implemented"
    return model.OpenAICompatibleAdapter(
        profile, api_key=key, transport=httpx.MockTransport(handler)
    )


def profile(**kwargs: Any) -> ModelProfile:
    return ModelProfile(
        base_url="https://example.test/custom/api/",
        model="arbitrary/model",
        auth_mode="none",
        **kwargs,
    )


def reply(**kwargs: Any) -> dict[str, Any]:
    return {
        "id": "chat-1",
        "created": 1,
        "model": "arbitrary/model",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Done"},
            }
        ],
        **kwargs,
    }


def deadline() -> float:
    return asyncio.get_running_loop().time() + 2


@pytest.mark.parametrize("token_parameter", ["max_tokens", "max_completion_tokens"])
async def test_custom_prefix_no_ambient_headers(monkeypatch: Any, token_parameter: str) -> None:
    for name, value in {
        "OPENAI_API_KEY": "ambient-secret",
        "OPENAI_ORG_ID": "org",
        "OPENAI_PROJECT_ID": "project",
        "OPENAI_ADMIN_KEY": "admin",
        "OPENAI_CUSTOM_HEADERS": "Authorization: Bearer leaked\nX-Secret: hidden",
    }.items():
        monkeypatch.setenv(name, value)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/custom/api/chat/completions"
        assert "authorization" not in request.headers
        assert "x-secret" not in request.headers
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        body = json.loads(request.content)
        assert body[token_parameter] == 17
        assert body["model"] == "arbitrary/model"
        assert body["stream"] is False
        return httpx.Response(200, json=reply(), headers={"x-request-id": "req-1"})

    client = adapter(profile(output_token_parameter=token_parameter), handler)
    try:
        result = await client.complete([{"role": "user", "content": "Hi"}], [], 17, deadline())
        assert result.public_text == "Done"
        assert result.usage is None
        assert result.provider_request_id == "req-1"
        assert len(seen) == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "missing_credential"),
        (403, "missing_credential"),
        (429, "model_transport_error"),
        (500, "model_transport_error"),
        (400, "unsupported_capability"),
    ],
)
async def test_errors_safe_and_never_retried(status: int, code: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer explicit"
        return httpx.Response(status, json={"error": {"message": "PRIVATE secret"}})

    selected = profile().model_copy(update={"auth_mode": "bearer"})
    client = adapter(selected, handler, SecretStr("explicit"))
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 5, deadline())
        assert exc.value.code == code
        assert "PRIVATE" not in str(exc.value)
        assert exc.value.details["http_status"] == status
        assert exc.value.details["retryable"] is (status in {429, 500})
        assert exc.value.details["suggested_action"]
        assert "PRIVATE" not in json.dumps(exc.value.details)
        assert calls == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(408, False), (429, True), (500, True), (501, True), (511, True)],
)
async def test_only_approved_http_statuses_are_transport_retryable(status, retryable):
    from skillrunner.model.openai_compatible import _status_error

    assert _status_error(status).details["retryable"] is retryable


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_reports_uncertainty_without_replaying(failure):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise failure("PRIVATE transport detail", request=request)

    client = adapter(profile(), handler)
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 5, deadline())
        assert exc.value.details["outcome_certainty"] == "unknown"
        assert exc.value.details["suggested_action"]
        assert "PRIVATE" not in str(exc.value) + json.dumps(exc.value.details)
        assert calls == 1
    finally:
        await client.aclose()


async def test_quota_exhaustion_is_not_marked_as_transient_rate_limit():
    client = adapter(
        profile(),
        lambda request: httpx.Response(
            429, json={"error": {"code": "insufficient_quota", "message": "PRIVATE"}}
        ),
    )
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 5, deadline())
        assert exc.value.details["retryable"] is False
        assert "quota" in exc.value.details["suggested_action"].lower()
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "message,finish",
    [
        ({"content": ""}, "stop"),
        ({"content": "x"}, "unknown"),
        (
            {
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "[]"}}
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "bad name", "arguments": "{}"},
                    }
                ]
            },
            "tool_calls",
        ),
    ],
)
async def test_malformed_reply(message: dict[str, Any], finish: str) -> None:
    client = adapter(
        profile(),
        lambda _: httpx.Response(
            200,
            json=reply(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", **message},
                    }
                ]
            ),
        ),
    )
    try:
        with pytest.raises(RunnerError, match="model_protocol_error"):
            await client.complete([], [], 10, deadline())
    finally:
        await client.aclose()


async def test_tools_and_usage() -> None:
    payload = reply(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    )
    client = adapter(profile(), lambda _: httpx.Response(200, json=payload))
    try:
        result = await client.complete([], [], 10, deadline())
        assert result.tool_calls[0].arguments == {"path": "x"}
        assert result.usage.input_tokens == 4
        assert result.usage.output_tokens == 2
    finally:
        await client.aclose()


async def test_deadline_and_cancellation() -> None:
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client = adapter(profile(), handler)
    try:
        with pytest.raises(RunnerError, match="model_timeout"):
            await client.complete([], [], 10, asyncio.get_running_loop().time() + 0.02)
        started.clear()
        task = asyncio.create_task(client.complete([], [], 10, deadline()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()


async def test_discovery_override_and_provenance() -> None:
    selected = profile(
        context_window_tokens=500,
        discovery=Discovery(
            path="metadata", context_window_field="limits.context", max_output_field="limits.output"
        ),
    )
    client = adapter(
        selected,
        lambda _: httpx.Response(
            200, json={"id": "arbitrary/model", "limits": {"context": 1000, "output": 100}}
        ),
    )
    try:
        result = await client.discover_capabilities(deadline())
        assert result.profile.context_window_tokens == 500
        assert result.profile.max_output_tokens == 100
        assert result.discovered["context_window_tokens"] == 1000
        assert result.configured["context_window_tokens"] == 500
        assert result.sources["max_output_tokens"] == "discovered"
    finally:
        await client.aclose()


@pytest.mark.parametrize("status", [200, 404])
async def test_ids_only_or_missing_route_requires_exact_configuration(status: int) -> None:
    client = adapter(profile(), lambda _: httpx.Response(status, json={"id": "arbitrary/model"}))
    try:
        with pytest.raises(RunnerError) as exc:
            await client.discover_capabilities(deadline())
        assert exc.value.code == "unsupported_capability"
        assert exc.value.details["required_keys"] == ["context_window_tokens", "max_output_tokens"]
    finally:
        await client.aclose()


@pytest.mark.parametrize("status", [401, 403, 302])
async def test_discovery_auth_and_redirects_do_not_fallback(status: int) -> None:
    client = adapter(
        profile(context_window_tokens=100, max_output_tokens=10),
        lambda _: httpx.Response(status, headers={"location": "https://elsewhere.test/"}),
    )
    try:
        with pytest.raises(RunnerError):
            await client.discover_capabilities(deadline())
    finally:
        await client.aclose()


@pytest.mark.parametrize("tool_calls", [{}, "", False, 0])
async def test_wrong_tool_call_container_is_not_silently_empty(tool_calls: Any) -> None:
    payload = reply()
    payload["choices"][0]["message"]["tool_calls"] = tool_calls
    client = adapter(profile(), lambda _: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError, match="model_protocol_error"):
            await client.complete([], [], 10, deadline())
    finally:
        await client.aclose()


async def test_duplicate_tool_ids_rejected_before_dispatch() -> None:
    call = {"id": "same", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    payload = reply(
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "tool_calls": [call, call]},
            }
        ]
    )
    client = adapter(profile(), lambda _: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError, match="model_protocol_error"):
            await client.complete([], [], 10, deadline())
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 9},
        {},
        None,
    ],
)
async def test_unusable_usage_left_for_coordinator_estimation(usage: Any) -> None:
    client = adapter(profile(), lambda _: httpx.Response(200, json=reply(usage=usage)))
    try:
        assert (await client.complete([], [], 10, deadline())).usage is None
    finally:
        await client.aclose()


@pytest.mark.parametrize("context,output", [(False, 10), (100, 0), (100, 101), ("100", 10)])
async def test_invalid_discovered_capacity_blocked(context: Any, output: Any) -> None:
    client = adapter(
        profile(
            discovery=Discovery(
                path="metadata", context_window_field="context", max_output_field="output"
            )
        ),
        lambda _: httpx.Response(200, json={"context": context, "output": output}),
    )
    try:
        with pytest.raises(RunnerError, match="unsupported_capability"):
            await client.discover_capabilities(deadline())
    finally:
        await client.aclose()


async def test_metadata_fallback_and_bounded_body() -> None:
    client = adapter(
        profile(context_window_tokens=100, max_output_tokens=10), lambda _: httpx.Response(404)
    )
    try:
        result = await client.discover_capabilities(deadline())
        assert result.sources["context_window_tokens"] == "configured"
        assert result.discovered == {}
    finally:
        await client.aclose()
    client = adapter(
        profile(context_window_tokens=100, max_output_tokens=10),
        lambda _: httpx.Response(200, content=b" " * (1_048_576 + 1)),
    )
    try:
        with pytest.raises(RunnerError, match="model_protocol_error"):
            await client.discover_capabilities(deadline())
    finally:
        await client.aclose()


async def test_stateful_compatible_endpoint_exchange() -> None:
    class LocalEndpoint(httpx.AsyncBaseTransport):
        count = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.count += 1
            body = json.loads(request.content)
            assert request.url.path == "/inference/chat/completions"
            if self.count == 1:
                assert body["tools"][0]["function"]["name"] == "read"
                return httpx.Response(
                    200,
                    json=reply(
                        choices=[
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "read-1",
                                            "type": "function",
                                            "function": {
                                                "name": "read",
                                                "arguments": '{ "path": "x" }',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    ),
                )
            assert body["messages"][-1]["tool_call_id"] == "read-1"
            return httpx.Response(200, json=reply())

    from skillrunner.model import OpenAICompatibleAdapter

    transport = LocalEndpoint()
    client = OpenAICompatibleAdapter(
        profile().model_copy(update={"base_url": "http://local.test/inference"}),
        transport=transport,
    )
    schemas = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    try:
        result = await client.complete(
            [{"role": "user", "content": "Read"}], schemas, 10, deadline()
        )
        call = result.tool_calls[0]
        assert call.raw_arguments == '{ "path": "x" }'
        result = await client.complete(
            [{"role": "tool", "tool_call_id": call.id, "content": "file"}], schemas, 10, deadline()
        )
        assert result.public_text == "Done"
        assert transport.count == 2
    finally:
        await client.aclose()


async def test_default_metadata_route_preserves_literal_model_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path == b"/custom/api/models/vendor%3Amodel%2Fversion"
        assert "authorization" not in request.headers
        return httpx.Response(404)

    selected = profile(context_window_tokens=100, max_output_tokens=10).model_copy(
        update={"model": "vendor:model/version"}
    )
    client = adapter(selected, handler)
    try:
        result = await client.discover_capabilities(deadline())
        assert result.profile.model == "vendor:model/version"
    finally:
        await client.aclose()


async def test_no_request_after_deadline_or_above_output_capacity() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("No request should be sent")

    client = adapter(profile(max_output_tokens=10), handler)
    try:
        with pytest.raises(RunnerError, match="model_timeout"):
            await client.complete([], [], 5, asyncio.get_running_loop().time() - 1)
        with pytest.raises(RunnerError, match="context_capacity_exceeded"):
            await client.complete([], [], 11, deadline())
    finally:
        await client.aclose()


async def test_provider_context_capacity_failure_has_limit_status() -> None:
    client = adapter(
        profile(),
        lambda _: httpx.Response(
            400,
            json={
                "error": {"code": "context_length_exceeded", "message": "PRIVATE prompt details"}
            },
        ),
    )
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 10, deadline())
        assert exc.value.code == "context_capacity_exceeded"
        assert exc.value.status == "limit_exceeded"
        assert "PRIVATE" not in str(exc.value)
    finally:
        await client.aclose()


async def test_deep_metadata_response_is_safe_protocol_failure(monkeypatch: Any) -> None:
    def exhausted_parser(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("PRIVATE metadata details")

    client = adapter(profile(), lambda _: httpx.Response(200, content=b"{}"))
    monkeypatch.setattr(json, "loads", exhausted_parser)
    try:
        with pytest.raises(RunnerError, match="model_protocol_error") as exc:
            await client.discover_capabilities(deadline())
        assert "PRIVATE" not in str(exc.value)
    finally:
        await client.aclose()


@pytest.mark.parametrize("arguments", ['{"value":1e400}', '{"nested":[{"value":-1e400}]}'])
async def test_tool_arguments_reject_overflowed_json_numbers(arguments: str) -> None:
    payload = reply(
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "one",
                            "type": "function",
                            "function": {"name": "run", "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    )
    client = adapter(profile(), lambda _: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError, match="model_protocol_error"):
            await client.complete([], [], 10, deadline())
    finally:
        await client.aclose()


async def test_completion_json_integer_limit_failure_is_safe() -> None:
    content = b'{"PRIVATE":' + b"9" * 5000 + b"}"
    client = adapter(profile(), lambda _: httpx.Response(200, content=content))
    try:
        with pytest.raises(RunnerError, match="model_protocol_error") as exc:
            await client.complete([], [], 10, deadline())
        assert "PRIVATE" not in str(exc.value)
    finally:
        await client.aclose()


@pytest.mark.parametrize("content", [None, "", "Partial answer"])
@pytest.mark.parametrize("arguments", [None, '{"path":', '{"path":"scratch/file"}'])
async def test_length_response_preserves_usage_without_action_batch(content, arguments):
    message = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "one",
                "type": "function",
                "function": {
                    "name": "read_text",
                    "arguments": arguments,
                },
            }
        ]
    payload = reply(
        choices=[{"finish_reason": "length", "message": message}],
        usage={"prompt_tokens": 100, "completion_tokens": 17, "total_tokens": 117},
    )
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        result = await client.complete([], [], 17, deadline())
        assert result.finish_reason == "length"
        assert result.public_text == content
        assert result.tool_calls == ()
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 17
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "partial"},
        {"role": "assistant", "content": {}},
        {"role": "assistant", "tool_calls": "invalid"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "read_text", "arguments": {}},
                }
            ],
        },
    ],
)
async def test_length_response_still_rejects_malformed_envelopes(message):
    payload = reply(choices=[{"finish_reason": "length", "message": message}])
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError) as caught:
            await client.complete([], [], 17, deadline())
        assert caught.value.code == "model_protocol_error"
    finally:
        await client.aclose()


async def test_truncated_arguments_without_usage_are_counted_but_not_dispatchable():
    from skillrunner.runtime.agent import returned_output_estimate

    arguments = '{"content":"' + "界" * 500
    payload = reply(
        choices=[
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "Partial",
                    "tool_calls": [
                        {
                            "id": "one",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": arguments,
                            },
                        }
                    ],
                    "reasoning_content": "PRIVATE_REASONING" * 1000,
                },
            }
        ]
    )
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        result = await client.complete([], [], 4096, deadline())
        assert result.usage is None
        assert result.tool_calls == ()
        assert len(arguments.encode("utf-8")) <= returned_output_estimate(result) < 2000
        assert "PRIVATE_REASONING" not in repr(result)
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "choice,field",
    [
        (None, "choices"),
        ({"finish_reason": "PRIVATE", "message": {}}, "choices[0].finish_reason"),
        ({"finish_reason": "stop", "message": {"role": "PRIVATE"}}, "choices[0].message.role"),
        (
            {"finish_reason": "stop", "message": {"role": "assistant", "content": {"PRIVATE": 1}}},
            "choices[0].message.content",
        ),
        (
            {
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "tool_calls": "PRIVATE"},
            },
            "choices[0].message.tool_calls",
        ),
        (
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "one",
                            "type": "function",
                            "function": {"name": "read_text", "arguments": "PRIVATE"},
                        }
                    ],
                },
            },
            "choices[0].message.tool_calls[].function.arguments",
        ),
        (
            {"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "PRIVATE"}},
            "completion_consistency",
        ),
    ],
)
async def test_protocol_errors_identify_safe_response_field(choice, field):
    payload = reply(choices=[] if choice is None else [choice])
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 17, deadline())
        assert exc.value.code == "model_protocol_error"
        assert exc.value.details["response_field"] == field
        assert exc.value.details.get("retryable") is not True
        assert "PRIVATE" not in str(exc.value)
        assert "PRIVATE" not in json.dumps(exc.value.details)
    finally:
        await client.aclose()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
@pytest.mark.parametrize("tools", [None, []])
async def test_empty_stop_is_retryable_but_still_protocol_error(content, tools):
    payload = reply(
        choices=[
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": tools},
            }
        ]
    )
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 17, deadline())
        assert exc.value.code == "model_protocol_error"
        assert exc.value.details["retryable"] is True
        assert exc.value.details["response_field"] == "completion_consistency"
    finally:
        await client.aclose()


@pytest.mark.parametrize("finish", ["tool_calls", "content_filter"])
async def test_other_empty_terminal_shapes_are_not_retryable(finish):
    payload = reply(
        choices=[{"finish_reason": finish, "message": {"role": "assistant", "content": None}}]
    )
    client = adapter(profile(), lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(RunnerError) as exc:
            await client.complete([], [], 17, deadline())
        assert exc.value.details.get("retryable") is not True
    finally:
        await client.aclose()
