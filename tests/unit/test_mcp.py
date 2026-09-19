import json

import httpx2
import pytest

from skillrunner.config.models import MCPConfig, Policy
from skillrunner.domain.errors import RunnerError
from skillrunner.mcp import MCPManager
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.processes import ProcessSupervisor


def manager(config, **kwargs):
    policy = Policy()
    return MCPManager(
        {"remote": config},
        policy,
        ProcessSupervisor(policy),
        {"TOKEN": "secret"},
        Deadline(3),
        **kwargs,
    )


async def test_http_explicit_credentials_catalog_and_result():
    requests = []

    async def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "secret"
        if request.method != "POST":
            return httpx2.Response(405)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        method = body["method"]
        if method == "initialize":
            assert "sampling" not in body["params"]["capabilities"]
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "local", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "run",
                        "inputSchema": {"type": "object", "required": ["server_validates"]},
                    }
                ]
            }
        else:
            assert method == "tools/call"
            result = {"content": [{"type": "text", "text": "done"}]}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    bridge = manager(
        MCPConfig(
            transport="streamable-http",
            url="https://example.invalid/mcp",
            headers={"authorization": "TOKEN"},
            allowed_tools=["run"],
        ),
        http_transport=httpx2.MockTransport(handle),
    )
    try:
        await bridge.connect()
        result = await bridge.invoke(bridge.tools[0].name, {})
        assert result["content"][0]["text"] == "done"
        with pytest.raises(RunnerError, match="invalid_arguments"):
            await bridge.invoke(bridge.tools[0].name, [])
        with pytest.raises(RunnerError, match="invalid_arguments"):
            await bridge.invoke(bridge.tools[0].name, {"x": float("nan")})
    finally:
        await bridge.aclose()
    assert len([r for r in requests if r.method == "POST"]) == 4


async def test_missing_http_reference_fails_without_network():
    bridge = manager(
        MCPConfig(
            transport="streamable-http",
            url="https://example.invalid/mcp",
            headers={"authorization": "MISSING"},
        )
    )
    try:
        with pytest.raises(RunnerError, match="missing_credential"):
            await bridge.connect()
    finally:
        await bridge.aclose()


async def test_http_body_is_bounded_before_sdk_decoding():
    from skillrunner.mcp import _BoundedTransport

    transport = _BoundedTransport(
        httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"x" * 2048)), 1024
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        with pytest.raises(RunnerError, match="budget_exhausted"):
            await client.get("https://example.invalid")


async def test_close_attempts_every_connection_and_preserves_first_failure():
    bridge = manager(MCPConfig(transport="streamable-http", url="https://example.invalid"))
    closed = []
    first = RunnerError("mcp_cleanup_failed", "First connection did not close.")

    class Connection:
        def __init__(self, name, error=None):
            self.name, self.error = name, error

        async def aclose(self):
            closed.append(self.name)
            if self.error:
                raise self.error

    bridge.connections = {"one": Connection("one", first), "two": Connection("two")}
    with pytest.raises(RunnerError) as caught:
        await bridge.aclose()
    assert closed == ["one", "two"]
    assert caught.value is first
    assert bridge.diagnostics[-1]["code"] == "mcp_cleanup_failed"


async def test_sdk_transport_cleanup_failure_is_reported():
    async def handle(request):
        if request.method != "POST":
            return httpx2.Response(405)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "local", "version": "1"},
            }
        else:
            result = {"tools": []}
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    class BrokenClose(httpx2.MockTransport):
        async def aclose(self):
            raise OSError("sensitive transport text")

    bridge = manager(
        MCPConfig(transport="streamable-http", url="https://example.invalid"),
        http_transport=BrokenClose(handle),
    )
    await bridge.connect()
    with pytest.raises(RunnerError, match="mcp_cleanup_failed") as caught:
        await bridge.aclose()
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("status", [401, 403])
async def test_http_authentication_failures_are_blocked(status):
    bridge = manager(
        MCPConfig(transport="streamable-http", url="https://example.invalid"),
        http_transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                status, headers={"WWW-Authenticate": "Bearer secret-challenge"}, text="secret-body"
            )
        ),
    )
    try:
        with pytest.raises(RunnerError) as caught:
            await bridge.connect()
        assert caught.value.status == "blocked"
        assert "secret" not in str(caught.value)
    finally:
        await bridge.aclose()
