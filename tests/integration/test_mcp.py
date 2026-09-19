import asyncio
import json
import sys
from pathlib import Path

import pytest

from skillrunner.config.models import MCPConfig, Policy
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.processes import ProcessSupervisor

SERVER = r"""
import sys,json,os
for line in sys.stdin:
 r=json.loads(line); method=r.get('method')
 if 'id' not in r: continue
 if method=='initialize':
  assert 'sampling' not in r['params']['capabilities']
  result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'local','version':'1'}}
 elif method=='tools/list':
  result={'tools':[{'name':name,'inputSchema':{'type':'object'}}
                   for name in ('echo','denied')]}
 elif method=='tools/call':
  if r['params']['arguments'].get('disconnect'): sys.exit(0)
  result={'content':[{'type':'text','text':json.dumps(r['params']['arguments'])}]}
 else: continue
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
"""


def manager(tmp_path, **kwargs):
    from skillrunner.mcp import MCPManager

    executable = str(Path(sys.executable).resolve())
    stat = Path(executable).stat()
    policy = Policy(
        allowed_executables=[executable],
        executable_identities={
            executable: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        },
    )
    supervisor = ProcessSupervisor(policy, shutdown_grace=0)
    config = MCPConfig(
        transport="stdio",
        command=executable,
        args=["-c", SERVER],
        cwd=tmp_path,
        allowed_tools=["echo", "absent"],
    )
    return MCPManager({"local": config}, policy, supervisor, {}, Deadline(5), **kwargs), supervisor


async def test_stdio_catalog_invoke_and_close_from_other_task(tmp_path):
    bridge, supervisor = manager(tmp_path)
    await bridge.connect()
    try:
        assert len(bridge.tools) == 1
        tool = bridge.tools[0]
        assert bridge.reverse_map[tool.name] == ("local", "echo")
        assert bridge.diagnostics
        result = await bridge.invoke(tool.name, {"hello": "world"})
        assert json.loads(result["content"][0]["text"]) == {"hello": "world"}
        with pytest.raises(RunnerError, match="mcp_tool_not_allowed"):
            await bridge.invoke("denied", {})
    finally:
        try:
            await asyncio.create_task(bridge.aclose())
        except RunnerError as error:
            pytest.fail(f"{error.code}: {error.details}")
    assert not supervisor.active


async def test_disconnected_write_is_unknown_without_retry(tmp_path):
    bridge, supervisor = manager(tmp_path)
    await bridge.connect()
    try:
        with pytest.raises(RunnerError) as caught:
            await bridge.invoke(bridge.tools[0].name, {"disconnect": True})
        assert caught.value.details["outcome_unknown"] is True
    finally:
        await bridge.aclose()
    assert not supervisor.active


@pytest.mark.parametrize(
    "method,params,code",
    [
        ("sampling/createMessage", {"messages": [], "maxTokens": 1}, "unsupported_capability"),
        (
            "elicitation/create",
            {"message": "Input required", "requestedSchema": {"type": "object", "properties": {}}},
            "missing_decision",
        ),
    ],
)
async def test_server_interaction_is_rejected_without_prompt(tmp_path, method, params, code):
    bridge, supervisor = manager(tmp_path)
    request = {"jsonrpc": "2.0", "id": "server-request", "method": method, "params": params}
    injection = (
        "print("
        + repr(json.dumps(request))
        + ",flush=True)\n  response=json.loads(sys.stdin.readline())\n"
        + "  assert 'error' in response\n  "
    )
    bridge.configs["local"].args[1] = SERVER.replace(
        "result={'content':", injection + "result={'content':"
    )
    try:
        await bridge.connect()
        with pytest.raises(RunnerError) as caught:
            await bridge.invoke(bridge.tools[0].name, {})
        assert caught.value.code == code
    finally:
        await bridge.aclose()
    assert not supervisor.active


async def test_protocol_stdout_frame_limit(tmp_path):
    bridge, supervisor = manager(tmp_path, max_bytes=1024)
    bridge.configs["local"].args[1] = SERVER.replace(
        "result={'content':", "print('x'*2048,flush=True)\n  result={'content':"
    )
    try:
        await bridge.connect()
        with pytest.raises(RunnerError) as caught:
            await bridge.invoke(bridge.tools[0].name, {})
        assert caught.value.details["outcome_unknown"]
    finally:
        await bridge.aclose()
    assert not supervisor.active


async def test_close_interrupts_unresponsive_handshake(tmp_path):
    bridge, supervisor = manager(tmp_path)
    bridge.configs["local"].args[1] = "import time;time.sleep(60)"
    startup = asyncio.create_task(bridge.connect())
    while not supervisor.active:
        await asyncio.sleep(0.001)
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    await asyncio.wait_for(bridge.aclose(), 0.5)
    assert not supervisor.active


async def test_deadline_during_remote_call_keeps_unknown_metadata(tmp_path):
    bridge, supervisor = manager(tmp_path)
    bridge.deadline = Deadline(0.3)
    bridge.configs["local"].args[1] = SERVER.replace(
        "result={'content':", "import time;time.sleep(60)\n  result={'content':"
    )
    try:
        await bridge.connect()
        with pytest.raises(RunnerError) as caught:
            await asyncio.wait_for(bridge.invoke(bridge.tools[0].name, {}), 1)
        assert caught.value.code == "budget_exhausted"
        assert caught.value.details["outcome_unknown"]
    finally:
        await bridge.aclose()
    assert not supervisor.active


async def test_oversized_wire_request_does_not_strand_caller(tmp_path):
    bridge, supervisor = manager(tmp_path, max_bytes=1024)
    try:
        await bridge.connect()
        # Arguments fit the generic JSON bound but not the JSON-RPC envelope.
        with pytest.raises(RunnerError):
            await asyncio.wait_for(bridge.invoke(bridge.tools[0].name, {"x": "a" * 990}), 0.5)
    finally:
        await bridge.aclose()
    assert not supervisor.active


@pytest.mark.parametrize("disconnect", [False, True])
async def test_full_run_uses_supervised_mcp_and_closes_it(tmp_path, disconnect):
    from skillrunner.domain.request import RunRequest
    from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage
    from skillrunner.runtime.coordinator import run_task
    from tests.integration.test_coordinator import Adapter, fixture

    bridge, _ = manager(tmp_path)
    settings = fixture(tmp_path)
    settings.policy = bridge.policy
    settings.mcp = bridge.configs
    settings.limits.shutdown_grace = 0
    adapters = []

    class Model(Adapter):
        async def complete(self, messages, tool_schemas, *args):
            self.requests.append(messages)
            if len(self.requests) == 1:
                remote = next(
                    item["function"]
                    for item in tool_schemas
                    if item["function"]["name"].startswith("mcp_")
                )
                name, arguments = remote["name"], {"disconnect": disconnect}
            else:
                assert not disconnect, "Unknown external outcomes must stop without a retry"
                result = json.loads(messages[-1]["content"])
                assert result["value"]["content"][0]["text"] == '{"disconnect": false}'
                name, arguments = "finish_run", {"outcome": "succeeded", "report": "Done"}
            return ModelReply(
                None,
                (ModelToolCall(str(len(self.requests)), name, arguments, json.dumps(arguments)),),
                "tool_calls",
                ModelUsage(10, 10, 20),
                None,
            )

    def factory(profile, key):
        adapter = Model(profile, [])
        adapters.append(adapter)
        return adapter

    receipt = await run_task(
        RunRequest(prompt="Use remote", invocation_directory=tmp_path, required_skills=["writer"]),
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert receipt["status"] == ("failed" if disconnect else "succeeded"), receipt
    assert adapters[0].closed
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["lifecycle"]["cleanup"]["owned_pids_remaining"] == []
    assert manifest["diagnostics"]["mcp"][0]["code"] == "mcp_tools_unavailable"
    if disconnect:
        assert receipt["errors"][0]["outcome_certainty"] == "unknown"
        error = receipt["errors"][0]
        assert error["code"] == "external_outcome_unknown"
        assert receipt["exit_code"] == 6
        assert error["retryable"] is False
        assert "remote service state" in error["suggested_action"]
        assert "may already have completed" in error["suggested_action"]
        manifest = json.loads(Path(receipt["manifest_path"]).read_text())
        assert manifest["diagnostics"]["errors"][0] == error
        assert error["suggested_action"] in Path(receipt["report_path"]).read_text()


@pytest.mark.parametrize(
    "method,params,expected",
    [
        ("sampling/createMessage", {"messages": [], "maxTokens": 1}, "unsupported_capability"),
        (
            "elicitation/create",
            {"message": "Input", "requestedSchema": {"type": "object", "properties": {}}},
            "missing_decision",
        ),
    ],
)
@pytest.mark.parametrize("phase", ["initialize", "tools/call", "tools/list"])
async def test_interaction_reason_survives_server_error(tmp_path, method, params, expected, phase):
    bridge, supervisor = manager(tmp_path)
    request = {"jsonrpc": "2.0", "id": "server-request", "method": method, "params": params}
    injection = (
        " if method==" + repr(phase) + ":\n"
        "  print(" + repr(json.dumps(request)) + ",flush=True)\n"
        "  response=json.loads(sys.stdin.readline())\n"
        "  assert 'error' in response\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':r['id'],'error':"
        "{'code':-32603,'message':'dependency refused'}}),flush=True)\n"
        "  continue\n"
    )
    bridge.configs["local"].args[1] = SERVER.replace(
        " if method=='initialize':", injection + " if method=='initialize':"
    )
    try:
        with pytest.raises(RunnerError) as caught:
            await bridge.connect()
            await bridge.invoke(bridge.tools[0].name, {})
        assert caught.value.code == expected
    finally:
        await bridge.aclose()
    assert not supervisor.active


async def test_configured_large_frames_stream_through_bounded_stdin(tmp_path):
    bridge, supervisor = manager(tmp_path, max_bytes=2 * 1024 * 1024)
    arguments = {"payload": "x" * 1_100_000}
    try:
        await bridge.connect()
        result = await bridge.invoke(bridge.tools[0].name, arguments)
        assert json.loads(result["content"][0]["text"]) == arguments
    finally:
        await bridge.aclose()
    assert not supervisor.active
