"""Supervised MCP connections with one owner task per SDK session."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
from mcp import ClientSession, types
from mcp.client._transport import TransportStreams
from mcp.client.session import ClientRequestContext
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage

from skillrunner.config.models import MCPConfig, Policy
from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import build_child_environment
from skillrunner.runtime.processes import ProcessSupervisor, SupervisedProcess


class _BoundedStream(httpx2.AsyncByteStream):
    def __init__(self, stream: httpx2.AsyncByteStream, limit: int) -> None:
        self.stream, self.limit = stream, limit

    async def __aiter__(self) -> AsyncIterator[bytes]:
        consumed = 0
        async for chunk in self.stream:
            consumed += len(chunk)
            if consumed > self.limit:
                raise RunnerError("budget_exhausted", "MCP HTTP response exceeds byte limit.")
            yield chunk

    async def aclose(self) -> None:
        await self.stream.aclose()


class _BoundedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport: httpx2.AsyncBaseTransport, limit: int) -> None:
        self.transport, self.limit = transport, limit
        self.error: RunnerError | None = None

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self.transport.handle_async_request(request)
        if response.status_code in {401, 403}:
            self.error = RunnerError(
                "missing_credential" if response.status_code == 401 else "mcp_permission_denied",
                "MCP endpoint requires configured authorization.",
                status="blocked",
                exit_code=4,
            )
            await response.aclose()
            raise self.error
        if response.headers.get("content-encoding", "identity") != "identity":
            await response.aclose()
            raise RunnerError("mcp_protocol_error", "Compressed MCP responses are unsupported.")
        return httpx2.Response(
            response.status_code,
            headers=response.headers,
            stream=_BoundedStream(cast(httpx2.AsyncByteStream, response.stream), self.limit),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self.transport.aclose()


@dataclass(frozen=True)
class MCPTool:
    name: str
    server: str
    original_name: str
    description: str
    input_schema: dict[str, Any]


class _Session(ClientSession):
    # Explicitly unsupported callbacks record why a tool cannot complete, while
    # never advertising support or initiating interactive/model operations.
    def _build_capabilities(self, version: str) -> types.ClientCapabilities:
        result = super()._build_capabilities(version)
        result.sampling = None
        result.elicitation = None
        return result


@asynccontextmanager
async def _pipes(
    child: SupervisedProcess, max_bytes: int, diagnostic: bytearray
) -> AsyncIterator[
    tuple[
        anyio.streams.memory.MemoryObjectReceiveStream[SessionMessage | Exception],
        anyio.streams.memory.MemoryObjectSendStream[SessionMessage],
    ]
]:
    incoming, read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    write, outgoing = anyio.create_memory_object_stream[SessionMessage](1)

    async def receive() -> None:
        pending = bytearray()
        try:
            async with incoming:
                while chunk := await child.stdout.read(min(65536, max_bytes + 1)):
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        if len(line) > max_bytes:
                            raise ValueError("Protocol frame exceeds limit")
                        await incoming.send(
                            SessionMessage(types.jsonrpc_message_adapter.validate_json(line))
                        )
                    if len(pending) > max_bytes:
                        raise ValueError("Protocol frame exceeds limit")
                if pending:
                    raise ValueError("Incomplete protocol frame")
        except Exception:
            # Closing the read stream fails pending SDK requests without leaking
            # arbitrary protocol bytes into public diagnostics.
            await incoming.aclose()

    async def send() -> None:
        try:
            await send_messages()
        except Exception:
            # Ending the incoming stream also fails pending SDK request futures.
            await incoming.aclose()

    async def send_messages() -> None:
        assert child.stdin is not None
        async with outgoing:
            async for message in outgoing:
                data = (
                    message.message.model_dump_json(by_alias=True, exclude_none=True).encode()
                    + b"\n"
                )
                if len(data) > max_bytes:
                    raise RunnerError("budget_exhausted", "MCP request exceeds byte limit.")
                for offset in range(0, len(data), 65536):
                    child.stdin.write(data[offset : offset + 65536])
                    await child.stdin.drain()

    async def stderr() -> None:
        while chunk := await child.stderr.read():
            diagnostic.extend(chunk[: max(0, max_bytes - len(diagnostic))])

    tasks = [asyncio.create_task(operation()) for operation in (receive, send, stderr)]
    try:
        yield read, write
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await read.aclose()
        await write.aclose()


class _Connection:
    def __init__(self, manager: "MCPManager", server: str, config: MCPConfig) -> None:
        self.manager, self.server, self.config = manager, server, config
        self.queue: asyncio.Queue[
            tuple[str, dict[str, Any], asyncio.Future[dict[str, Any]]] | None
        ] = asyncio.Queue()
        self.ready: asyncio.Future[list[types.Tool]] = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self.run())
        self.stderr = bytearray()
        self.interaction: RunnerError | None = None
        self.current: asyncio.Future[dict[str, Any]] | None = None
        self.close_error: RunnerError | None = None
        self.http_transport: _BoundedTransport | None = None

    async def unsupported_sampling(
        self, context: ClientRequestContext, params: types.CreateMessageRequestParams
    ) -> types.ErrorData:
        self.interaction = RunnerError("unsupported_capability", "MCP sampling is unsupported.")
        return types.ErrorData(code=types.INVALID_REQUEST, message="Sampling unsupported")

    async def unsupported_elicitation(
        self, context: ClientRequestContext, params: types.ElicitRequestParams
    ) -> types.ErrorData:
        self.interaction = RunnerError("missing_decision", "MCP server requires human input.")
        return types.ErrorData(code=types.INVALID_REQUEST, message="Interactive input unavailable")

    async def run(self) -> None:
        manager, config = self.manager, self.config
        exiting = False
        try:
            async with AsyncExitStack() as stack:
                streams: TransportStreams
                if config.transport == "stdio":
                    assert config.command is not None
                    child = await stack.enter_async_context(
                        manager.supervisor.open(
                            config.command,
                            config.args,
                            cwd=config.cwd or manager.cwd,
                            environment=build_child_environment(
                                manager.environ, references=config.env
                            ),
                            deadline=manager.deadline,
                        )
                    )
                    streams = await stack.enter_async_context(
                        _pipes(child, manager.max_bytes, self.stderr)
                    )
                else:
                    assert config.url is not None
                    headers = {"Accept-Encoding": "identity"}
                    for name, reference in config.headers.items():
                        if reference not in manager.environ:
                            raise RunnerError(
                                "missing_credential", "Required MCP credential is missing."
                            )
                        headers[name] = manager.environ[reference]
                    self.http_transport = _BoundedTransport(
                        manager.http_transport
                        or httpx2.AsyncHTTPTransport(trust_env=False, retries=0),
                        manager.max_bytes,
                    )
                    client = await stack.enter_async_context(
                        httpx2.AsyncClient(
                            headers=headers,
                            trust_env=False,
                            follow_redirects=False,
                            timeout=manager.deadline.remaining,
                            transport=self.http_transport,
                        )
                    )
                    streams = await stack.enter_async_context(
                        streamable_http_client(config.url, http_client=client)
                    )
                session = await stack.enter_async_context(
                    _Session(
                        streams[0],
                        streams[1],
                        sampling_callback=self.unsupported_sampling,
                        elicitation_callback=self.unsupported_elicitation,
                    )
                )
                async with asyncio.timeout(manager.deadline.remaining):
                    await session.initialize()
                    if self.interaction is not None:
                        raise self.interaction
                    catalog = []
                    cursor = None
                    seen = set()
                    while True:
                        listing = await session.list_tools(
                            params=types.PaginatedRequestParams(cursor=cursor) if cursor else None
                        )
                        if self.interaction is not None:
                            raise self.interaction
                        catalog.extend(listing.tools)
                        manager.bounded(
                            [tool.model_dump(by_alias=True, exclude_none=True) for tool in catalog]
                        )
                        cursor = listing.next_cursor
                        if not cursor:
                            break
                        if cursor in seen:
                            raise RunnerError(
                                "mcp_protocol_error", "MCP tool pagination repeated a cursor."
                            )
                        seen.add(cursor)
                self.ready.set_result(catalog)
                while True:
                    operation = await self.queue.get()
                    if operation is None:
                        break
                    name, arguments, future = operation
                    if future.cancelled():
                        continue
                    self.current = future
                    self.interaction = None
                    try:
                        manager.deadline.check()
                        async with asyncio.timeout(manager.deadline.remaining):
                            result = await session.call_tool(name, arguments)
                        if self.interaction:
                            raise self.interaction
                        value = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                        manager.bounded(value)
                        future.set_result(value)
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(self.failure(exc, dispatched=True))
                        break
                    finally:
                        self.current = None
                exiting = True
        except BaseException as exc:
            if exiting:
                self.close_error = RunnerError(
                    "mcp_cleanup_failed",
                    "MCP connection cleanup failed.",
                    details={
                        "cause_type": type(exc).__name__,
                        "cause_code": exc.code if isinstance(exc, RunnerError) else None,
                        "cause_os_type": (
                            type(exc.__cause__).__name__ if exc.__cause__ is not None else None
                        ),
                        "cause_os_errno": getattr(exc.__cause__, "errno", None),
                    },
                )
            error = self.failure(exc, dispatched=self.current is not None)
            if not self.ready.done():
                self.ready.set_exception(error)
            if self.current is not None and not self.current.done():
                self.current.set_exception(error)
        finally:
            while not self.queue.empty():
                operation = self.queue.get_nowait()
                if operation is not None and not operation[2].done():
                    operation[2].set_exception(
                        RunnerError("mcp_connection_closed", "MCP connection closed.")
                    )

    def failure(self, exc: BaseException, *, dispatched: bool) -> RunnerError:
        if isinstance(exc, TimeoutError) or self.manager.deadline.remaining <= 0:
            error = RunnerError("budget_exhausted", "Execution timeout reached.")
        elif isinstance(exc, asyncio.CancelledError):
            error = RunnerError("cancelled", "MCP operation was cancelled.")
        elif self.interaction is not None:
            error = self.interaction
        elif self.http_transport is not None and self.http_transport.error is not None:
            error = self.http_transport.error
        elif isinstance(exc, RunnerError):
            error = exc
        else:
            error = RunnerError(
                "external_outcome_unknown" if dispatched else "mcp_connection_failed",
                "MCP response unavailable." if dispatched else "MCP connection failed.",
            )
        if dispatched:
            error.details["outcome_unknown"] = True
            error.details["retryable"] = False
            error.details["suggested_action"] = (
                "Check the remote service state before rerunning; "
                "the operation may already have completed."
            )
        return error

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.task.done():
            raise RunnerError("mcp_connection_closed", "MCP connection closed.")
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        await self.queue.put((name, arguments, future))
        try:
            return await future
        except asyncio.CancelledError as exc:
            cast(Any, exc).outcome_unknown = self.current is future
            self.task.cancel()
            raise

    async def aclose(self) -> None:
        if not self.ready.done() or self.ready.cancelled() or self.current is not None:
            self.task.cancel()
        await self.queue.put(None)
        await asyncio.shield(self.task)
        if self.close_error is not None:
            raise self.close_error


class MCPManager:
    def __init__(
        self,
        configs: dict[str, MCPConfig],
        policy: Policy,
        supervisor: ProcessSupervisor,
        environ: Mapping[str, str],
        deadline: Deadline,
        max_bytes: int = 1_048_576,
        *,
        cwd: Path | None = None,
        http_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 2:
            raise ValueError("MCP byte limit must be at least two")
        self.configs, self.policy, self.supervisor = configs, policy, supervisor
        self.environ, self.deadline, self.max_bytes = environ, deadline, max_bytes
        self.cwd, self.http_transport = cwd or Path.cwd(), http_transport
        self.tools: list[MCPTool] = []
        self.reverse_map: dict[str, tuple[str, str]] = {}
        self.diagnostics: list[dict[str, str]] = []
        self.connections: dict[str, _Connection] = {}

    def bounded(self, value: Any) -> None:
        try:
            size = len(json.dumps(value, allow_nan=False, ensure_ascii=True).encode())
        except (ValueError, TypeError, RecursionError):
            raise RunnerError("invalid_arguments", "MCP data must be finite JSON.") from None
        if size > self.max_bytes:
            raise RunnerError("budget_exhausted", "MCP data exceeds byte limit.")

    async def connect(self) -> None:
        for server, config in sorted(self.configs.items()):
            self.deadline.check()
            connection = _Connection(self, server, config)
            self.connections[server] = connection
            catalog = await connection.ready
            discovered = set()
            for tool in catalog:
                if tool.name in discovered:
                    raise RunnerError("mcp_protocol_error", "MCP returned duplicate tool names.")
                discovered.add(tool.name)
                if tool.name not in config.allowed_tools:
                    continue
                schema = tool.input_schema
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    raise RunnerError(
                        "mcp_protocol_error", "MCP tool schema must describe an object."
                    )
                self.bounded(schema)
                digest = hashlib.sha256(json.dumps([server, tool.name]).encode()).hexdigest()
                name = "mcp_" + digest[:60]
                if name in self.reverse_map:
                    raise RunnerError("mcp_protocol_error", "MCP tool identifier collision.")
                self.reverse_map[name] = (server, tool.name)
                self.tools.append(MCPTool(name, server, tool.name, tool.description or "", schema))
            if set(config.allowed_tools) - discovered:
                self.diagnostics.append({"server": server, "code": "mcp_tools_unavailable"})

    async def invoke(self, model_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.deadline.check()
        if model_name not in self.reverse_map:
            raise RunnerError("mcp_tool_not_allowed", "MCP tool is not allowed.")
        server, name = self.reverse_map[model_name]
        if name not in self.configs[server].allowed_tools:
            raise RunnerError("mcp_tool_not_allowed", "MCP tool is not allowed.")
        if not isinstance(arguments, dict) or any(not isinstance(k, str) for k in arguments):
            raise RunnerError("invalid_arguments", "MCP arguments must be a JSON object.")
        self.bounded(arguments)
        return await self.connections[server].invoke(name, arguments)

    async def aclose(self) -> None:
        first_error: BaseException | None = None
        for server, connection in self.connections.items():
            try:
                await connection.aclose()
            except BaseException as exc:
                self.diagnostics.append({"server": server, "code": "mcp_cleanup_failed"})
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note("Another MCP connection failed to close.")
        if first_error is not None:
            raise first_error
