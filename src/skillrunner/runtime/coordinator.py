"""Own a task's resources, enforce completion gates, and persist its outcome."""

import asyncio
import copy
import hashlib
import json
import re
import signal
import stat
import uuid
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr

from skillrunner.artifacts.external import validate_external
from skillrunner.artifacts.publication import PublicationTarget
from skillrunner.artifacts.registry import ArtifactRecord, ArtifactRegistry
from skillrunner.artifacts.validation import MEDIA_TYPES, validate_builtin
from skillrunner.catalog.activation import ActivationService
from skillrunner.catalog.discovery import discover
from skillrunner.catalog.snapshots import scan_tree, snapshot_tree, validate_output_locations
from skillrunner.config.models import ModelProfile, ResolvedSettings
from skillrunner.config.secrets import resolve_api_key
from skillrunner.config.sources import select_model
from skillrunner.domain.errors import RunnerError
from skillrunner.domain.request import FORMAT_ALIASES, RunRequest
from skillrunner.mcp import MCPManager
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.recording.bundle import RunBundle
from skillrunner.runtime.agent import AgentLoop
from skillrunner.runtime.budgets import Deadline, UsageLedger
from skillrunner.runtime.cleanup import remove_work_tree
from skillrunner.runtime.context import RunContext
from skillrunner.runtime.environment import build_child_environment
from skillrunner.runtime.processes import ProcessSupervisor
from skillrunner.runtime.storage import check_tree_bytes, monitor_operation
from skillrunner.tools import schemas
from skillrunner.tools.dispatch import ToolRegistry, ToolResult
from skillrunner.tools.files import FileTools

AdapterFactory = Callable[[ModelProfile, SecretStr | None], Any]


def requested_output_filename(prompt: str) -> str | None:
    """Return the basename of a file explicitly named as the output destination."""
    pattern = (
        r"(?:\b(?:write|save|publish|output|deliver|create|produce)\b"
        r"(?:\s+(?:the|a|an|final|primary|output|result|report|file|artifact|deliverable)){0,5}"
        r"\s+(?:to|as|at|named|called)\s+"
        r"|\b(?:output|deliverable|file)\s+(?:should|must|will|shall)\s+be\s+"
        r"(?:written|saved|published)\s+(?:to|as|at|in)\s+)"
        r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([^\s,;:!?)}\]]+))"
    )
    for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
        raw = next(value for value in match.groups() if value is not None).strip()
        if match.group(4) is not None:
            raw = raw.rstrip(".")
        name = re.split(r"[/\\]", raw)[-1]
        if (
            name not in {"", ".", ".."}
            and not re.search(r'[<>:"|?*\x00]', name)
            and (Path(name).suffix or raw != name or match.group(4) is None)
        ):
            return name
    return None


INSTRUCTIONS = """Execute the user's task using installed skills. Catalog and input metadata are
resources, not permission grants. Activate relevant skills before performing their work. Required
skills must all be activated; additional skills are allowed. Activation of a new skill must be in a
separate tool response before actions based on its instructions. Use only supplied tools and roots.
Select the minimum set of skills needed to perform the requested operations. Do not activate an
extra skill solely because its name matches an output format or a reference artifact. When skills
overlap, prefer the one whose description most directly matches the user's goal. When a task
explicitly requires a tool's CLI or wrapper, prefer the skill for that tool.
Otherwise, when testing a supplied local web app, prefer a local-web-app testing skill over a
general browser-automation skill, even if the prompt names the browser technology.
Do not install dependencies or request human interaction. Do not invent paths outside the workspace.
File tools accept paths such as scratch/answer.md, artifacts/data.csv, input-1/source.txt,
or absolute paths inside the registered roots. Root names are directory prefixes, not URI schemes.
Host commands creating output files must use the absolute directories in generated_roots. Logical
paths such as artifacts/data.csv are resolved by file tools; host commands use OS filesystem paths.
For run_command, explicitly set cwd to generated_roots.scratch or generated_roots.artifacts
when creating or packaging output. Its omitted cwd defaults to the activated skill package,
where relative scratch/ and artifacts/ paths do not refer to the generated roots. Use the
absolute paths in generated_roots for command arguments that refer to output files.
Register command-generated files using the same absolute paths used to create them.
Resolve skill-relative references under the package_root returned by activate_skill. Read reference
files required by the activated instructions with read_text before completing their task.
For source reviews, read the task-relevant guidance, including security, accessibility, or styling
references when the request concerns those topics; do not cite a reference you have not read.
When instructions require reading an entire file, truncated=true means the read is incomplete:
read the same path again with offset=next_offset until truncated=false. Before finish_run, check
each activated skill's required references and steps against completed tool results; complete any
missing required reads or actions before claiming success.
Before registering or finishing an artifact, inspect the encoded final file and any files inside
its archive. Check each explicit user requirement against the actual bytes or decoded content. If
a helper loses a required property such as transparency, fix the final file and verify again.
Command success or a validation summary alone is not proof.
New files use
overwrite=false and expected_sha256=null; replacements require the current read_text SHA-256 digest.
A plain Markdown answer may be returned directly in finish_run.report without writing a file.
In that case, report must contain the complete requested deliverable itself; a statement that the
deliverable was created is not a substitute for its content.
Host commands have ordinary OS access, not sandbox isolation. Do not disclose credentials or private
reasoning. Register generated artifacts, then submit finish_run with the requested outcome, public
report, primary artifact ID and secondary IDs. Record any low-impact assumptions in
finish_run.assumptions; required decisions or approvals must not be assumed. A completion proposal
must be the only tool call in its response. Report no_matching_skill if no installed skill applies,
needs_input for a material
missing decision, and blocked for missing required capability. Success is checked by the runner.
"""


class Coordinator:
    def __init__(
        self,
        request: RunRequest,
        settings: ResolvedSettings,
        environ: Mapping[str, str],
        adapter_factory: AdapterFactory,
    ) -> None:
        self.request = request
        self.settings = settings
        self.environ = environ
        self.adapter_factory = adapter_factory
        self.deadline = Deadline(settings.limits.timeout)
        self.ledger = UsageLedger(
            max_steps=settings.limits.max_steps,
            max_tool_calls=settings.limits.max_tool_calls,
            max_tokens=settings.limits.max_tokens,
        )
        self.resources = AsyncExitStack()
        self.registry: ArtifactRegistry | None = None
        self.publication: PublicationTarget | None = None
        self.publication_directory: Path | None = None
        self.publication_directory_identity: tuple[int, int] | None = None
        self.publication_protected_roots: tuple[Path, ...] = ()
        self.stop: RunnerError | None = None
        self.signal_number: int | None = None
        self.terminal_committed = False
        self.answer = ""
        self.keep_work = settings.diagnostics.retain_work
        self.proposal: dict[str, Any] = {}
        self.bundle: RunBundle
        self.files: FileTools
        self.activation: ActivationService
        self.context: RunContext
        self.tools: ToolRegistry
        self.profile: ModelProfile
        self.supervisor = ProcessSupervisor(
            settings.policy,
            shutdown_grace=settings.limits.shutdown_grace,
            max_output_bytes=settings.storage.max_tool_output_bytes,
            on_stopped=lambda pid, returncode: self.event(
                "process_stopped", {"pid": pid, "returncode": returncode}
            ),
        )

    def check(self) -> None:
        self.deadline.check()
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    def event(self, name: str, payload: dict[str, Any]) -> None:
        if self.bundle.events:
            metadata = {key: value for key, value in payload.items() if key != "_content"}
            tools = getattr(self, "tools", None)
            current = tools.current_call if tools is not None else None
            self.bundle.events.emit(
                name,
                metadata,
                content=payload.get("_content"),
                call_id=payload.get("call_id", current.id if current else None),
                tool_name=payload.get(
                    "tool_name",
                    current.name
                    if current
                    else (payload.get("name") if name.startswith("tool_") else None),
                ),
                skill_id=payload.get(
                    "skill_id", payload.get("name") if name == "skill_activated" else None
                ),
            )
        if name in {"tool_completed", "tool_failed"}:
            self.monitor_storage()

    def monitor_storage(self) -> None:
        root = self.bundle.root
        check_tree_bytes(
            [root / "work/scratch", root / "work/staging"],
            self.settings.storage.max_scratch_bytes,
            self.check,
        )
        check_tree_bytes([root / "artifacts"], self.settings.storage.max_artifact_bytes, self.check)

    def phase(self, phase: str) -> None:
        self.bundle.state["lifecycle"]["phase"] = phase
        self.bundle.save()

    def error(self, error: RunnerError) -> None:
        if self.stop is None:
            self.stop = error
            self.bundle.state["lifecycle"]["stop_reason"] = error.code
        details = error.details
        self.bundle.state["diagnostics"]["errors"].append(
            {
                "code": error.code,
                "message": error.message,
                "stage": self.bundle.state["lifecycle"]["phase"],
                "details": details,
                "skill": details.get("skill"),
                "tool": details.get("tool"),
                "retryable": bool(details.get("retryable", False)),
                "outcome_certainty": details.get(
                    "outcome_certainty",
                    "unknown" if details.get("outcome_unknown") else "known",
                ),
                "suggested_action": details.get(
                    "suggested_action",
                    "Inspect the report and configuration before rerunning.",
                ),
            }
        )
        if error.code in {"budget_exhausted", "context_capacity_exceeded"}:
            try:
                self.event(
                    "limit_reached",
                    {
                        "error_code": error.code,
                        "message": error.message,
                        "stage": self.bundle.state["lifecycle"]["phase"],
                        "tool_name": details.get("tool"),
                        "skill_id": details.get("skill"),
                    },
                )
            except (RunnerError, OSError):
                message = "Could not persist the limit event."
                details.setdefault("reporting_errors", []).append(message)
                self.bundle.state["diagnostics"]["warnings"].append(message)

    def cancelled(self, cause: asyncio.CancelledError | None = None) -> None:
        code = "terminated" if self.signal_number == signal.SIGTERM else "cancelled"
        unknown = bool(getattr(cause, "outcome_unknown", False))
        if self.stop is None:
            self.error(RunnerError(code, "Run interrupted.", details={"outcome_unknown": unknown}))
        elif unknown:
            self.stop.details["outcome_unknown"] = True
            for error in self.bundle.state["diagnostics"]["errors"]:
                if error["code"] == self.stop.code:
                    error["outcome_certainty"] = "unknown"
                    error["details"]["outcome_unknown"] = True

        notes = list(getattr(cause, "__notes__", ()))
        if notes and self.stop is not None:
            self.stop.details.setdefault("cleanup_errors", []).extend(notes)
            self.bundle.state["diagnostics"]["warnings"].extend(notes)

    def terminal_outcome(self, status: str, exit_code: int) -> tuple[str, int]:
        # This synchronous decision seals the terminal result before its final write.
        self.terminal_committed = True
        if status == "succeeded" and self.stop is not None:
            return self.stop.status, self.stop.exit_code
        return status, exit_code

    def _secrets(self) -> list[str]:
        # Configuration references are not secrets by themselves. Redacting every
        # mapped value made ordinary values such as PATH, "true", and "1" corrupt
        # otherwise useful diagnostics. Only explicit diagnostics.redact_env names,
        # model credentials, and authentication headers are secret automatically.
        names: list[str] = []

        def add(name: str) -> None:
            if name not in names:
                names.append(name)

        for name in self.settings.diagnostics.redact_env:
            add(name)
        add("OPENAI_API_KEY")
        for profile in self.settings.models.values():
            if profile.api_key_env:
                add(profile.api_key_env)
        if self.settings.direct_model.api_key_env:
            add(self.settings.direct_model.api_key_env)
        for server in self.settings.mcp.values():
            for reference in server.headers.values():
                add(reference)
        return [self.environ[name] for name in names if self.environ.get(name)]

    async def prepare(self) -> None:
        request, settings, storage = self.request, self.settings, self.settings.storage
        self.check()
        self.bundle.state["request"] = request.model_dump(mode="json")
        self.bundle.state["controls"] = {
            "limits": settings.limits.model_dump(),
            "storage": storage.model_dump(),
            "execution_mode": "host",
            "network": "normal_host_access",
            "policy_digest": hashlib.sha256(
                json.dumps(settings.policy.model_dump(mode="json"), sort_keys=True).encode()
            ).hexdigest(),
            "allowed_executables": settings.policy.allowed_executables,
            "configuration_sources": settings.sources,
        }
        self.phase("discovering")
        catalog = discover(
            settings.skills_dir, max_instruction_bytes=storage.max_package_bytes, check=self.check
        )
        self.event(
            "catalog_prepared",
            {"valid_packages": len(catalog.skills), "rejected_packages": len(catalog.rejected)},
        )
        self.bundle.state["diagnostics"]["warnings"].extend(
            f"Excluded package: {item.message}" for item in catalog.rejected
        )
        for name in dict.fromkeys(request.required_skills):
            catalog.require(name)
        if not catalog.skills:
            raise RunnerError("no_matching_skill", "No valid installed skills are available.")
        if storage.max_read_bytes < 1 or storage.max_tool_output_bytes < 2:
            raise RunnerError("budget_exhausted", "Tool byte limits cannot fit a usable response.")
        self.files = FileTools(
            max_read_bytes=storage.max_read_bytes,
            max_tool_output_bytes=storage.max_tool_output_bytes,
            check=self.check,
        )
        root = self.bundle.root
        self.files.register_root(
            "scratch", root / "work/scratch", writable=True, max_bytes=storage.max_scratch_bytes
        )
        self.files.register_root(
            "artifacts", root / "artifacts", writable=True, max_bytes=storage.max_artifact_bytes
        )
        used_files = used_bytes = 0
        inventory: list[dict[str, Any]] = []
        for index, source in enumerate(request.inputs, 1):
            logical = f"input-{index}"
            snapshot = snapshot_tree(
                source,
                root / "work/inputs" / logical,
                max_files=storage.max_input_files - used_files,
                max_bytes=storage.max_input_bytes - used_bytes,
                check=self.check,
            )
            used_files += snapshot.file_count
            used_bytes += snapshot.total_bytes
            self.files.register_root(logical, snapshot.root)
            record = {
                "id": logical,
                "source": str(source),
                "snapshot": str(snapshot.root),
                "digest": snapshot.digest,
                "files": [asdict(item) for item in snapshot.files],
            }
            self.bundle.state["provenance"]["inputs"].append(record)
            inventory.append({"root": logical, "files": [asdict(item) for item in snapshot.files]})
        protected = [
            *request.inputs,
            settings.skills_dir,
            root / "work/inputs",
            root / "work/skills",
        ]
        self.registry = ArtifactRegistry(
            generated_roots=[root / "work/scratch", root / "artifacts"],
            staging_root=root / "work/staging",
            artifacts_root=root / "artifacts",
            protected_roots=protected,
            max_bytes=storage.max_artifact_bytes,
            check=self.check,
        )
        if request.output:
            if request.output_is_directory:
                try:
                    directory = request.output.stat()
                except OSError as exc:
                    raise RunnerError(
                        "publication_failed", "Cannot access output directory."
                    ) from exc
                if not stat.S_ISDIR(directory.st_mode):
                    raise RunnerError("publication_failed", "Output directory changed.")
                self.publication_directory = request.output
                self.publication_directory_identity = (directory.st_dev, directory.st_ino)
                self.publication_protected_roots = tuple(protected)
            else:
                self.publication = PublicationTarget.prepare(
                    request.output, overwrite=request.overwrite, protected_roots=protected
                )
            self.bundle.state["outputs"]["publication"] = {
                "state": "pending",
                "path": str(request.output),
            }
        self.activation = ActivationService(
            catalog,
            root / "work/skills",
            max_files=storage.max_package_files,
            max_bytes=storage.max_package_bytes,
            check=self.check,
        )
        self.context = RunContext(
            runner_instructions=INSTRUCTIONS
            + "\nRun capabilities and requirements:\n"
            + json.dumps(
                {
                    "required_skills": list(dict.fromkeys(request.required_skills)),
                    "format": request.format,
                    "allowed_executables": settings.policy.allowed_executables,
                    "command_env_references": settings.policy.command_env,
                    "generated_roots": {
                        "scratch": str(root / "work/scratch"),
                        "artifacts": str(root / "artifacts"),
                    },
                    "input_roots": {
                        item["id"]: item["snapshot"]
                        for item in self.bundle.state["provenance"]["inputs"]
                    },
                }
            ),
            prompt=request.prompt,
            catalog=catalog.metadata_for_model(),
            input_inventory=inventory,
        )
        self.tools = ToolRegistry(
            max_result_bytes=storage.max_tool_output_bytes,
            on_start=lambda call: self.event(
                "tool_started", {"call_id": call.id, "name": call.name, "executed": False}
            ),
            activation_is_new=lambda call: (
                not isinstance(call.arguments.get("name"), str)
                or call.arguments["name"] not in self.activation.active
            ),
        )
        self._bind_tools()
        self.profile = select_model(settings)
        key = resolve_api_key(self.profile, self.environ)
        adapter = self.adapter_factory(self.profile, key)
        self.resources.push_async_callback(adapter.aclose)
        capacities = await adapter.discover_capabilities(
            asyncio.get_running_loop().time() + self.deadline.remaining
        )
        if capacities.profile != self.profile:
            self.profile = capacities.profile
            await adapter.aclose()
            adapter = self.adapter_factory(self.profile, key)
            self.resources.push_async_callback(adapter.aclose)
        if self.profile.context_window_tokens is None or self.profile.max_output_tokens is None:
            raise RunnerError(
                "invalid_configuration", "Model capacities must be discovered or configured."
            )
        self.bundle.state["model"] = {
            "alias": (
                settings.selected_model or settings.default_model
                if settings.base_url is None
                else None
            ),
            "base_url": self.profile.base_url,
            "model": self.profile.model,
            "context_window_tokens": self.profile.context_window_tokens,
            "max_output_tokens": self.profile.max_output_tokens,
            "capacity_sources": capacities.sources,
            "credential_reference": self.profile.api_key_env,
            "auth_mode": self.profile.auth_mode,
        }
        if settings.mcp:
            await self._connect_mcp()
        self.phase("activating")
        for name in dict.fromkeys(request.required_skills):
            self._activate(name, "Explicitly required by the user.")
        self.bundle.state["lifecycle"]["status"] = "running"
        self.phase("executing")
        self.proposal = await AgentLoop(
            adapter=adapter,
            context=self.context,
            dispatcher=self.tools,
            ledger=self.ledger,
            deadline=self.deadline,
            context_window=self.profile.context_window_tokens,
            max_output=self.profile.max_output_tokens,
            model_transport_retries=self.settings.limits.model_transport_retries,
            on_event=self.event,
        ).run()
        self.answer = self.proposal["report"]
        self.bundle.state["diagnostics"]["assumptions"] = self.proposal["assumptions"]

    async def _connect_mcp(self) -> None:
        manager = MCPManager(
            self.settings.mcp,
            self.settings.policy,
            self.supervisor,
            self.environ,
            self.deadline,
            max_bytes=self.settings.storage.max_tool_output_bytes,
            cwd=self.bundle.root / "work/scratch",
        )

        # Register before connection so partial startup failures retain ownership.
        async def close() -> None:
            try:
                await manager.aclose()
            finally:
                self.bundle.state["diagnostics"].setdefault("mcp", []).extend(manager.diagnostics)

        self.resources.push_async_callback(close)
        await manager.connect()
        self.bundle.state["provenance"]["mcp_tools"] = dict(manager.reverse_map)

        def bind(model_name: str, server: str, remote_name: str) -> Callable[[BaseModel], Any]:
            async def invoke(args: BaseModel) -> Any:
                self._require_active()
                identity = {"server": server, "tool": remote_name, "model_name": model_name}
                self.event("mcp_request_started", identity)
                try:
                    result = await manager.invoke(model_name, args.model_dump())
                except (RunnerError, asyncio.CancelledError) as error:
                    unknown = (
                        error.details.get("outcome_unknown", False)
                        if isinstance(error, RunnerError)
                        else getattr(error, "outcome_unknown", False)
                    )
                    self.event(
                        "mcp_request_completed",
                        {
                            **identity,
                            "ok": False,
                            "outcome_unknown": unknown,
                            "code": error.code if isinstance(error, RunnerError) else "cancelled",
                        },
                    )
                    raise
                self.event(
                    "mcp_request_completed", {**identity, "ok": not result.get("isError", False)}
                )
                return result

            return invoke

        for tool in manager.tools:
            self.tools.register(
                tool.name,
                f"MCP server {tool.server}, tool {tool.original_name}. {tool.description}",
                schemas.MCPArgs,
                bind(tool.name, tool.server, tool.original_name),
                parameters_schema=tool.input_schema,
                terminal_errors=frozenset({"unsupported_capability", "missing_decision"}),
            )

    def _activate(self, name: str, reason: str) -> dict[str, Any]:
        def admit(instructions: str) -> None:
            trial = copy.deepcopy(self.context)
            trial.activate(name, instructions, str(self.bundle.root / "work/skills" / name))
            call = self.tools.current_call
            if call is not None:
                value = {
                    "name": name,
                    "instructions": instructions,
                    "package_root": str(self.bundle.root / "work/skills" / name),
                }
                trial.append_tool_result(
                    call.id,
                    ToolResult(call.id, call.name, True, True, value).as_dict(),
                )
            estimate = trial.estimate(self.tools.model_schemas())
            if (
                self.profile.context_window_tokens is None
                or estimate >= self.profile.context_window_tokens
            ):
                raise RunnerError(
                    "context_capacity_exceeded",
                    "Complete skill instructions do not fit model context.",
                )

        previous = name in self.activation.active
        try:
            descriptor = self.activation.activate(name, reason=reason, admit_instructions=admit)
        except RunnerError as error:
            error.details.setdefault("skill", name)
            error.details.setdefault("tool", "activate_skill")
            if error.code == "context_capacity_exceeded":
                error.details.setdefault(
                    "suggested_action",
                    (
                        f"Reduce the {name} skill instructions or configure a larger model "
                        "context, then rerun."
                    ),
                )
            else:
                error.details.setdefault(
                    "suggested_action", f"Inspect the {name} skill package and rerun."
                )
            raise
        assert descriptor.snapshot is not None
        self.context.activate(name, descriptor.instructions, str(descriptor.snapshot.root))
        if not previous:
            self.files.register_root(f"skill-{name}", descriptor.snapshot.root)
        self.bundle.state["provenance"]["activated_skills"] = self.activation.records.copy()
        self.event("skill_activated", {"name": name, "reason": reason, "reused": previous})
        self.bundle.save()
        result = {"name": name, "package_root": str(descriptor.snapshot.root)}
        if not previous:
            result["instructions"] = descriptor.instructions
        return result

    def _require_active(self) -> None:
        if not self.activation.active:
            raise RunnerError(
                "skill_not_activated", "Activate a relevant skill before doing its work."
            )

    def _bind_tools(self) -> None:
        for name, args_model, description in (
            (
                "list_files",
                schemas.ListFilesArgs,
                "List files within an approved root. Results are bounded; if truncated, "
                "continue with the returned next_offset to inspect remaining entries.",
            ),
            (
                "read_text",
                schemas.ReadTextArgs,
                "Read a UTF-8 text page and the full-file SHA-256. offset and length count "
                "bytes, not lines or characters. If truncated, continue from next_offset. "
                "Read all required pages before claiming to have read the complete file.",
            ),
            (
                "search_text",
                schemas.SearchTextArgs,
                "Search for literal text within an approved root, with an optional filename "
                "glob. Matches contain file paths, one-based line numbers and matching text. "
                "Use these locations to verify source citations. Results are bounded: "
                "truncated or skipped files/lines mean absence of a match is inconclusive.",
            ),
            (
                "read_media",
                schemas.ReadMediaArgs,
                "Request a media representation from an approved path. Returns an explicit "
                "unsupported-capability error when no compatible media adapter is available.",
            ),
            (
                "write_file",
                schemas.WriteFileArgs,
                "Write UTF-8 text in scratch or artifacts. Inputs and skill packages are "
                "read-only. Replacing a file requires overwrite=true and its current "
                "read_text SHA-256 as expected_sha256.",
            ),
            (
                "edit_file",
                schemas.EditFileArgs,
                "Replace a unique literal text match in a generated workspace file. Supply "
                "its current SHA-256; stale digests and ambiguous matches are rejected. "
                "Inputs and skill packages are read-only.",
            ),
        ):

            def bind(tool_name: str) -> Callable[[BaseModel], Any]:
                async def invoke(args: BaseModel) -> Any:
                    if tool_name in {"write_file", "edit_file"}:
                        self._require_active()
                    return getattr(self.files, tool_name)(**args.model_dump(exclude_none=True))

                return invoke

            self.tools.register(name, description, args_model, bind(name))

        async def activate(args: BaseModel) -> Any:
            values = args.model_dump()
            return self._activate(values["name"], values["reason"])

        async def register(args: BaseModel) -> Any:
            self._require_active()
            assert self.registry is not None
            values = args.model_dump()
            _, path, _ = self.files._resolve(values.pop("path"))
            record = self.registry.register(path, **values)
            self.record_artifact_registration(record)
            return asdict(record)

        async def finish(args: BaseModel) -> Any:
            return args.model_dump()

        async def command(args: BaseModel) -> Any:
            self._require_active()
            values = args.model_dump()
            executable = values["executable"]
            configured = self.settings.policy.command_env.get(executable, {})
            if any(
                configured.get(name) != reference for name, reference in values["env_refs"].items()
            ):
                raise RunnerError(
                    "command_not_allowed",
                    "env_refs must map each child variable name to its configured host "
                    "reference for this executable; omit env_refs to use its configured "
                    "environment.",
                )
            requested_cwd = values["cwd"]
            if requested_cwd is None:
                active = list(self.activation.active.values())
                if len(active) != 1:
                    raise RunnerError(
                        "invalid_arguments",
                        "Working directory is ambiguous across active skills.",
                        details={
                            "tool": "run_command",
                            "suggested_action": (
                                "Provide an explicit cwd inside a validated workspace root."
                            ),
                        },
                    )
                assert active[0].snapshot is not None
                cwd = active[0].snapshot.root
            else:
                _, cwd, _ = self.files._resolve(requested_cwd)
            environment = build_child_environment(self.environ, references=configured)
            remaining = self.deadline.remaining
            timeout = min(remaining, values["timeout"]) if values["timeout"] else remaining
            self.check()
            self.event(
                "command_started",
                {
                    "executable": executable,
                    "cwd": str(cwd),
                    "argument_count": len(values["argv"]),
                    "environment_references": configured,
                    "_content": {"argv": values["argv"]},
                },
            )
            result = await monitor_operation(
                lambda: self.supervisor.run(
                    executable,
                    values["argv"],
                    cwd=cwd,
                    environment=environment,
                    deadline=Deadline(timeout),
                ),
                self.monitor_storage,
            )
            return {
                **asdict(result),
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
            }

        self.tools.register(
            "activate_skill",
            "Load a complete installed skill before using it.",
            schemas.ActivateSkillArgs,
            activate,
            kind="activation",
        )
        self.tools.register(
            "register_artifact",
            "Register a generated candidate for validation.",
            schemas.RegisterArtifactArgs,
            register,
        )
        scratch_root = self.bundle.root / "work/scratch"
        artifact_root = self.bundle.root / "artifacts"
        command_schema = schemas.RunCommandArgs.model_json_schema()
        command_schema["properties"]["cwd"]["description"] = (
            f"Use the exact absolute output directory {artifact_root} or scratch directory "
            f"{scratch_root}. Omitted cwd means the active skill package. Do not insert work/ "
            "before artifacts."
        )
        command_schema["properties"]["argv"]["description"] = (
            f"Use OS paths. Write command outputs under {artifact_root} or {scratch_root}; "
            "file-tool paths such as artifacts/file.txt are not OS paths."
        )
        self.tools.register(
            "run_command",
            "Run an allowlisted executable. Use the exact generated-root paths in cwd and argv "
            "for output; omit env_refs to use the configured environment.",
            schemas.RunCommandArgs,
            command,
            parameters_schema=command_schema,
        )
        self.tools.register(
            "finish_run",
            "Propose the final outcome for runner validation.",
            schemas.FinishRunArgs,
            finish,
            kind="completion",
        )

    async def complete(self) -> None:
        self.check()
        assert self.registry is not None
        outcome = self.proposal["outcome"]
        missing = self.proposal["missing_requirements"]
        details: dict[str, Any] = {}
        if missing:
            details["missing_requirements"] = missing
            details["suggested_action"] = missing[0]
        if outcome != "succeeded":
            code = {
                "no_matching_skill": "no_matching_skill",
                "needs_input": "missing_decision",
                "blocked": "missing_dependency",
                "failed": "task_failed",
            }[outcome]
            raise RunnerError(
                code,
                self.answer or "The skill could not complete the task.",
                details=details,
            )
        if (
            not self.activation.active
            or set(self.request.required_skills) - self.activation.active.keys()
        ):
            raise RunnerError(
                "incomplete_run", "A successful run must activate its required skills."
            )
        if missing:
            raise RunnerError(
                "missing_dependency",
                "Completion still has unresolved required capabilities.",
                details=details,
            )
        self.phase("validating")
        primary = self.registry.select_primary()
        requested_id = self.proposal.get("primary_artifact_id")
        records = {record.id: record for record in self.registry.records}
        for identifier in self.proposal["secondary_ids"]:
            if identifier not in records:
                raise RunnerError("artifact_invalid", "Completion references an unknown artifact.")
        if requested_id:
            if requested_id not in records or (primary and primary.id != requested_id):
                raise RunnerError(
                    "artifact_invalid",
                    "Completion primary artifact does not match registered output.",
                )
            primary = records[requested_id]
        if primary is None:
            if self.request.format not in {"md", "txt", "json", "csv"}:
                raise RunnerError(
                    "artifact_invalid", "The requested format requires a generated artifact."
                )
            if not self.answer.strip():
                raise RunnerError(
                    "incomplete_run",
                    "A successful text-only run needs a nonblank answer.",
                    details={
                        "suggested_action": (
                            "Provide the requested answer or register a primary artifact "
                            "before finishing."
                        )
                    },
                )
            if self.request.format == "md" and self.request.output is None:
                self.bundle.state["outputs"]["primary_output"] = str(self.bundle.root / "result.md")
            else:
                path = (
                    self.bundle.root
                    / "work/scratch"
                    / f"answer-{uuid.uuid4().hex}.{self.request.format}"
                )
                self.files.write_file(str(path), self.answer)
                primary = self.registry.register(
                    path, format=self.request.format, role="primary", description="Primary answer"
                )
                self.record_artifact_registration(primary)
        for record in self.registry.records:
            self.check()
            frozen = self.registry.freeze(record, writers_stopped=True)
            self.monitor_storage()
            format_name = FORMAT_ALIASES.get(record.format.lower(), record.format.lower())
            if primary and record.id == primary.id and format_name != self.request.format:
                raise RunnerError(
                    "artifact_invalid",
                    "Primary artifact format conflicts with the requested format.",
                )
            validator_record: dict[str, Any] | None = None
            if format_name in MEDIA_TYPES:
                validation = validate_builtin(
                    frozen.path,
                    format_name,
                    size_limit=self.settings.storage.max_artifact_bytes,
                    archive_expanded_limit=self.settings.storage.max_archive_expanded_bytes,
                    check=self.check,
                )
            else:
                validator = self.settings.artifacts.validators.get(format_name)
                if validator is None:
                    raise RunnerError(
                        "unsupported_capability",
                        "Configure an allowlisted artifacts.validators parser for this format.",
                    )
                environment = build_child_environment(
                    self.environ,
                    references=self.settings.policy.command_env.get(validator.command, {}),
                )
                checked = await monitor_operation(
                    partial(
                        validate_external,
                        frozen.path,
                        validator,
                        supervisor=self.supervisor,
                        environment=environment,
                        deadline=self.deadline,
                        expected_digest=frozen.digest,
                        expected_size=frozen.size,
                        writers_stopped=True,
                    ),
                    self.monitor_storage,
                )
                validation = checked.validation
                validator_record = {
                    "command": checked.command,
                    "returncode": checked.process.returncode,
                    "stdout_bytes": checked.process.stdout_bytes,
                    "stderr_bytes": checked.process.stderr_bytes,
                    "stdout_truncated": checked.process.stdout_truncated,
                    "stderr_truncated": checked.process.stderr_truncated,
                }
                if self.bundle.events:
                    self.bundle.events.emit(
                        "artifact_external_validation",
                        validator_record,
                        content={
                            "stdout": checked.process.stdout.decode("utf-8", errors="replace"),
                            "stderr": checked.process.stderr.decode("utf-8", errors="replace"),
                        },
                    )
                if not validation.valid:
                    raise RunnerError(
                        "artifact_invalid",
                        "Configured validator rejected the generated artifact.",
                        details={
                            **validator_record,
                            "stdout": checked.process.stdout.decode("utf-8", errors="replace"),
                            "stderr": checked.process.stderr.decode("utf-8", errors="replace"),
                        },
                    )
            if not validation.valid:
                raise RunnerError(
                    "artifact_invalid", "Generated artifact failed format validation."
                )
            retained = self.registry.retain(frozen)
            item = {
                "id": record.id,
                "path": str(retained.path),
                "format": format_name,
                "role": record.role,
                "description": record.description,
                "status": "validated",
                "validation_level": validation.validation_level,
                "media_type": validation.media_type,
                "size": retained.size,
                "digest": retained.digest,
            }
            if validator_record is not None:
                item["validator"] = validator_record
            self.bundle.state["outputs"]["artifacts"].append(item)
            if primary and record.id == primary.id:
                self.bundle.state["outputs"]["primary_output"] = str(retained.path)
            self.event(
                "artifact_validated",
                {
                    "artifact_id": record.id,
                    "format": format_name,
                    "validation_level": validation.validation_level,
                    "size": retained.size,
                    "digest": retained.digest,
                },
            )
        if self.publication_directory is not None:
            if primary is None:
                raise RunnerError("artifact_invalid", "No primary artifact to publish.")
            filename = requested_output_filename(self.request.prompt) or (
                f"output-{primary.id}.{self.request.format}"
            )
            target = self.publication_directory / filename
            try:
                directory = self.publication_directory.stat()
                if (directory.st_dev, directory.st_ino) != self.publication_directory_identity:
                    raise RunnerError("publication_failed", "Output directory changed.")
                self.publication = PublicationTarget.prepare(
                    target,
                    overwrite=self.request.overwrite,
                    protected_roots=self.publication_protected_roots,
                )
                directory = self.publication_directory.stat()
                if (directory.st_dev, directory.st_ino) != self.publication_directory_identity:
                    raise RunnerError("publication_failed", "Output directory changed.")
            except OSError as exc:
                error = RunnerError("publication_failed", "Cannot access output directory.")
                self.bundle.state["outputs"]["publication"] = {
                    "state": "failed",
                    "path": str(target),
                    "error": error.code,
                }
                self.record_publication(error)
                raise error from exc
            except RunnerError as error:
                self.bundle.state["outputs"]["publication"] = {
                    "state": "failed",
                    "path": str(target),
                    "error": error.code,
                }
                self.record_publication(error)
                raise
            self.bundle.state["outputs"]["publication"] = {"state": "pending", "path": str(target)}
        if self.publication is not None:
            await asyncio.sleep(0)
            self.check()
            self.phase("publishing")
            selected = next(
                item
                for item in self.bundle.state["outputs"]["artifacts"]
                if primary and item["id"] == primary.id
            )
            try:
                published = self.publication.publish(
                    Path(selected["path"]),
                    expected_digest=selected["digest"],
                    expected_size=selected["size"],
                    max_bytes=self.settings.storage.max_artifact_bytes,
                    check=self.check,
                )
            except RunnerError as error:
                if error.details.get("committed"):
                    self.bundle.state["outputs"]["primary_output"] = str(self.publication.path)
                    self.bundle.state["outputs"]["publication"] = {
                        "state": "committed",
                        "path": str(self.publication.path),
                    }
                else:
                    self.bundle.state["outputs"]["publication"] = {
                        "state": "failed",
                        "path": str(self.publication.path),
                        "error": error.code,
                    }
                self.record_publication(error)
                raise
            self.bundle.state["outputs"]["primary_output"] = str(published.path)
            self.bundle.state["outputs"]["publication"] = {
                "state": "committed",
                "path": str(published.path),
                "digest": published.digest,
                "size": published.size,
                "directory_synced": published.directory_synced,
            }
            self.bundle.state["diagnostics"]["warnings"].extend(published.warnings)
            self.record_publication()

    def record_artifact_registration(self, record: ArtifactRecord) -> None:
        self.event(
            "artifact_registered",
            {"artifact_id": record.id, "format": record.format, "role": record.role},
        )

    def record_publication(self, error: RunnerError | None = None) -> None:
        publication = self.bundle.state["outputs"]["publication"]
        payload = dict(publication)
        if error is not None:
            payload["error"] = error.code
        try:
            self.event(f"publication_{publication['state']}", payload)
        except (RunnerError, OSError) as logging_error:
            message = "Could not persist the publication event."
            if error is not None:
                error.details.setdefault("reporting_errors", []).append(message)
            else:
                raise RunnerError(
                    "post_publication_reporting_failed",
                    message,
                    details={
                        "committed": True,
                        "path": publication["path"],
                        "suggested_action": "Inspect the preserved output and event-log storage.",
                    },
                ) from logging_error

    def preserve(self) -> None:
        if self.registry is None:
            return
        if self.supervisor.active:
            self.keep_work = True
            self.bundle.state["diagnostics"]["warnings"].append(
                "Owned writers remain; raw work preserved without freezing candidates."
            )
            return
        self.registry.check = lambda: None
        registered = {record.path for record in self.registry.records}
        retained_paths = {item.path for item in self.registry.retained}
        for root in self.registry.generated_roots:
            try:
                entries = scan_tree(root, max_bytes=self.settings.storage.max_scratch_bytes)
                for entry in entries:
                    if (
                        entry.directory
                        or entry.source in registered
                        or entry.source in retained_paths
                    ):
                        continue
                    try:
                        record = self.registry.register(
                            entry.source,
                            format=entry.source.suffix.lstrip(".") or "unknown",
                            role="secondary",
                            description="Unregistered incomplete output",
                        )
                    except RunnerError:
                        self.keep_work = True
                    else:
                        try:
                            self.record_artifact_registration(record)
                        except (RunnerError, OSError):
                            self.error(
                                RunnerError(
                                    "reporting_failed",
                                    "Could not record an incomplete artifact registration.",
                                    details={"artifact_id": record.id},
                                )
                            )
            except RunnerError:
                self.keep_work = True
        existing = {item["id"] for item in self.bundle.state["outputs"]["artifacts"]}
        for record in self.registry.records:
            if record.id in existing:
                continue
            try:
                frozen = self.registry.freeze(record, writers_stopped=True)
                retained = self.registry.retain(frozen, incomplete=True)
                self.bundle.state["outputs"]["artifacts"].append(
                    {
                        "id": record.id,
                        "path": str(retained.path),
                        "format": record.format,
                        "role": record.role,
                        "description": record.description,
                        "status": "incomplete",
                        "size": retained.size,
                        "digest": retained.digest,
                        "validation_level": "none",
                    }
                )
            except (RunnerError, OSError):
                self.keep_work = True
                self.bundle.state["diagnostics"]["warnings"].append(
                    "An incomplete candidate could not be retained; work directory preserved."
                )

    async def settle(self) -> None:
        task = asyncio.create_task(self.resources.aclose())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if not task.cancelled():
                    self.cancelled(error)
            except Exception:
                break
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            self.error(RunnerError("cleanup_failed", "An owned resource could not be closed."))
            self.keep_work = True

    async def run(self) -> dict[str, Any]:
        # Reject overlapping destinations before creating anything inside an input tree.
        validate_output_locations(
            self.request.inputs, self.settings.output_dir, self.request.output
        )
        self.bundle = RunBundle.create(
            self.settings.output_dir,
            invocation_directory=self.request.invocation_directory,
            secrets=self._secrets(),
            max_event_bytes=self.settings.storage.max_event_log_bytes,
            log_content=self.settings.diagnostics.log_content,
        )
        self.resources.push_async_callback(self.supervisor.aclose)
        try:
            await self.prepare()
        except asyncio.CancelledError as error:
            self.cancelled(error)
        except RunnerError as error:
            self.error(error)
        except Exception:
            self.error(
                RunnerError("execution_failed", "Task execution failed; inspect the run evidence.")
            )
        execution_elapsed = self.deadline.clock() - self.deadline.started_at
        await self.settle()
        if self.stop is None:
            self.resources.push_async_callback(self.supervisor.aclose)
            try:
                await self.complete()
            except asyncio.CancelledError as error:
                self.cancelled(error)
            except RunnerError as error:
                self.error(error)
            except Exception:
                self.error(RunnerError("finalization_failed", "Output finalization failed."))
        await self.settle()
        if self.stop is not None:
            self.preserve()
        if not self.keep_work:
            try:
                remove_work_tree(self.bundle.root / "work")
            except OSError:
                self.error(
                    RunnerError("cleanup_failed", "Could not remove the run work directory.")
                )
        pre_finalize_elapsed = self.deadline.clock() - self.deadline.started_at
        self.bundle.state["usage"] = {
            **self.ledger.summary(),
            "execution_elapsed_seconds": execution_elapsed,
            "post_execution_elapsed_seconds": pre_finalize_elapsed - execution_elapsed,
            "elapsed_seconds": pre_finalize_elapsed,
        }
        self.bundle.state["lifecycle"]["cleanup"] = {
            "work_retained": self.keep_work,
            "owned_pids_remaining": list(self.supervisor.active),
        }
        if self.keep_work:
            self.bundle.state["diagnostics"].setdefault("incomplete_work", []).append(
                f"Work directory retained at {self.bundle.root / 'work'}."
            )
        status, exit_code = (
            (self.stop.status, self.stop.exit_code) if self.stop else ("succeeded", 0)
        )
        return self.bundle.finalize(
            status=status,
            exit_code=exit_code,
            answer=self.answer,
            reconcile_outcome=self.terminal_outcome,
            elapsed_seconds=lambda: self.deadline.clock() - self.deadline.started_at,
        )


async def run_task(
    request: RunRequest,
    settings: ResolvedSettings,
    *,
    environ: Mapping[str, str],
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    factory = adapter_factory or (
        lambda profile, key: OpenAICompatibleAdapter(profile, api_key=key)
    )
    return await Coordinator(request, settings, environ, factory).run()
