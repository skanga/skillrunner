# Skill Runner

Skill Runner runs one task using local Agent Skills packages, writes the results and an execution report, then exits. It selects relevant skills from their descriptions. `--skill` requires named skills while still allowing additional skills when needed.

The implementation is under release qualification. The CI matrix targets Linux, macOS, and Windows with Python 3.13 and 3.14. A configured matrix is not evidence that all native platform checks or real-model quality gates have passed.

## Install and inspect

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13 or later. From this checkout:

```console
uv sync --locked --dev
uv run skillrun --help
uv run skillrun skills list
uv run skillrun skills validate ./skills
```

For a standalone command, install from source or build and install a wheel:

```console
uv tool install .
uv build
uv tool install ./dist/skillrunner-0.1.0-py3-none-any.whl
skillrun --help
```

The two `uv tool install` forms are alternatives. When replacing an existing installation, use `--reinstall`. Run `uv tool update-shell` if the tool's bin directory is not on your PATH. Use `uv sync --locked` for development environments that must reproduce `uv.lock`; tool installation resolves dependencies separately.

Place standard packages under a catalog directory:

```text
skills/
  summarize/
    SKILL.md
    references/
    scripts/
    assets/
```

A minimal `skills/summarize/SKILL.md`:

```markdown
---
name: summarize
description: Summarize supplied notes into a concise Markdown document.
---
Read the supplied notes. Write a summary that preserves the important facts.
```

Packages do not need a runner-specific manifest. Discovery stops inside a package, so nested resources are not separate skills. Invalid packages are listed and excluded. Duplicate declared names are a catalog error. Inspection never runs package scripts or calls the model.

## Configure and run

Copy [examples/skillrun.toml](examples/skillrun.toml) to `skillrun.toml` in your working directory. Set the endpoint, literal model ID, authentication mode, and documented model capacities. The sample uses placeholder capacities and a local endpoint; edit them for your model.

```console
skillrun run "Summarize these notes" -i notes.txt -o summary.md
skillrun run "Create a report" -s summarize -i notes.txt -j
skillrun run -p task.txt -i notes.txt
skillrun run --prompt-stdin -i notes.txt
skillrun run "Summarize notes" -i notes.txt -b http://localhost:8000/v1 -m your-model-id
```

`--prompt-stdin` reads until EOF. Supply exactly one nonempty prompt source: positional text, `-p/--prompt-file`, or `--prompt-stdin`. Task execution always requires `run`. All task flags follow it. For example, `skillrun run "doctor"` is a task with that prompt.

Use `skillrun doctor` to check configuration, credential availability, model connectivity, and installed runtime paths. **Every doctor invocation contacts the configured model and may consume provider resources.** It attempts discovery and a small completion request. Missing configuration or credentials prevent the check and produce a failure. Runtime presence checks do not launch binaries. No runtime is universally required except the coordinator's Python; skills determine their own dependencies.

## Options

Short aliases are case-sensitive. Repeated input and skill options accumulate in order. Repeated scalar options use the last value, including mixed short and long forms. Unknown options and abbreviations are rejected.

| Option | Purpose |
|---|---|
| `-p, --prompt-file` | Read UTF-8 task text |
| `--prompt-stdin` | Read task text through EOF |
| `-S, --skills-dir` | Catalog root; default `./skills` |
| `-s, --skill` | Required skill name; repeatable |
| `-i, --input` | Input file or directory; repeatable |
| `-o, --output` | Publish the primary deliverable at a file path or inside an existing directory |
| `-O, --output-dir` | Bundle parent; default `./outputs` |
| `-f, --format` | Primary output format; otherwise inferred, then Markdown |
| `-m, --model` | Configured alias, or literal ID with a base URL |
| `-b, --base-url` | OpenAI-compatible endpoint |
| `-c, --config` | Select one TOML file instead of `./skillrun.toml` |
| `--policy` | Replace the entire embedded policy with a TOML policy file |
| `-t, --timeout` | Execution deadline; default `10m` |
| `--max-steps` | Model-turn limit; default `40` |
| `--max-tool-calls` | Aggregate tool-call limit; default `100` |
| `--max-tokens` | Aggregate input/output token budget; default `100000` |
| `--model-transport-retries` | Shared retries for transient model transport failures and empty terminal completions; default `1` |
| `--shutdown-grace` | Child shutdown grace; default `5s` |
| `--overwrite` | Allow replacing the explicitly requested primary output |
| `-j, --json` | One terminal JSON receipt on stdout |
| `-q, --quiet` | Suppress progress; retain errors and receipt |
| `-h, --help` | Usage and examples without starting a run |

Durations require `ms`, `s`, `m`, or `h`, for example `500ms`, `1.5m`, or `1h`. TOML durations must be strings. Only shutdown grace permits zero (`0s`). Integer limits must be positive. Cleanup and final reporting can extend beyond the execution deadline.

An empty assistant completion (`finish_reason="stop"`, blank or null content, and no tool calls) is recorded as a protocol error and may use the same retry allowance as HTTP 429/5xx and connection failures. Each attempt is separately charged within existing budgets; zero retries disables recovery. Other protocol errors and output-length stops are not retried, and dispatched tools are never replayed automatically.

Storage limits are configured in `[storage]`. Defaults permit 10,000 input files totaling 1 GiB, 20,000 activated-package files totaling 512 MiB, 2 GiB of artifacts, and 4 GiB of scratch data. Tool output is bounded at 1 MiB, event logs at 10 MiB, individual reads at 64 KiB, and expanded archives at 256 MiB. Exceeding a limit produces an explicit failure; required input and instructions are not silently omitted. MCP tool results share the tool-output and execution budgets.

## Configuration and model capabilities

Precedence is **explicit CLI > documented environment variables > selected TOML file > defaults**. Explicit `--config` replaces the default file. Parent directories, user-wide configuration, and `.env` files are not loaded. Missing optional `./skillrun.toml` is allowed; a missing explicit file is an error.

CLI and environment paths resolve against the invocation directory. Paths written in configuration or policy files resolve against that file's directory. A policy file contains policy keys directly, such as `allowed_executables = []`, without an enclosing `[policy]` table.

| Environment variable | Setting |
|---|---|
| `SKILLRUN_SKILLS_DIR` | Catalog root |
| `SKILLRUN_OUTPUT_DIR` | Bundle parent |
| `SKILLRUN_TIMEOUT` | Execution deadline |
| `SKILLRUN_SHUTDOWN_GRACE` | Shutdown grace |
| `SKILLRUN_MAX_STEPS` | Model-turn limit |
| `SKILLRUN_MAX_TOOL_CALLS` | Tool-call limit |
| `SKILLRUN_MAX_TOKENS` | Token budget |
| `OPENAI_BASE_URL` | Direct endpoint; requires literal `--model` |
| `OPENAI_API_KEY` | Default bearer credential reference |

Without a base URL override, `--model` chooses `[models.<alias>]`; otherwise `default_model` chooses the alias. With `--base-url` or `OPENAI_BASE_URL`, `--model` is a literal model ID and `[direct_model]` supplies authentication and capacity settings. It does not borrow a named profile's credentials. No model name is hardcoded or allowlisted.

The adapter uses Chat Completions and preserves the endpoint path prefix. It does not add `/v1`. Use `auth_mode = "none"` for unauthenticated endpoints; these requests omit the Authorization header. For bearer authentication, `api_key_env` names an environment variable. Store credentials there, never in TOML or task text.

Provide `context_window_tokens` and `max_output_tokens` from the model's documented limits, or configure `[models.<alias>.discovery]` with an endpoint-relative `path` and dotted `context_window_field` / `max_output_field` mappings. The adapter probes model metadata but does not assume every service exposes capacity fields. Missing capacities block execution rather than guessing. Configured values take precedence over discovered values. Set `output_token_parameter` to `max_tokens` or `max_completion_tokens` as required by the endpoint.

[examples/skillrun.toml](examples/skillrun.toml) shows all supported top-level sections: model profiles, direct-model settings, limits, storage, policy, MCP, artifact validators, and diagnostics. Optional inference options belong in `[models.<alias>.request_options]` or `[direct_model.request_options]`. PNG inspection is opt-in: configure `input_modalities = ["text", "image"]`, `image_accounting = "openai-patch-high-v1"`, and an allowlisted `[artifacts.validators.png]` parser. Then `read_media` with `representation = "image"` validates a copy and sends the PNG at high detail after the tool batch. The original input is unchanged. Other formats, missing capabilities, and missing validators return `unsupported_capability`.

The `openai-patch-high-v1` accounting contract uses 32-pixel patches, a 2048-pixel dimension limit, a 2500-patch budget, and a 1.2 multiplier, plus one token for rounding. Select it only for an endpoint implementing those rules; model names do not select it automatically. Image tokens are estimated separately and charged on every request retaining the image. Context, aggregate-token, and tool-output byte limits still apply. Oversized images fail rather than being truncated. Logs retain image metadata and digests, never the image payload, even with `--log-content`. Audio, video, remote-image fetching, and automatic PDF conversion are unsupported; provisioned tools can render PDF pages to PNG before inspection.

For Gemma 4 deployments using the standard single-image processor, explicitly select `image_accounting = "gemma4-image-max-v1"`. This reserves 1,122 tokens per image per request: up to 1,120 visual tokens plus two image boundary tokens. Select this contract only when the deployment uses those limits; custom cropping or expansion needs a matching contract. It preserves the same PNG validation, high-detail transport, byte limits, and metadata-only logs. Configure context and output limits for your deployment separately.

The conversation and skill instructions must fit the context window. The runner does not silently truncate instructions, summarize history, or discard earlier turns to continue. Task requests require model tool calling; incompatible responses produce a failure instead of fabricated execution.

## Files, host execution, and connectors

Inputs and activated packages are snapshotted and read-only through runner-managed file tools. Paths mentioned only in a prompt do not grant file access; pass files with `--input`. Put output locations outside input trees. In particular, `--input .` conflicts with default `./outputs`; choose a separate `--output-dir` and, if supplied, `--output`.

With no `-i/--input`, the run receives no input snapshot. With no `-o/--output`, the primary deliverable stays in its run bundle under `./outputs` by default.

**Host scripts are not sandboxed.** They run with the host user's filesystem and network access. Runner-managed path checks and executable allowlists govern dispatch but cannot contain a script after launch. Run only trusted packages. Container isolation is deferred.

`[policy] allowed_executables = []` permits no commands. Add installed executables that your skills need. An installed binary still cannot run until allowed. Executable identities are resolved and checked before launch. Python, Node.js, and a POSIX shell are independent capabilities; Windows does not require a shell unless a selected skill does. The runner does not install dependencies automatically.

Child processes receive a minimal environment. `[policy.command_env.<executable>]` explicitly maps child variable names to environment-variable references. Do not depend on arbitrary inherited secrets. Process groups on POSIX and Job Objects on Windows manage child cleanup; they do not provide containment.

Model Context Protocol (MCP) servers must be configured under `[mcp.<name>]`. The sample configuration contains commented stdio and Streamable HTTP examples. Each server needs an explicit `allowed_tools` list. Stdio commands also need an executable allowlist entry. `env` values and HTTP `headers` values name credential environment variables; an Authorization reference contains the complete header value. No connector is discovered from a skill name or arbitrary URL. External writes with an uncertain outcome must not be blindly retried.

Prompts, loaded skill instructions, input content read into context, and tool results can be sent to the selected model. Configured MCP servers receive their tool calls and arguments. Host scripts can make their own network requests. There is no product telemetry by default. Content logging is opt-in through `[diagnostics] log_content`; retained files and logs may contain task data.

## Results and exit codes

Each accepted execution creates a unique persistent bundle beneath the output directory. `result.md` is the report, `run.json` is the manifest, and retained artifacts are stored alongside them. The work directory is removed after cleanup unless diagnostic retention is enabled or incomplete work must be preserved. The JSON receipt identifies the run, status, exit code, primary output, report, manifest, artifacts, and errors. Tool chatter and diagnostics go to stderr.

`--output FILE` publishes the primary deliverable at that exact path while retaining the bundle. `--output DIRECTORY` publishes inside an existing directory using a unique generated filename with the validated format's extension. If the prompt explicitly names an output file, its basename is used inside that directory even when the model registers the artifact under another name. A path that does not yet exist is treated as a file path. Existing destination files remain unchanged unless `--overwrite` is supplied. Publication failure is nonzero and points to recoverable artifacts. If publication succeeds but final reporting fails, the published file remains. A report alone does not satisfy a requested binary deliverable.

| Exit | Status | Meaning |
|---|---|---|
| `0` | `succeeded` | Task and required publication completed |
| `2` | `invalid_request` | Arguments, configuration, catalog, or explicit skill error |
| `3` | `no_matching_skill` | No installed skill fits |
| `4` | `blocked` | Missing capability, dependency, permission, or credential |
| `5` | `needs_input` | A decision or required input is missing |
| `6` | `failed` | Model, execution, validation, or publication failure |
| `7` | `limit_exceeded` | Deadline or execution budget exhausted |
| `130` | `cancelled` | SIGINT |
| `143` | `cancelled` | SIGTERM |

The runner never pauses for follow-up input. Resolve a `needs_input` result and start a new invocation. Partial outputs do not imply success. Early argument errors may have no bundle; an uncatchable kill may leave a bundle without terminal status.

Bundles remain local until you delete them. After reviewing or archiving a completed run, delete its individual bundle directory using your normal file manager or removal command. Confirm that the process has exited before deleting its files. Automatic retention and crash resumption are not implemented.

## Development and verification

```console
uv sync --locked --dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv build
```

The installation tests build and install a wheel offline, reuse existing development dependencies, and invoke the installed command outside the repository. Run `uv sync` first to populate the build cache. Tests use mocked model transports or local fixtures; they do not qualify real-model quality. The [CI workflow](.github/workflows/ci.yml) schedules all six OS/Python combinations using the [official uv setup integration](https://docs.astral.sh/uv/guides/integration/github/). Native results and real-endpoint qualification must be reviewed separately before claiming release conformance.

If a model response reaches its output-token limit (`finish_reason="length"`),
the runner accounts for usage and exits with code 7 (`limit_exceeded`). It preserves
partial artifacts and the failure report, leaves an unpublished destination unchanged,
and neither dispatches returned tools nor retries that response. When the provider
omits usage, returned public text and tool-call content (including truncated
arguments) are charged using the documented UTF-8 estimate.


### Optional image inspector and task acceptance checks

A text-only executor can use a separate named image-capable profile through
`image_inspector = "vision"` at the top level of the configuration. Define
`[models.vision]` with the endpoint, model, credential environment reference,
capacities, `input_modalities = ["text", "image"]`, and an explicitly supported
`image_accounting` contract. The existing allowlisted PNG validator is still
required. The `inspect_image` tool sends a validated PNG and a question to that
profile, returning observations and image metadata to the executor. It cannot
execute task commands or finish the parent run. Both models share the run's
turn, token, tool-call and time limits; image bytes are omitted from logs.
The executor receives the configured `storage.max_tool_output_bytes` PNG limit.
It can use allowed commands to prepare a smaller preview or crop for inspection
while preserving the original deliverable. The runner does not resize images;
an oversized inspection request still stops with exit 7.

Optional task acceptance checks reject a success proposal when the configured
command returns nonzero. For example, on a host with this checker installed:

```toml
[acceptance]
max_repairs = 2

[acceptance.checks.deliverable]
command = "/opt/checkers/check-deliverable"
args = ["{path}"]
```

The checker command must also be in `policy.allowed_executables`. Configure its
environment through the existing `policy.command_env` references. `{path}` is
replaced with the path of an isolated copy of the primary candidate, or the
proposed text report when there is no primary artifact. With acceptance enabled,
a checked text report is retained as a primary artifact; `result.md` remains the
separate diagnostic report. Arguments are passed
directly, without shell evaluation. Checkers must inspect actual output and
return useful bounded diagnostics; a model-written test summary is not evidence
that tests ran. Provision checkers outside model-writable workspace roots.

A rejected proposal returns findings to the model for correction. With acceptance
checks configured, register one primary artifact and repair that file in place,
reusing its artifact ID. A second primary registration is rejected before changing
the registry; additional outputs can be registered as secondary. The default
allows two repairs; another rejection ends the run with exit 6. These repairs
use the existing run budgets and are distinct from transport retries. Passing
checks bind to the final candidate digest: changing the primary bytes after
acceptance prevents publication. Format validation remains required. Checks
prove only the requirements they actually test, not unrestricted semantic quality.
Without acceptance checks or an image inspector, existing behavior is preserved.
