# Usability remediation plan

## Goal and completion criteria
Address all twelve findings in the usability review with regression tests, without weakening host-execution safeguards or changing the user's untracked configuration, skills, or outputs. No new dependencies or live provider calls are required.

## Constraints and decisions
- Keep explicit credentials, executable allowlists, protected inputs, no implicit installation, and lossless history.
- Preserve legacy `--output` semantics; add unambiguous destination options rather than silently reinterpreting existing paths.
- Make `doctor` offline by default; `--network` explicitly opts into a resource-consuming tool-calling probe. Report independent checks even on failure.
- Skip implicit metadata discovery when capacities are configured; explicit discovery remains strict.
- Bound reads against remaining context without silently omitting content: retain pagination and disclose the bound. Conservative byte accounting remains the portable fallback, not a real tokenizer.
- Keep configuration precedence; warn when an ambient endpoint changes alias semantics, and provide explicit alias selection.
- Reject obsolete nonempty `allowed_env` with migration instructions; empty legacy lists remain accepted.
- Input exclusions are explicit and recorded; never automatically hide input files.
- Cleanup previews by default and requires confirmation; never remove active or unknown bundles.

## Milestones
1. [x] Safe field-level config diagnostics, complete receipts, JSON parser failures, version/config inspection.
2. [x] Offline doctor/preflight, network tool probe, optional metadata, early validator checks.
3. [x] Context-aware reads, visible progress, clean primary answers, explicit output destinations.
4. [x] Local/authenticated initializer, empty catalog guidance, alias warnings, environment-policy migration.
5. [x] Explicit input exclusions/preview, proxy/CA settings, run listing/usage/safe cleanup.
6. [x] Documentation and example alignment; full tests, Ruff, mypy, final diff review (offline-cache limitation recorded below).

## Verification
Write failing regression tests before each implementation slice. Use temporary workspaces, fake adapters and HTTP mock transports. Run focused tests after each milestone, then unit/integration suites and static checks. Record infrastructure failures separately from product failures. Existing baseline: unit+CLI 569 passed/8 skipped; broader integration 245 passed/25 skipped with one missing offline wheel-cache dependency and one transient MCP timeout (six-test suite passed on rerun).

## Edge cases
Missing/invalid config, unsafe values in errors, CLI parse errors with JSON, empty catalogs, bad model aliases, ambient endpoint mode changes, offline checks without credentials, truncated/invalid tool probes, validators without executable permission, read pagination near capacity, batch reads, missing output directories, existing initializer files, symlinks and active runs during cleanup, exclusions at snapshot recheck, unavailable proxy/CA paths.

## Progress
Plan recorded before implementation. User-owned untracked `skillrun.toml`, `skills/`, and `summary.md` remain untouched.

- Milestone regressions were first observed failing, then made green (diagnostics/setup, preflight, execution feedback, operational tooling).
- Focused run: 702 passed / 8 skipped before the last boundary tests and artifact-extension improvement.
- First full run: 839 passed / 33 skipped / 3 failures. Updated two old assertions expecting late PDF failure and an ignored allowed_env setting to the new intentional contracts.
- Final full run: **851 passed / 33 skipped / 1 failed** using `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short`. The only failure is the pre-existing offline dependency-cache problem in `test_offline_wheel_installation`: locked wheels are absent (ruamel-yaml/jsonschema reported on different runs). Wheel build and wheel-only installation succeed; the subsequent offline dependency installation cannot complete. No network/package changes were made to conceal this limitation.
- `ruff check .`, `ruff format --check .`, strict `mypy` (68 source files), and `git diff --check` pass. Final diff review covered configuration/redaction, context accounting, file destinations, retention and cleanup.
- Added dedicated regression coverage in six test_usability modules, plus updated existing tests for intentional contract changes. Tests use temporary workspaces and mock transports; native non-Windows and real-provider behavior remain unverified in this session.
- Capacity accounting intentionally remains conservative and history remains lossless. Context-aware pagination prevents a single read from consuming response capacity, not unlimited-document processing.
- Historical benchmark claims were moved to docs/qualification-history.md and are explicitly not new real-model qualification.

## Release preparation follow-up
- Usability commit `fb83a1a` passed all six native CI combinations (Linux/macOS/Windows, Python 3.13/3.14): https://github.com/skanga/skillrunner/actions/runs/37159919217.
- During v0.2.0 preparation, isolated source/wheel installs passed. The offline-install test requires the exact URL-based pylock cache, not only the registry cache populated by normal tool installation. Populating that cache from the unchanged, hash-verified lock in a temporary environment made both installation tests pass. No test was skipped or weakened to work around missing dependencies.
- Release notes and final candidate validation are published with the GitHub release; no new paid model qualification is claimed.
