"""Observe automatic skill selection without dispatching task actions."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import SecretStr

from skillrunner.catalog.discovery import discover
from skillrunner.config.models import ModelProfile
from skillrunner.config.sources import resolve_settings, select_model
from skillrunner.domain.request import RunRequest
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.model.protocol import ModelAdapter, ModelReply, ModelToolCall
from skillrunner.qualification.budget import PaidAdapter, SpendingLedger
from skillrunner.qualification.corpus import _verify
from skillrunner.qualification.run import (
    apply_request_context_cap,
    corpus_cases,
    qualification_rates,
)
from skillrunner.recording.bundle import atomic_write
from skillrunner.runtime.signals import run_with_signals


class QualificationAdapter(ModelAdapter, Protocol):
    async def discover_capabilities(self, deadline: float) -> Any: ...

    async def aclose(self) -> None: ...


class SelectionAdapter:
    """Let activation finish, then replace the first task action with a safe stop."""

    def __init__(self, adapter: QualificationAdapter) -> None:
        self.adapter = adapter
        self.decision_reached = False
        self.intercepted_tool_names: list[str] = []

    async def discover_capabilities(self, deadline: float) -> Any:
        return await self.adapter.discover_capabilities(deadline)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        output_limit: int,
        request_deadline: float,
    ) -> ModelReply:
        reply = await self.adapter.complete(messages, tool_schemas, output_limit, request_deadline)
        names = [call.name for call in reply.tool_calls]
        if any(name not in {"activate_skill", "finish_run"} for name in names):
            self.intercepted_tool_names = names
            self.decision_reached = True
            arguments = {
                "outcome": "blocked",
                "report": "Selection observation stopped before task tools executed.",
                "missing_requirements": [
                    "Run the complete task separately for task-success grading."
                ],
            }
            return ModelReply(
                None,
                (
                    ModelToolCall(
                        "selection-observation-stop",
                        "finish_run",
                        arguments,
                        json.dumps(arguments),
                    ),
                ),
                "tool_calls",
                reply.usage,
                reply.provider_request_id,
            )
        if names == ["finish_run"]:
            self.decision_reached = True
        return reply

    async def aclose(self) -> None:
        await self.adapter.aclose()


def score_selection(
    active: list[str], expected: list[str], *, decision_reached: bool, status: str
) -> dict[str, Any]:
    false_activations = sorted(set(active) - set(expected))
    missed_skills = sorted(set(expected) - set(active))
    return {
        "correct": bool(
            decision_reached
            and not false_activations
            and not missed_skills
            and (expected or status == "no_matching_skill")
        ),
        "false_activations": false_activations,
        "missed_skills": missed_skills,
    }


def summarize_selection(cases: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [case["id"] for case in cases]
    actual_ids = [run["case"] for run in runs]
    complete = (
        len(expected_ids) == len(set(expected_ids))
        and len(actual_ids) == len(expected_ids)
        and set(actual_ids) == set(expected_ids)
        and all("score" in run and "evidence_error" not in run for run in runs)
    )
    correct = sum(run.get("score", {}).get("correct") is True for run in runs)
    false_activations = sum(len(run.get("score", {}).get("false_activations", [])) for run in runs)
    total = len(cases)
    return {
        "correct": correct,
        "total": total,
        "false_activations": false_activations,
        "population_complete": complete,
        "selection_gate_passed": bool(complete and total and correct * 100 >= 95 * total),
    }


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def freeze_selection_population(corpus_path: Path, smoke_path: Path) -> list[dict[str, Any]]:
    corpus = json.loads(corpus_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    population = [
        {
            "id": case["id"],
            "source": "corpus.json",
            "expected_skills": case["selection"]["automatic_expected_skills"],
            "prompt_sha256": _prompt_hash(case["prompt"]),
        }
        for case in corpus["cases"]
    ]
    no_match = [case for case in smoke["cases"] if case["id"] == "no-match"]
    if len(no_match) != 1 or no_match[0]["expected_skills"] != []:
        raise ValueError("Selection set needs the declared no-match case")
    population.append(
        {
            "id": "smoke/no-match",
            "source": "smoke-cases.json",
            "expected_skills": [],
            "expected_status": "no_matching_skill",
            "prompt_sha256": _prompt_hash(no_match[0]["prompt"]),
        }
    )
    if len({item["id"] for item in population}) != len(population):
        raise ValueError("Selection set has duplicate case IDs")
    return population


def verify_selection_population(
    frozen: list[dict[str, Any]], corpus_path: Path, smoke_path: Path
) -> None:
    if frozen != freeze_selection_population(corpus_path, smoke_path):
        raise ValueError("Frozen selection population drifted from its source manifests")


def load_selection_cases(
    selection_path: Path,
    corpus_path: Path,
    smoke_path: Path,
    prepared_path: Path,
    preflight_path: Path,
) -> list[dict[str, Any]]:
    """Load frozen labels and verify prepared inputs, ignoring task-only readiness."""
    selection = json.loads(selection_path.read_text())
    for key, path in (
        ("corpus_sha256", corpus_path),
        ("smoke_cases_sha256", smoke_path),
        ("prepared_inputs_sha256", prepared_path),
    ):
        if selection.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Selection source or prepared-input drift")
    verify_selection_population(selection["population"], corpus_path, smoke_path)
    corpus = json.loads(corpus_path.read_text())
    prepared = json.loads(prepared_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    mapped = {item["id"]: item for item in corpus_cases(corpus, prepared, preflight, "automatic")}
    no_match = next(item for item in smoke["cases"] if item["id"] == "no-match")
    cases = []
    for label in selection["population"]:
        if label["source"] == "corpus.json":
            source = mapped[label["id"]]
            cases.append(
                {
                    "id": label["id"],
                    "prompt": source["prompt"],
                    "inputs": source["inputs"],
                    "format": source["expected"]["primary_format"],
                    "expected_skills": label["expected_skills"],
                }
            )
        else:
            cases.append(
                {
                    "id": label["id"],
                    "prompt": no_match["prompt"],
                    "inputs": [],
                    "format": "md",
                    "expected_skills": [],
                }
            )
    return cases


async def run_selection_case(
    case: dict[str, Any],
    settings: Any,
    *,
    environ: Mapping[str, str],
    adapter_factory: Callable[[ModelProfile, SecretStr | None], QualificationAdapter],
) -> dict[str, Any]:
    """Use the real coordinator through selection, stopping before task tools."""
    adapters: list[SelectionAdapter] = []

    def factory(profile: ModelProfile, key: SecretStr | None) -> SelectionAdapter:
        adapter = SelectionAdapter(adapter_factory(profile, key))
        adapters.append(adapter)
        return adapter

    request = RunRequest(
        prompt=case["prompt"],
        invocation_directory=Path(case.get("invocation_directory", Path.cwd())),
        inputs=[Path(value) for value in case.get("inputs", [])],
        format=case.get("format", "md"),
    )
    receipt = await run_with_signals(request, settings, environ=environ, adapter_factory=factory)
    activated: list[str] = []
    evidence_error = None
    try:
        manifest = json.loads(Path(receipt["manifest_path"]).read_text())
        activated = [item["name"] for item in manifest["provenance"]["activated_skills"]]
    except (OSError, ValueError, KeyError, TypeError):
        evidence_error = "Could not read activated skills from the returned manifest."
    adapter = adapters[-1] if adapters else None
    decision_reached = bool(adapter and adapter.decision_reached and evidence_error is None)
    result = {
        "case": case["id"],
        "receipt": receipt,
        "activated_skills": activated,
        "decision_reached": decision_reached,
        "intercepted_tool_names": adapter.intercepted_tool_names if adapter else [],
        "score": score_selection(
            activated,
            case["expected_skills"],
            decision_reached=decision_reached,
            status=receipt["status"],
        ),
    }
    if evidence_error:
        result["evidence_error"] = evidence_error
    return result


def verify_selection_catalog(skills_dir: Path, corpus_path: Path, preparation_path: Path) -> int:
    """Refuse a changed or augmented catalog before sending any model request."""
    corpus = json.loads(corpus_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    if preparation["corpus_version"] != corpus["corpus_version"]:
        raise ValueError("Prepared catalog belongs to another corpus version")
    packages = {item["id"]: item for item in corpus["packages"]}
    expected_names = set()
    for item in preparation["packages"]:
        manifest = packages[item["id"]]
        if item["package_sha256"] != manifest["package_sha256"]:
            raise ValueError("Prepared package digest drift")
        name = Path(item["path"]).name
        expected_names.add(name)
        _verify(skills_dir / name, manifest["files"])
    catalog = discover(skills_dir)
    if catalog.rejected or set(catalog.skills) != expected_names:
        raise ValueError("Selection catalog drift or rejected package")
    return len(catalog.skills)


def validate_selection_set(selection: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """Accept refrozen revisions while retaining the complete labeled population."""
    if (
        re.fullmatch(
            r"full-catalog-automatic-selection-v[1-9][0-9]*",
            str(selection.get("set_id", "")),
        )
        is None
        or len(cases) != selection.get("total_cases")
        or selection.get("positive_cases") != 21
        or selection.get("no_match_cases") != 1
    ):
        raise ValueError("The frozen 22-case selection population is incomplete")


async def evaluate_selection(args: argparse.Namespace) -> dict[str, Any]:
    """Run one selection observation per frozen label under the shared ledger."""
    cases = load_selection_cases(
        args.selection_set,
        args.corpus,
        args.smoke_cases,
        args.prepared_inputs,
        args.preflight,
    )
    selection = json.loads(args.selection_set.read_text())
    validate_selection_set(selection, cases)
    selected_ids = set(args.case or [case["id"] for case in cases])
    if not selected_ids or selected_ids - {case["id"] for case in cases}:
        raise ValueError("Select only known frozen selection case IDs")
    package_count = verify_selection_catalog(args.skills_dir, args.corpus, args.preparation)
    if package_count != 20:
        raise ValueError("Selection requires the complete unchanged 20-package catalog")
    settings = resolve_settings(
        Path.cwd(),
        {
            "config": args.config.resolve(),
            "skills_dir": args.skills_dir.resolve(),
            "output_dir": args.output_dir.resolve(),
            "model": args.model,
            "max_steps": args.max_steps,
            "max_tool_calls": args.max_tool_calls,
            "max_tokens": args.max_tokens,
            "timeout": "3m",
        },
        {},
    )
    settings.diagnostics.log_content = args.log_content
    profile = select_model(settings)
    if args.max_output < 1:
        raise ValueError("Qualification output cap must be positive")
    if profile.max_output_tokens is not None:
        profile.max_output_tokens = min(profile.max_output_tokens, args.max_output)
    rates, cost_basis = qualification_rates(profile)
    published_context, request_context_cap = apply_request_context_cap(
        settings, profile, rates, getattr(args, "request_context_cap", 128000)
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "method": (
            "Actual full-catalog coordinator selection; first task-tool batch "
            "intercepted before dispatch"
        ),
        "selection_set_sha256": hashlib.sha256(args.selection_set.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "prepared_inputs_sha256": hashlib.sha256(args.prepared_inputs.read_bytes()).hexdigest(),
        "preparation_sha256": hashlib.sha256(args.preparation.read_bytes()).hexdigest(),
        "model": profile.model,
        "endpoint": profile.base_url,
        "cost_basis": cost_basis,
        "published_context_window_tokens": published_context,
        "request_context_cap_tokens": request_context_cap,
        "package_count": package_count,
        "selected_case_ids": sorted(selected_ids),
        "runs": [],
        "summary": summarize_selection(cases, []),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8") as writer:
        json.dump(result, writer, indent=2)
    with SpendingLedger(args.ledger) as ledger:
        ledger.require_authorized_ceiling()

        def factory(selected: ModelProfile, key: SecretStr | None) -> QualificationAdapter:
            return PaidAdapter(
                OpenAICompatibleAdapter(selected, api_key=key),
                selected,
                ledger,
                rates,
                max_output=args.max_output,
            )

        for case in cases:
            if case["id"] not in selected_ids:
                continue
            observed = await run_selection_case(
                case, settings, environ=os.environ, adapter_factory=factory
            )
            result["runs"].append(observed)
            result["summary"] = summarize_selection(cases, result["runs"])
            result["charged_usd"] = ledger.state["charged_usd"]
            atomic_write(args.result, (json.dumps(result, indent=2) + "\n").encode())
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "selection_correct": observed["score"]["correct"],
                        "charged_usd": result["charged_usd"],
                    }
                ),
                flush=True,
            )
            if ledger.state.get("halted") or observed.get("evidence_error"):
                break
            if any(
                error["code"] == "budget_exhausted"
                for error in observed["receipt"].get("errors", [])
            ):
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "skills-dir",
        "output-dir",
        "selection-set",
        "corpus",
        "smoke-cases",
        "prepared-inputs",
        "preflight",
        "preparation",
        "ledger",
        "result",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="luna")
    parser.add_argument("--case", action="append")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--max-output", type=int, default=512)
    parser.add_argument("--request-context-cap", type=int, default=128000)
    parser.add_argument("--log-content", action="store_true")
    result = asyncio.run(evaluate_selection(parser.parse_args()))
    if not result["summary"]["population_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
