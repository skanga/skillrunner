"""Run explicitly selected, preprovisioned cases under the cumulative paid ledger."""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from skillrunner.config.sources import resolve_settings, select_model
from skillrunner.domain.request import RunRequest
from skillrunner.model.openai_compatible import OpenAICompatibleAdapter
from skillrunner.qualification.budget import PaidAdapter, SpendingLedger, StandardRates
from skillrunner.qualification.corpus import _relative
from skillrunner.qualification.inputs import input_tree_digest
from skillrunner.recording.bundle import atomic_write
from skillrunner.runtime.signals import run_with_signals

# Qualification only: user-authorized published standards, never runner model defaults.
RATES = {
    "gpt-5.6-luna": StandardRates(Decimal("0.20"), Decimal("1.20")),
    "gpt-5.5": StandardRates(Decimal("5"), Decimal("30")),
}


def qualification_rates(profile: Any) -> tuple[StandardRates, str]:
    if (
        profile.base_url == "https://ehl.infra.adobe.net/v1"
        and profile.model == "hosted_vllm/openai/gpt-oss-120b"
    ):
        return StandardRates(Decimal("0"), Decimal("0")), "User confirmed this endpoint is unbilled"
    if profile.model not in RATES:
        raise ValueError("No user-authorized qualification pricing for this model and endpoint")
    return (
        RATES[profile.model],
        "Published standards; conservative cache-write allowance, not proxy invoice",
    )


def selection_matches(active: list[str], expected: list[str], status: str) -> bool:
    if not expected:
        return not active and status == "no_matching_skill"
    return set(active) == set(expected)


def corpus_cases(
    document: dict[str, Any],
    prepared: dict[str, Any],
    preflight: dict[str, Any],
    track: str,
) -> list[dict[str, Any]]:
    """Map the full candidate population using explicit operator prerequisite evidence.

    Evidence references are attestations, not automatic proof of runtime readiness.
    Pending cases remain visible; the caller must refuse to execute them.
    """
    if track not in {"explicit", "automatic"}:
        raise ValueError("Select an explicit or automatic corpus track")
    if prepared["corpus_version"] != document["corpus_version"]:
        raise ValueError("Prepared inputs belong to another corpus version")

    def indexed(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {item["id"]: item for item in items}
        if len(result) != len(items):
            raise ValueError("Duplicate corpus case ID")
        return result

    source = indexed(document["cases"])
    inputs = indexed(prepared["cases"])
    readiness = indexed(preflight["cases"])
    if source.keys() != inputs.keys() or source.keys() != readiness.keys():
        raise ValueError("Preparation and preflight must cover the complete corpus")
    result = []
    for case_id, case in source.items():
        review = readiness[case_id]
        if review["status"] not in {"ready", "pending"}:
            raise ValueError("Unknown prerequisite status")
        if review["status"] == "ready":
            evidence = review["prerequisites"]
            if [entry["requirement"] for entry in evidence] != case["prerequisites"] or any(
                not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip()
                for entry in evidence
            ):
                raise ValueError("Every prerequisite requires matching explicit evidence")
        paths = inputs[case_id]["input_paths"]
        declarations = [
            item for item in case["inputs"] if item.get("kind") != "configured-mcp-fixture"
        ]
        mcp_declarations = [
            item for item in case["inputs"] if item.get("kind") == "configured-mcp-fixture"
        ]
        mcp_records = inputs[case_id].get("mcp_fixtures", [])
        if len(mcp_records) != len(mcp_declarations):
            raise ValueError("Prepared MCP declaration population differs from the corpus")
        for record, declaration in zip(mcp_records, mcp_declarations, strict=True):
            path = Path(record["declaration_file"])
            if (
                path.is_symlink()
                or not path.is_file()
                or json.loads(path.read_bytes()) != declaration
            ):
                raise ValueError("Prepared MCP declaration changed")
        if len(paths) != len(declarations) or len(set(paths)) != len(paths):
            raise ValueError("Prepared input population differs from the corpus")
        for value, declaration in zip(paths, declarations, strict=True):
            path = Path(value)
            suffix = _relative(declaration["path"])
            if not path.is_absolute() or path.parts[-len(suffix.parts) :] != suffix.parts:
                raise ValueError("Prepared input path does not match its declaration")
            if declaration.get("kind", "text") == "text":
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Prepared text input is not a regular file")
                if hashlib.sha256(path.read_bytes()).hexdigest() != declaration["sha256"]:
                    raise ValueError("Prepared input bytes changed")
            elif not path.is_dir() or path.is_symlink():
                raise ValueError("Prepared repository is not an ordinary directory")
            else:
                records = [
                    item
                    for item in inputs[case_id].get("prepared_inputs", [])
                    if item["path"] == value
                ]
                if len(records) != 1 or records[0].get("tree_sha256") != input_tree_digest(path):
                    raise ValueError("Prepared repository contents changed or lack a digest")
        result.append(
            {
                **case,
                "inputs": paths,
                "required_skills": case["selection"]["explicit_required_skills"]
                if track == "explicit"
                else [],
                "expected_skills": case["selection"]["automatic_expected_skills"],
                "qualification_status": review["status"],
                "prerequisite_evidence": review.get("prerequisites", []),
            }
        )
    return result


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    case_document = json.loads(args.cases.read_text())
    corpus_evidence = None
    if "corpus_version" in case_document:
        if not getattr(args, "prepared_inputs", None) or not getattr(args, "preflight", None):
            raise ValueError("Corpus execution requires prepared inputs and preflight evidence")
        prepared_bytes = args.prepared_inputs.read_bytes()
        preflight_bytes = args.preflight.read_bytes()
        prepared = json.loads(prepared_bytes)
        preflight = json.loads(preflight_bytes)
        source_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
        if (
            any(item.get("source_manifest_sha256") != source_hash for item in (prepared, preflight))
            or preflight.get("config_sha256")
            != hashlib.sha256(args.config.read_bytes()).hexdigest()
            or preflight.get("prepared_inputs_sha256") != hashlib.sha256(prepared_bytes).hexdigest()
        ):
            raise ValueError("Corpus preflight is not bound to the selected manifest/configuration")
        case_document = {
            **case_document,
            "cases": corpus_cases(case_document, prepared, preflight, args.track),
        }
        corpus_evidence = {
            "version": case_document["corpus_version"],
            "track": args.track,
            "prepared_inputs_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
            "preflight_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
            "population": case_document["cases"],
        }
    wanted = set(args.case)
    cases = [case for case in case_document["cases"] if case["id"] in wanted]
    if len(cases) != len(wanted) or not wanted or args.repeats < 1:
        raise ValueError("Select known case IDs and a positive repeat count")
    if any(case.get("qualification_status", "ready") != "ready" for case in cases):
        raise ValueError("Selected corpus case has pending prerequisites")
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
            "timeout": getattr(args, "timeout", "3m"),
        },
        {},
    )
    settings.diagnostics.log_content = bool(getattr(args, "log_content", False))
    profile = select_model(settings)
    if args.max_output < 1:
        raise ValueError("Qualification output cap must be positive")
    if profile.max_output_tokens is not None:
        profile.max_output_tokens = min(profile.max_output_tokens, args.max_output)
    rates, cost_basis = qualification_rates(profile)
    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "model": profile.model,
        "endpoint": profile.base_url,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "settings": settings.model_dump(mode="json"),
        "runs": [],
        "semantic_grading": "pending_review",
        "release_qualified": False,
        "cost_basis": cost_basis,
        "corpus": corpus_evidence,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    # Evidence must never be silently replaced or used to reset cumulative spending.
    with args.result.open("x", encoding="utf-8") as writer:
        json.dump(result, writer, indent=2)
    with SpendingLedger(args.ledger) as ledger:
        ledger.require_authorized_ceiling()

        def factory(selected: Any, key: Any) -> PaidAdapter:
            return PaidAdapter(
                OpenAICompatibleAdapter(selected, api_key=key),
                selected,
                ledger,
                rates,
                max_output=args.max_output,
            )

        for case in cases:
            for repetition in range(1, args.repeats + 1):
                request = RunRequest(
                    prompt=case["prompt"],
                    invocation_directory=Path.cwd(),
                    inputs=[Path(value).resolve() for value in case.get("inputs", [])],
                    required_skills=case.get("required_skills", []),
                    format=case.get("expected", {}).get("primary_format", "md"),
                )
                receipt = await run_with_signals(
                    request,
                    settings,
                    environ=os.environ,
                    adapter_factory=factory,
                )
                try:
                    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
                    active = [item["name"] for item in manifest["provenance"]["activated_skills"]]
                except (OSError, ValueError, KeyError, TypeError):
                    result["runs"].append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "receipt": receipt,
                            "selection_matches": None,
                            "semantic_grade": None,
                            "evidence_error": "Could not read the returned run manifest.",
                        }
                    )
                    result["evidence_status"] = "incomplete"
                    result["charged_usd"] = ledger.state["charged_usd"]
                    atomic_write(args.result, (json.dumps(result, indent=2) + "\n").encode())
                    return result
                expected = case.get("expected", {})
                output = receipt["primary_output"]
                result["runs"].append(
                    {
                        "case": case["id"],
                        "repetition": repetition,
                        "receipt": receipt,
                        "activated_skills": active,
                        "status_matches": receipt["status"] == expected.get("status", "succeeded"),
                        "selection_matches": selection_matches(
                            active, case["expected_skills"], receipt["status"]
                        )
                        if "expected_skills" in case
                        else None,
                        "primary_exists": bool(output and Path(output).is_file()),
                        "semantic_grade": None,
                    }
                )
                result["charged_usd"] = ledger.state["charged_usd"]
                atomic_write(args.result, (json.dumps(result, indent=2) + "\n").encode())
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "status": receipt["status"],
                            "charged_usd": result["charged_usd"],
                        }
                    ),
                    flush=True,
                )
                if receipt["status"] in {"cancelled", "limit_exceeded"} or ledger.state.get(
                    "halted"
                ):
                    return result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "skills-dir", "output-dir", "cases", "ledger", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--prepared-inputs", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--track", choices=("explicit", "automatic"), default="explicit")
    parser.add_argument(
        "--log-content",
        action="store_true",
        help="Opt in to redacted task/tool content in diagnostic logs",
    )
    parser.add_argument("--model", default="luna")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--max-output", type=int, default=2048)
    parser.add_argument("--timeout", default="3m", help="Per-run deadline (for example, 9m)")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args))
    if result.get("evidence_status") == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
