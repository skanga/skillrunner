"""Selection qualification stops before task actions and grades exact activations."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from skillrunner.model.protocol import ModelReply, ModelToolCall, ModelUsage


def api():
    return importlib.import_module("skillrunner.qualification.selection")


def reply(*names: str) -> ModelReply:
    calls = tuple(ModelToolCall(str(i), name, {}, "{}") for i, name in enumerate(names, 1))
    return ModelReply(None, calls, "tool_calls", ModelUsage(10, 5, 15), "provider-1")


class Adapter:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.closed = False

    async def complete(self, messages, tool_schemas, output_limit, request_deadline):
        return next(self.replies)

    async def discover_capabilities(self, deadline):
        return None

    async def aclose(self):
        self.closed = True


async def test_selection_adapter_preserves_activation_and_intercepts_first_task_action():
    inner = Adapter([reply("activate_skill"), reply("write_file")])
    selected = api().SelectionAdapter(inner)

    first = await selected.complete([], [], 100, 1.0)
    second = await selected.complete([], [], 100, 1.0)

    assert first.tool_calls[0].name == "activate_skill"
    assert [call.name for call in second.tool_calls] == ["finish_run"]
    assert second.tool_calls[0].arguments["outcome"] == "blocked"
    assert second.usage == ModelUsage(10, 5, 15)
    assert selected.intercepted_tool_names == ["write_file"]
    assert selected.decision_reached
    await selected.aclose()
    assert inner.closed


async def test_selection_adapter_never_dispatches_a_mixed_activation_and_action_batch():
    selected = api().SelectionAdapter(Adapter([reply("activate_skill", "run_command")]))

    observed = await selected.complete([], [], 100, 1.0)

    assert [call.name for call in observed.tool_calls] == ["finish_run"]
    assert selected.intercepted_tool_names == ["activate_skill", "run_command"]


async def test_selection_adapter_preserves_real_no_match_completion():
    actual = reply("finish_run")
    selected = api().SelectionAdapter(Adapter([actual]))

    assert await selected.complete([], [], 100, 1.0) is actual
    assert selected.decision_reached
    assert selected.intercepted_tool_names == []


def test_selection_score_requires_a_decision_and_reports_false_activations():
    score = api().score_selection
    assert score(["writer"], ["writer"], decision_reached=True, status="blocked") == {
        "correct": True,
        "false_activations": [],
        "missed_skills": [],
    }
    assert score(["writer"], ["other"], decision_reached=True, status="blocked") == {
        "correct": False,
        "false_activations": ["writer"],
        "missed_skills": ["other"],
    }
    assert score([], [], decision_reached=True, status="no_matching_skill")["correct"]
    assert not score([], [], decision_reached=True, status="blocked")["correct"]
    assert not score(["writer"], ["writer"], decision_reached=False, status="limit_exceeded")[
        "correct"
    ]


def test_frozen_selection_population_detects_label_or_prompt_drift(tmp_path):
    corpus = {
        "corpus_version": "v1",
        "cases": [
            {
                "id": "writer",
                "prompt": "Write",
                "selection": {"automatic_expected_skills": ["writer"]},
            }
        ],
    }
    smoke = {"cases": [{"id": "no-match", "prompt": "No matching skill", "expected_skills": []}]}
    corpus_path = tmp_path / "corpus.json"
    smoke_path = tmp_path / "smoke.json"
    corpus_path.write_text(json.dumps(corpus))
    smoke_path.write_text(json.dumps(smoke))
    frozen = api().freeze_selection_population(corpus_path, smoke_path)
    assert [case["id"] for case in frozen] == ["writer", "smoke/no-match"]
    assert frozen[0]["expected_skills"] == ["writer"]

    corpus["cases"][0]["prompt"] = "Changed"
    corpus_path.write_text(json.dumps(corpus))
    try:
        api().verify_selection_population(frozen, corpus_path, smoke_path)
    except ValueError as error:
        assert "drift" in str(error)
    else:
        raise AssertionError("Changed prompt must invalidate frozen selection labels")


def test_refrozen_selection_set_keeps_full_population_requirement():
    validate = api().validate_selection_set
    cases = [{"id": str(index)} for index in range(22)]
    selection = {
        "set_id": "full-catalog-automatic-selection-v2",
        "total_cases": 22,
        "positive_cases": 21,
        "no_match_cases": 1,
    }
    validate(selection, cases)

    for invalid in (
        {**selection, "set_id": "unrelated-v2"},
        {**selection, "total_cases": 21},
        {**selection, "no_match_cases": 0},
    ):
        try:
            validate(invalid, cases)
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("Invalid frozen selection metadata must be rejected")


async def test_selection_case_records_activated_skill_before_task_action(tmp_path):
    from tests.integration.test_coordinator import Adapter as CoordinatorAdapter
    from tests.integration.test_coordinator import call, fixture

    settings = fixture(tmp_path)
    case = {
        "id": "writer",
        "prompt": "Write a report",
        "inputs": [],
        "format": "md",
        "expected_skills": ["writer"],
    }
    result = await api().run_selection_case(
        case,
        settings,
        environ={},
        adapter_factory=lambda profile, key: CoordinatorAdapter(
            profile,
            [
                [call("activate_skill", {"name": "writer", "reason": "Matches the task"})],
                [call("write_file", {"path": "artifacts/should-not-exist.md", "content": "bad"})],
            ],
        ),
    )
    assert result["score"]["correct"]
    assert result["activated_skills"] == ["writer"]
    assert result["intercepted_tool_names"] == ["write_file"]
    assert result["receipt"]["status"] == "blocked"
    assert result["receipt"]["primary_output"] is None
    manifest = json.loads(Path(result["receipt"]["manifest_path"]).read_text())
    assert manifest["provenance"]["activated_skills"][0]["name"] == "writer"
    assert not list(tmp_path.rglob("should-not-exist.md"))


async def test_selection_case_grades_no_match_without_activating_a_skill(tmp_path):
    from tests.integration.test_coordinator import Adapter as CoordinatorAdapter
    from tests.integration.test_coordinator import finish, fixture

    settings = fixture(tmp_path)
    result = await api().run_selection_case(
        {
            "id": "no-match",
            "prompt": "Transfer payroll using an unavailable service",
            "inputs": [],
            "format": "md",
            "expected_skills": [],
        },
        settings,
        environ={},
        adapter_factory=lambda profile, key: CoordinatorAdapter(
            profile, [finish("no_matching_skill")]
        ),
    )
    assert result["score"]["correct"]
    assert result["activated_skills"] == []
    assert result["receipt"]["status"] == "no_matching_skill"


def test_selection_cases_use_verified_inputs_without_task_readiness(tmp_path):
    corpus = {
        "corpus_version": "v1",
        "cases": [
            {
                "id": "writer",
                "prompt": "Write with this skill",
                "inputs": [],
                "selection": {
                    "automatic_expected_skills": ["writer"],
                    "explicit_required_skills": ["writer"],
                },
                "expected": {"primary_format": "md"},
                "prerequisites": ["External prerequisite remains pending"],
            }
        ],
    }
    smoke = {"cases": [{"id": "no-match", "prompt": "No skill", "expected_skills": []}]}
    prepared = {"corpus_version": "v1", "cases": [{"id": "writer", "input_paths": []}]}
    preflight = {"cases": [{"id": "writer", "status": "pending", "prerequisites": []}]}
    paths = {}
    for name, value in {
        "corpus": corpus,
        "smoke": smoke,
        "prepared": prepared,
        "preflight": preflight,
    }.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    selection = {
        "corpus_sha256": hashlib.sha256(paths["corpus"].read_bytes()).hexdigest(),
        "prepared_inputs_sha256": hashlib.sha256(paths["prepared"].read_bytes()).hexdigest(),
        "smoke_cases_sha256": hashlib.sha256(paths["smoke"].read_bytes()).hexdigest(),
        "population": api().freeze_selection_population(paths["corpus"], paths["smoke"]),
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    cases = api().load_selection_cases(
        selection_path,
        paths["corpus"],
        paths["smoke"],
        paths["prepared"],
        paths["preflight"],
    )
    assert [(case["id"], case["expected_skills"]) for case in cases] == [
        ("writer", ["writer"]),
        ("smoke/no-match", []),
    ]

    prepared["cases"].append({"id": "unlabeled", "input_paths": []})
    paths["prepared"].write_text(json.dumps(prepared))
    try:
        api().load_selection_cases(
            selection_path,
            paths["corpus"],
            paths["smoke"],
            paths["prepared"],
            paths["preflight"],
        )
    except ValueError as error:
        assert "drift" in str(error)
    else:
        raise AssertionError("Changed prepared inputs must invalidate selection")


def test_selection_summary_requires_complete_frozen_population_and_counts_false_activations():
    selected = api()
    expected = [
        {"id": "writer", "expected_skills": ["writer"]},
        {"id": "no-match", "expected_skills": []},
    ]
    runs = [
        {
            "case": "writer",
            "decision_reached": True,
            "score": {"correct": True, "false_activations": []},
        },
        {
            "case": "no-match",
            "decision_reached": True,
            "score": {"correct": False, "false_activations": ["writer"]},
        },
    ]
    summary = selected.summarize_selection(expected, runs)
    assert summary["correct"] == 1
    assert summary["total"] == 2
    assert summary["false_activations"] == 1
    assert summary["selection_gate_passed"] is False
    assert selected.summarize_selection(expected, runs[:1])["selection_gate_passed"] is False


@pytest.mark.parametrize("no_match", [False, True])
async def test_selection_uses_called_executor_when_unused_inspector_is_created_last(
    tmp_path, no_match
):
    from tests.integration.test_coordinator import Adapter as CoordinatorAdapter
    from tests.integration.test_coordinator import call, finish, fixture

    settings = fixture(tmp_path)
    profile = settings.models["test"].model_copy(
        update={
            "input_modalities": ["text", "image"],
            "image_accounting": "gemma4-image-max-v1",
        }
    )
    settings.models["test"] = profile
    settings.models["vision"] = profile
    settings.image_inspector = "vision"
    adapters = []

    def factory(selected, key):
        calls = (
            [finish("no_matching_skill")]
            if no_match
            else [
                [call("activate_skill", {"name": "writer", "reason": "Matches task"})],
                [call("inspect_image", {"path": "scratch/never.png", "question": "Inspect"})],
            ]
        )
        adapter = CoordinatorAdapter(selected, calls)
        adapters.append(adapter)
        return adapter

    result = await api().run_selection_case(
        {
            "id": "test",
            "prompt": "Choose skills",
            "inputs": [],
            "format": "md",
            "expected_skills": [] if no_match else ["writer"],
        },
        settings,
        environ={},
        adapter_factory=factory,
    )
    assert len(adapters) == 2 and all(a.closed for a in adapters)
    assert adapters[0].requests and not adapters[1].requests
    assert result["decision_reached"]
    assert result["score"]["correct"]
    assert result["intercepted_tool_names"] == ([] if no_match else ["inspect_image"])
