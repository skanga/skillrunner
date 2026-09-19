import copy
import hashlib
import json

import pytest

from skillrunner.qualification import run


def fixture(tmp_path):
    source = tmp_path / "input.txt"
    source.write_bytes(b"exact input\r\n")
    case = {
        "id": "writer",
        "prompt": "Transform the input",
        "inputs": [
            {"path": "input.txt", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
        "prerequisites": ["Runtime available"],
        "selection": {
            "explicit_required_skills": ["writer"],
            "automatic_expected_skills": ["writer"],
        },
        "expected": {"primary_format": "md"},
        "rubric": {"criteria": ["Retain the facts"]},
    }
    corpus = {"corpus_version": "v1", "cases": [case]}
    prepared = {"corpus_version": "v1", "cases": [{"id": "writer", "input_paths": [str(source)]}]}
    preflight = {
        "cases": [
            {
                "id": "writer",
                "status": "ready",
                "prerequisites": [
                    {"requirement": "Runtime available", "evidence": "runtime-probe.json"}
                ],
            }
        ]
    }
    return corpus, prepared, preflight, source


@pytest.mark.parametrize("track", ["explicit", "automatic"])
def test_corpus_mapping_preserves_rubric_and_selection_track(tmp_path, track):
    corpus, prepared, preflight, source = fixture(tmp_path)
    original = copy.deepcopy(corpus)
    result = run.corpus_cases(corpus, prepared, preflight, track)
    assert result[0]["inputs"] == [str(source)]
    assert result[0]["required_skills"] == (["writer"] if track == "explicit" else [])
    assert result[0]["expected_skills"] == ["writer"]
    assert result[0]["rubric"] == corpus["cases"][0]["rubric"]
    assert result[0]["qualification_status"] == "ready"
    assert corpus == original


def test_pending_case_remains_in_frozen_population(tmp_path):
    corpus, prepared, preflight, _ = fixture(tmp_path)
    preflight["cases"][0].update(status="pending", prerequisites=[])
    result = run.corpus_cases(corpus, prepared, preflight, "automatic")
    assert len(result) == 1
    assert result[0]["qualification_status"] == "pending"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_case",
        "duplicate_case",
        "missing_requirement",
        "changed_requirement",
        "empty_evidence",
        "changed_input",
        "extra_input",
        "wrong_version",
    ],
)
def test_corpus_mapping_rejects_incomplete_or_stale_preparation(tmp_path, fault):
    corpus, prepared, preflight, source = fixture(tmp_path)
    if fault == "missing_case":
        preflight["cases"] = []
    elif fault == "duplicate_case":
        prepared["cases"] *= 2
    elif fault == "missing_requirement":
        preflight["cases"][0]["prerequisites"] = []
    elif fault == "changed_requirement":
        preflight["cases"][0]["prerequisites"][0]["requirement"] = "Different"
    elif fault == "empty_evidence":
        preflight["cases"][0]["prerequisites"][0]["evidence"] = ""
    elif fault == "changed_input":
        source.write_bytes(b"changed")
    elif fault == "extra_input":
        prepared["cases"][0]["input_paths"].append(str(source))
    else:
        prepared["corpus_version"] = "different"
    with pytest.raises(ValueError):
        run.corpus_cases(corpus, prepared, preflight, "explicit")


async def test_raw_corpus_without_preflight_is_rejected_before_any_effect(tmp_path):
    from argparse import Namespace

    corpus, _, _, _ = fixture(tmp_path)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    args = Namespace(cases=path, case=["writer"], repeats=3, result=tmp_path / "result.json")
    with pytest.raises(ValueError, match="preflight"):
        await run.evaluate(args)
    assert not args.result.exists()


def test_repository_changes_invalidate_prepared_input(tmp_path):
    import shutil

    from skillrunner.qualification.inputs import prepare_inputs

    corpus, _, preflight, _ = fixture(tmp_path)
    corpus["cases"][0]["inputs"] = [
        {
            "path": "repo",
            "kind": "synthetic-git-repository",
            "commits": [
                {
                    "author": "A <a@example.invalid>",
                    "date": "2026-09-01T00:00:00Z",
                    "files": {"a.txt": "original"},
                }
            ],
        }
    ]
    prepared = prepare_inputs(corpus, tmp_path / "prepared", git_executable=shutil.which("git"))
    assert run.corpus_cases(corpus, prepared, preflight, "explicit")
    (tmp_path / "prepared/writer/repo/a.txt").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        run.corpus_cases(corpus, prepared, preflight, "explicit")


def test_mcp_declaration_must_match_corpus_before_execution(tmp_path):
    from skillrunner.qualification.inputs import prepare_inputs

    corpus, _, preflight, _ = fixture(tmp_path)
    corpus["cases"][0]["inputs"] = [
        {
            "kind": "configured-mcp-fixture",
            "path": "notion",
            "destination": "qualification-only",
            "records": [],
        }
    ]
    prepared = prepare_inputs(corpus, tmp_path / "prepared")
    assert run.corpus_cases(corpus, prepared, preflight, "explicit")
    path = tmp_path / "prepared/writer/notion/declaration.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="MCP declaration"):
        run.corpus_cases(corpus, prepared, preflight, "explicit")
