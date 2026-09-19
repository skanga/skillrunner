"""Prepare exact corpus inputs without assuming external prerequisites are ready."""

import hashlib
import importlib
import json
import shutil
import subprocess

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.qualification.inputs") is not None
    return importlib.import_module("skillrunner.qualification.inputs")


def text_input(path="nested/input.txt", content="café\r\n"):
    return {
        "path": path,
        "encoding": "utf-8",
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def document(inputs):
    return {"corpus_version": "test", "cases": [{"id": "case", "inputs": inputs}]}


def test_text_input_preserves_exact_bytes_and_records_digest(tmp_path):
    source = text_input()
    result = api().prepare_inputs(document([source]), tmp_path / "inputs")
    case = result["cases"][0]
    path = tmp_path / "inputs/case/nested/input.txt"
    assert path.read_bytes() == b"caf\xc3\xa9\r\n"
    assert case["input_paths"] == [str(path)]
    assert case["prepared_inputs"][0]["sha256"] == source["sha256"]
    assert result["case_prerequisites_verified"] is False
    with pytest.raises(FileExistsError):
        api().prepare_inputs(document([source]), tmp_path / "inputs")


def test_noncanonical_case_alias_is_rejected_before_creation(tmp_path):
    source = {
        "corpus_version": "test",
        "cases": [{"id": "case", "inputs": []}, {"id": "case/", "inputs": []}],
    }
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError, match="case ID"):
        api().prepare_inputs(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("change", ["hash", "unsafe", "duplicate", "overlap", "kind"])
def test_invalid_input_rejected_before_writing(tmp_path, change):
    inputs = [text_input()]
    if change == "hash":
        inputs[0]["sha256"] = "0" * 64
    elif change == "unsafe":
        inputs[0]["path"] = "../outside"
    elif change == "duplicate":
        inputs.append(text_input())
    elif change == "overlap":
        inputs.append(text_input("nested"))
    else:
        inputs[0]["kind"] = "run-arbitrary-script"
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError):
        api().prepare_inputs(document(inputs), destination)
    assert not destination.exists()


def test_mcp_fixture_is_a_declaration_not_a_claim_of_server_readiness(tmp_path):
    declaration = {
        "path": "notion-fixture",
        "kind": "configured-mcp-fixture",
        "records": [{"id": "a", "users": 10}],
        "required_tools": ["notion-fetch"],
    }
    result = api().prepare_inputs(document([declaration]), tmp_path / "inputs")
    case = result["cases"][0]
    assert case["input_paths"] == []
    assert case["mcp_fixtures"][0]["status"] == "server_configuration_pending"
    from pathlib import Path

    assert json.loads(Path(case["mcp_fixtures"][0]["declaration_file"]).read_text()) == declaration


def test_git_file_directory_overlap_across_commits_fails_before_writing(tmp_path):
    fixture = {
        "path": "repo",
        "kind": "synthetic-git-repository",
        "commits": [
            {
                "author": "Alice <alice@example.invalid>",
                "date": "2026-09-01T12:00:00Z",
                "files": {"a": "file"},
            },
            {
                "author": "Alice <alice@example.invalid>",
                "date": "2026-09-02T12:00:00Z",
                "files": {"a/b": "nested"},
            },
        ],
    }
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError, match="overlap"):
        api().prepare_inputs(document([fixture]), destination, git_executable=shutil.which("git"))
    assert not destination.exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="Git required for corpus history fixture")
def test_synthetic_git_history_is_repeatable_and_ignores_ambient_git_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Wrong author")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong"))
    fixture = {
        "path": "repo",
        "kind": "synthetic-git-repository",
        "commits": [
            {
                "author": "Alice <alice@example.invalid>",
                "date": "2026-09-01T12:00:00Z",
                "files": {"auth.py": "first\n"},
            },
            {
                "author": "Bob <bob@example.invalid>",
                "date": "2026-09-02T12:00:00Z",
                "files": {"auth.py": "second\n"},
            },
        ],
    }
    outputs = [
        api().prepare_inputs(
            document([fixture]), tmp_path / name, git_executable=shutil.which("git")
        )
        for name in ["one", "two"]
    ]
    histories = [r["cases"][0]["prepared_inputs"][0]["commits"] for r in outputs]
    assert histories[0] == histories[1]
    assert len(histories[0]) == 2 and len(set(histories[0])) == 2
    record = outputs[0]["cases"][0]["prepared_inputs"][0]
    assert len(record["tree_sha256"]) == 64
    root = tmp_path / "one/case/repo"
    assert (root / "auth.py").read_text() == "second\n"
    monkeypatch.delenv("GIT_DIR")
    history = subprocess.check_output(
        [shutil.which("git"), "-C", str(root), "log", "--reverse", "--format=%an|%ae|%aI"],
        text=True,
    )
    assert history.replace("Z", "+00:00").splitlines() == [
        "Alice|alice@example.invalid|2026-09-01T12:00:00+00:00",
        "Bob|bob@example.invalid|2026-09-02T12:00:00+00:00",
    ]
