"""Materialize trusted corpus declarations; no dependency installation or MCP launch."""

import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from skillrunner.qualification.corpus import _digest, _relative


def input_tree_digest(root: Path) -> str:
    """Bind prepared repository bytes and file modes, including its Git metadata."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prepared repository must be an ordinary directory")
    entries = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Prepared repository contains a non-file entry")
        data = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "mode": stat.S_IMODE(mode),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return _digest(entries)


def _git_identity(commit: dict[str, Any]) -> tuple[str, str]:
    match = re.fullmatch(r"([^\n<>]+) <([^\n<>]+)>", commit["author"])
    if match is None or not commit["date"].endswith("Z"):
        raise ValueError("Git fixture requires an explicit author and UTC date")
    datetime.fromisoformat(commit["date"])
    return match[1], match[2]


def _validate_input(item: dict[str, Any]) -> None:
    _relative(item["path"])
    kind = item.get("kind", "text")
    if kind == "text":
        if item.get("encoding") != "utf-8":
            raise ValueError("Corpus text inputs require explicit UTF-8 encoding")
        if hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != item["sha256"]:
            raise ValueError("Corpus input digest mismatch")
    elif kind == "synthetic-git-repository":
        if not item["commits"]:
            raise ValueError("Git fixture needs commits")
        previous: dict[Path, Path] = {}
        for commit in item["commits"]:
            _git_identity(commit)
            for name, content in commit["files"].items():
                path = _relative(name)
                if ".git" in [part.casefold() for part in path.parts]:
                    raise ValueError("Git fixture cannot replace its metadata")
                normalized = Path(path.as_posix().casefold())
                for other, original in previous.items():
                    if normalized == other and path == original:
                        continue  # A later commit may update an existing file.
                    if normalized.is_relative_to(other) or other.is_relative_to(normalized):
                        raise ValueError("Git fixture file paths overlap")
                previous[normalized] = path
                content.encode("utf-8")
    elif kind != "configured-mcp-fixture":
        raise ValueError("Unknown corpus input kind")


def _git_fixture(root: Path, item: dict[str, Any], executable: Path) -> list[str]:
    root.mkdir(parents=True)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP")
        if key in os.environ
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ATTR_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )

    def git(*args: str) -> str:
        command = [
            str(executable),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.logAllRefUpdates=false",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "core.attributesFile=" + os.devnull,
            *args,
        ]
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            raise ValueError("Git fixture preparation failed")
        return result.stdout.decode("utf-8").strip()

    git("init", "--quiet", "--template=", "--initial-branch=main", "--object-format=sha1")
    commits = []
    for index, commit in enumerate(item["commits"], start=1):
        name, email = _git_identity(commit)
        environment.update(
            GIT_AUTHOR_NAME=name,
            GIT_COMMITTER_NAME=name,
            GIT_AUTHOR_EMAIL=email,
            GIT_COMMITTER_EMAIL=email,
            GIT_AUTHOR_DATE=commit["date"],
            GIT_COMMITTER_DATE=commit["date"],
        )
        for name, content in commit["files"].items():
            target = root / _relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        git("add", "--all")
        git("commit", "--quiet", "--allow-empty", "-m", f"Qualification fixture commit {index}")
        commits.append(git("rev-parse", "HEAD"))
    return commits


def prepare_inputs(
    document: dict[str, Any], destination: Path, *, git_executable: str | Path | None = None
) -> dict[str, Any]:
    """Prepare every positive case exclusively, leaving all prerequisites unapproved.

    MCP declarations are saved for later server provisioning and never substituted
    for a working connector. Git history uses fixed identities, dates and messages;
    working-tree index timestamps are not claimed to be reproducible Git content.
    """
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    seen: set[str] = set()
    needs_git = False
    for case in document["cases"]:
        name = _relative(case["id"])
        if len(name.parts) != 1 or name.as_posix() != case["id"] or case["id"].casefold() in seen:
            raise ValueError("Invalid or duplicate corpus case ID")
        seen.add(case["id"].casefold())
        paths: list[Path] = []
        for item in case["inputs"]:
            _validate_input(item)
            path = Path(_relative(item["path"]).as_posix().casefold())
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths):
                raise ValueError("Corpus input paths overlap")
            paths.append(path)
            needs_git |= item.get("kind") == "synthetic-git-repository"
    executable = Path(git_executable).resolve(strict=True) if git_executable else None
    if needs_git and executable is None:
        raise ValueError("Configure the preinstalled Git executable for corpus inputs")
    destination.mkdir(parents=True)
    prepared_cases = []
    for case in document["cases"]:
        root = destination / case["id"]
        root.mkdir()
        prepared: dict[str, Any] = {
            "id": case["id"],
            "input_paths": [],
            "prepared_inputs": [],
            "mcp_fixtures": [],
        }
        for item in case["inputs"]:
            path = root / _relative(item["path"])
            kind = item.get("kind", "text")
            if kind == "configured-mcp-fixture":
                path.mkdir(parents=True)
                declaration = path / "declaration.json"
                declaration.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
                prepared["mcp_fixtures"].append(
                    {"declaration_file": str(declaration), "status": "server_configuration_pending"}
                )
                continue
            if kind == "synthetic-git-repository":
                assert executable is not None
                record = {
                    "path": str(path),
                    "kind": kind,
                    "commits": _git_fixture(path, item, executable),
                    "tree_sha256": input_tree_digest(path),
                }
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(item["content"].encode("utf-8"))
                record = {"path": str(path), "kind": kind, "sha256": item["sha256"]}
            prepared["input_paths"].append(str(path))
            prepared["prepared_inputs"].append(record)
        prepared_cases.append(prepared)
    return {
        "corpus_version": document["corpus_version"],
        "cases": prepared_cases,
        "case_prerequisites_verified": False,
    }
