"""Prepare unchanged candidate packages from an already pinned, trusted corpus manifest.

This is a local qualification utility, not a package downloader or dependency installer.
Hashes bind checkout bytes to the reviewed manifest; no upstream code is executed.
"""

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any


def _relative(value: str) -> Path:
    parts = PurePosixPath(value).parts
    if (
        not parts
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Unsafe corpus path")
    return Path(*parts)


def _digest(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(entries, key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify(root: Path, entries: list[dict[str, Any]]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Corpus package must be an ordinary directory")
    expected = {item["path"]: item for item in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate corpus file path")
    for expected_entry in entries:
        _relative(expected_entry["path"])
    actual = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        name = path.relative_to(root).as_posix()
        actual.add(name)
        entry = expected.get(name)
        if entry is None:
            raise ValueError(f"Unexpected corpus file: {name}")
        if stat.S_ISLNK(info.st_mode):
            mode = "120000"
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Corpus symlink escapes package")
            content = os.fsencode(os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
            content = path.read_bytes()
        else:
            raise ValueError("Corpus contains a non-file entry")
        if mode != entry["mode"]:
            raise ValueError(f"Corpus file mode mismatch: {name}")
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Corpus file bytes mismatch: {name}")
    if actual != set(expected):
        raise ValueError("Corpus package is missing recorded files")


def prepare_catalog(
    document: dict[str, Any], checkouts: dict[str, Path], destination: Path
) -> dict[str, Any]:
    """Verify all candidates, copy whole packages, then verify again.

    Existing destinations are never reused. A failed copy leaves its partial catalog
    available for investigation; no successful preparation evidence is returned.
    This does not establish runtime prerequisites or qualification eligibility.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    sources: list[tuple[dict[str, Any], Path, str]] = []
    names: set[str] = set()
    for package in document["packages"]:
        if package["classification"] != "candidate":
            continue
        relative = _relative(package["path"])
        name = relative.name
        if name.casefold() in names:
            raise ValueError("Duplicate corpus destination name")
        names.add(name.casefold())
        checkout = checkouts[package["repository"]].resolve(strict=True)
        root = checkout / relative
        if not root.resolve().is_relative_to(checkout):
            raise ValueError("Corpus source escapes checkout")
        if destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("Corpus destination overlaps its source")
        if _digest(package["files"]) != package["package_sha256"]:
            raise ValueError("Corpus package manifest digest mismatch")
        _verify(root, package["files"])
        sources.append((package, root, name))
    if not sources:
        raise ValueError("Corpus contains no candidate packages")
    destination.mkdir(parents=True, exist_ok=False)
    prepared = []
    for package, source, name in sources:
        target = destination / name
        shutil.copytree(source, target, symlinks=True)
        _verify(target, package["files"])
        prepared.append(
            {
                "id": package["id"],
                "path": str(target.resolve()),
                "package_sha256": package["package_sha256"],
            }
        )
    return {
        "corpus_version": document["corpus_version"],
        "packages": prepared,
        "case_prerequisites_verified": False,
    }
