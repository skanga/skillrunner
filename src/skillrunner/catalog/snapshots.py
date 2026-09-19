"""Bounded stable copies; these checks are not host-process containment."""

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skillrunner.domain.errors import RunnerError
from skillrunner.runtime.cleanup import remove_work_tree

Identity = tuple[int, int, int, int, int, int]


def file_identity(info: os.stat_result) -> Identity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        # Windows path stat and fstat can disagree on legacy ctime semantics.
        # Their explicit creation time agrees; POSIX retains change-time detection.
        int(getattr(info, "st_birthtime_ns")) if os.name == "nt" else info.st_ctime_ns,  # noqa: B009 - absent from POSIX type stubs
    )


@dataclass(frozen=True)
class TreeEntry:
    relative_path: str
    source: Path
    identity: Identity
    directory: bool
    materialized_link: bool


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    size: int
    digest: str
    materialized_link: bool


@dataclass(frozen=True)
class Snapshot:
    source: Path
    root: Path
    files: tuple[SnapshotFile, ...]
    total_bytes: int
    digest: str

    @property
    def file_count(self) -> int:
        return len(self.files)


def validate_output_locations(inputs: list[Path], output_root: Path, primary: Path | None) -> None:
    destinations = [output_root.resolve()]
    if primary is not None:
        destinations.append(primary.resolve())
    for source in inputs:
        resolved = source.resolve()
        if any(
            destination == resolved or destination.is_relative_to(resolved)
            for destination in destinations
        ):
            raise RunnerError(
                "invalid_arguments",
                "Output location overlaps an input. Choose --output-dir and --output "
                "outside the input tree.",
            )


def scan_tree(
    source: Path,
    *,
    check: Callable[[], None] | None = None,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> tuple[TreeEntry, ...]:
    """Capture metadata without reading resource bodies; materialize only internal links."""
    entries: list[TreeEntry] = []
    file_count = 0
    byte_count = 0
    root = source.resolve()
    boundary = root if root.is_dir() else root.parent

    def visit(
        path: Path,
        relative: str,
        ancestors: frozenset[tuple[int, int]],
        inherited_link: bool = False,
    ) -> None:
        nonlocal file_count, byte_count
        if check:
            check()
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(boundary):
            raise RunnerError("invalid_arguments", "A source link escapes its supplied root.")
        info = resolved.stat()
        directory = stat.S_ISDIR(info.st_mode)
        linked = inherited_link or path.is_symlink() or path.is_junction()
        if not directory and not stat.S_ISREG(info.st_mode):
            raise RunnerError("invalid_arguments", "Source contains an unsupported special file.")
        if not directory:
            file_count += 1
            byte_count += info.st_size
            if (max_files is not None and file_count > max_files) or (
                max_bytes is not None and byte_count > max_bytes
            ):
                raise RunnerError(
                    "budget_exhausted", "Snapshot exceeds configured file-count or byte limit."
                )
        identity = file_identity(info)
        entries.append(TreeEntry(relative, resolved, identity, directory, linked))
        if directory:
            key = (info.st_dev, info.st_ino)
            if key in ancestors:
                raise RunnerError("invalid_arguments", "Source contains a directory-link cycle.")
            with os.scandir(resolved) as children:
                for child in children:
                    child_relative = f"{relative}/{child.name}" if relative else child.name
                    visit(resolved / child.name, child_relative, ancestors | {key}, linked)

    try:
        visit(source, "" if root.is_dir() else source.name, frozenset())
    except RunnerError:
        raise
    except (OSError, ValueError, RecursionError) as exc:
        raise RunnerError("invalid_arguments", "Cannot enumerate a stable source tree.") from exc
    return tuple(sorted(entries, key=lambda entry: entry.relative_path))


def tree_fingerprint(entries: tuple[TreeEntry, ...]) -> str:
    data = [(e.relative_path, str(e.source), e.identity, e.materialized_link) for e in entries]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def snapshot_tree(
    source: Path,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int,
    check: Callable[[], None] | None = None,
    expected_fingerprint: str | None = None,
) -> Snapshot:
    """Publish a completed private snapshot or remove only our incomplete destination."""
    for bound in (max_files, max_bytes):
        if type(bound) is not int or bound < 0:
            raise ValueError("Snapshot bounds must be nonnegative integers")
    source = source.absolute()
    root = source.resolve()
    if destination.resolve() == root or destination.resolve().is_relative_to(root):
        raise RunnerError("invalid_arguments", "Snapshot destination overlaps its source.")
    try:
        entries = scan_tree(source, check=check, max_files=max_files, max_bytes=max_bytes)
    except RunnerError as exc:
        if expected_fingerprint is None or exc.code != "invalid_arguments":
            raise
        raise RunnerError(
            "source_changed",
            "Source changed since discovery; rerun the task.",
            status="failed",
            exit_code=6,
        ) from exc
    before = tree_fingerprint(entries)
    if expected_fingerprint is not None and before != expected_fingerprint:
        raise RunnerError(
            "source_changed",
            "Source changed since discovery; rerun the task.",
            status="failed",
            exit_code=6,
        )
    files = [entry for entry in entries if not entry.directory]
    if len(files) > max_files or sum(entry.identity[3] for entry in files) > max_bytes:
        raise RunnerError(
            "budget_exhausted", "Snapshot exceeds configured file-count or byte limit."
        )
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise RunnerError("invalid_arguments", "Cannot create a new snapshot destination.") from exc
    results: list[SnapshotFile] = []
    total = 0
    try:
        for entry in entries:
            if check:
                check()
            target = destination / entry.relative_path
            if entry.directory:
                if entry.relative_path:
                    target.mkdir(exist_ok=False)
                continue
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(entry.source, flags)
            with os.fdopen(descriptor, "rb") as reader:
                initial = os.fstat(reader.fileno())
                if not stat.S_ISREG(initial.st_mode) or file_identity(initial) != entry.identity:
                    raise RunnerError(
                        "source_changed",
                        "Source changed before copying.",
                        status="failed",
                        exit_code=6,
                    )
                digest = hashlib.sha256()
                copied = 0
                with target.open("xb") as writer:
                    while True:
                        if check:
                            check()
                        chunk = reader.read(64 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        copied += len(chunk)
                        if total > max_bytes:
                            raise RunnerError("budget_exhausted", "Snapshot byte limit exceeded.")
                        digest.update(chunk)
                        writer.write(chunk)
                if (
                    file_identity(os.fstat(reader.fileno())) != entry.identity
                    or file_identity(entry.source.stat()) != entry.identity
                ):
                    raise RunnerError(
                        "source_changed",
                        "Source changed during copying.",
                        status="failed",
                        exit_code=6,
                    )
            target.chmod(stat.S_IMODE(initial.st_mode) & 0o777)
            results.append(
                SnapshotFile(
                    entry.relative_path, copied, digest.hexdigest(), entry.materialized_link
                )
            )
        try:
            after = scan_tree(source, check=check, max_files=max_files, max_bytes=max_bytes)
        except RunnerError as exc:
            if exc.code != "invalid_arguments":
                raise
            raise RunnerError(
                "source_changed",
                "Source tree changed during copying.",
                status="failed",
                exit_code=6,
            ) from exc
        if tree_fingerprint(after) != before:
            raise RunnerError(
                "source_changed",
                "Source tree changed during copying.",
                status="failed",
                exit_code=6,
            )
        digest_data = [(item.relative_path, item.digest) for item in results]
        tree_digest = hashlib.sha256(
            json.dumps(digest_data, ensure_ascii=False).encode()
        ).hexdigest()
        return Snapshot(source, destination, tuple(results), total, tree_digest)
    except (FileNotFoundError, NotADirectoryError) as exc:
        remove_work_tree(destination)
        raise RunnerError(
            "source_changed", "Source disappeared during copying.", status="failed", exit_code=6
        ) from exc
    except BaseException:
        remove_work_tree(destination)
        raise
