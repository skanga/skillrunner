"""Verified primary-output publication with an explicit commit boundary."""

import hashlib
import os
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from skillrunner.domain.errors import RunnerError


def _identity(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise RunnerError("publication_failed", "Output destination must be a regular file.")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _protected(path: Path, roots: tuple[Path, ...]) -> None:
    resolved = path.resolve()
    destination = path.stat() if path.exists() else None
    for root in roots:
        canonical = root.resolve()
        if resolved == canonical or resolved.is_relative_to(canonical):
            raise RunnerError(
                "publication_failed", "Output destination overlaps a protected source."
            )
        if destination is None:
            continue
        entries: Iterable[Path] = root.rglob("*") if root.is_dir() else (root,)
        for entry in entries:
            if entry.is_file():
                info = entry.stat()
                if (info.st_dev, info.st_ino) == (destination.st_dev, destination.st_ino):
                    raise RunnerError(
                        "publication_failed", "Output destination aliases a protected source."
                    )


@dataclass(frozen=True)
class PublishedOutput:
    path: Path
    digest: str
    size: int
    directory_synced: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicationTarget:
    path: Path
    overwrite: bool
    protected_roots: tuple[Path, ...]
    original_identity: tuple[int, int, int, int, int] | None
    parent_identity: tuple[int, int]

    @classmethod
    def prepare(
        cls,
        destination: Path,
        *,
        overwrite: bool = False,
        protected_roots: Iterable[Path] = (),
    ) -> "PublicationTarget":
        path = Path(os.path.abspath(destination))
        roots = tuple(Path(os.path.abspath(root)) for root in protected_roots)
        try:
            _protected(path, roots)
            original = _identity(path)
            if original is not None and not overwrite:
                raise RunnerError(
                    "publication_failed", "Output already exists; overwrite is disabled."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            parent = path.parent.stat()
            return cls(path, overwrite, roots, original, (parent.st_dev, parent.st_ino))
        except OSError as exc:
            raise RunnerError("publication_failed", "Cannot prepare output destination.") from exc

    def _recheck(self) -> None:
        parent = self.path.parent.stat()
        if (parent.st_dev, parent.st_ino) != self.parent_identity:
            raise RunnerError("publication_failed", "Output parent changed after preflight.")
        _protected(self.path, self.protected_roots)
        if _identity(self.path) != self.original_identity:
            raise RunnerError("publication_failed", "Output destination changed after preflight.")

    def publish(
        self,
        candidate: Path,
        *,
        expected_digest: str,
        expected_size: int,
        max_bytes: int,
        check: Callable[[], None] | None = None,
    ) -> PublishedOutput:
        checkpoint = check or (lambda: None)
        temporary: Path | None = None
        parent_fd: int | None = None
        initiating: BaseException | None = None
        committed = False
        warnings: list[str] = []
        directory_guard = ExitStack()
        try:
            checkpoint()
            if sys.platform == "win32":
                from skillrunner.artifacts.windows_paths import lock_directory_chain

                directory_guard.enter_context(lock_directory_chain(self.path.parent))
            self._recheck()
            if sys.platform != "win32":
                parent_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                parent = os.fstat(parent_fd)
                if (parent.st_dev, parent.st_ino) != self.parent_identity:
                    raise RunnerError(
                        "publication_failed", "Output parent changed after preflight."
                    )
            if expected_size < 0 or expected_size > max_bytes:
                raise RunnerError("artifact_invalid", "Primary artifact exceeds its byte limit.")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(os.open(candidate, flags), "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                    raise RunnerError(
                        "artifact_invalid", "Primary artifact changed after validation."
                    )
                if parent_fd is not None:
                    staging_name = Path(f".{self.path.name}-{uuid.uuid4().hex}")
                    descriptor = os.open(
                        staging_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                    temporary = staging_name
                else:
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".{self.path.name}-", dir=self.path.parent
                    )
                    temporary = Path(name)
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(descriptor, "wb") as output:
                    while chunk := source.read(65536):
                        checkpoint()
                        size += len(chunk)
                        if size > expected_size or size > max_bytes:
                            raise RunnerError(
                                "artifact_invalid", "Primary artifact exceeds its byte limit."
                            )
                        digest.update(chunk)
                        output.write(chunk)
                    after = os.fstat(source.fileno())
                    if (
                        size != expected_size
                        or digest.hexdigest() != expected_digest
                        or (before.st_mtime_ns, before.st_ctime_ns)
                        != (after.st_mtime_ns, after.st_ctime_ns)
                    ):
                        raise RunnerError(
                            "artifact_invalid", "Primary artifact changed after validation."
                        )
                    output.flush()
                    os.fsync(output.fileno())
            checkpoint()
            self._recheck()
            if self.overwrite and self.original_identity is not None:
                if parent_fd is None:
                    os.replace(temporary, self.path)
                else:
                    os.replace(
                        temporary, self.path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
                    )
                temporary = None
            else:
                if parent_fd is None:
                    os.link(temporary, self.path)
                else:
                    os.link(temporary, self.path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            committed = True
            parent = self.path.parent.stat()
            if (parent.st_dev, parent.st_ino) != self.parent_identity:
                raise RunnerError(
                    "publication_failed",
                    "Output parent changed during publication.",
                    details={"committed": True, "path": str(self.path)},
                )
            synced = False
            if parent_fd is not None:
                try:
                    os.fsync(parent_fd)
                    synced = True
                except OSError:
                    warnings.append("Published output directory could not be synchronized.")
            return PublishedOutput(
                self.path, expected_digest, expected_size, synced, tuple(warnings)
            )
        except OSError as exc:
            initiating = RunnerError(
                "publication_failed",
                "Output publication failed.",
                details={"committed": committed, "path": str(self.path) if committed else None},
            )
            raise initiating from exc
        except BaseException as exc:
            initiating = exc
            raise
        finally:
            try:
                if temporary is not None:
                    try:
                        if parent_fd is None:
                            temporary.unlink(missing_ok=True)
                        else:
                            os.unlink(temporary, dir_fd=parent_fd)
                    except OSError:
                        message = "Cannot remove publication staging file."
                        if initiating is not None:
                            initiating.add_note(message)
                            if isinstance(initiating, RunnerError):
                                initiating.details.setdefault("cleanup_errors", []).append(message)
                        else:
                            raise RunnerError(
                                "publication_failed",
                                message,
                                details={"committed": committed, "path": str(self.path)},
                            ) from None
            finally:
                try:
                    if parent_fd is not None:
                        os.close(parent_fd)
                finally:
                    directory_guard.close()
