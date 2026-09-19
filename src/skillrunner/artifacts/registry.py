"""Register, freeze and retain generated files; validation and publication stay separate.

Copies are independent regular files, made read-only after completion. Host mode
cannot prevent another same-user process from changing their permissions or bytes;
subsequent use must verify the recorded digest, as retention does here.
"""

import hashlib
import os
import stat
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from skillrunner.catalog.snapshots import file_identity
from skillrunner.domain.errors import RunnerError
from skillrunner.tools.files import FileTools


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    path: Path
    format: str
    role: Literal["primary", "secondary"]
    description: str
    status: str = "registered"


@dataclass(frozen=True)
class FrozenArtifact:
    record: ArtifactRecord
    path: Path
    size: int
    digest: str


@dataclass(frozen=True)
class RetainedArtifact:
    record: ArtifactRecord
    path: Path
    size: int
    digest: str
    status: Literal["unvalidated", "incomplete"]


class ArtifactRegistry:
    def __init__(
        self,
        *,
        generated_roots: Iterable[Path],
        staging_root: Path,
        artifacts_root: Path,
        protected_roots: Iterable[Path] = (),
        max_bytes: int,
        check: Callable[[], None] | None = None,
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Artifact quota must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.check = check or (lambda: None)
        self._access = FileTools(check=lambda: self.check())
        self._protected_files: list[Path] = []
        self._records: dict[str, ArtifactRecord] = {}
        self._frozen: dict[str, FrozenArtifact] = {}
        self._retained: dict[str, RetainedArtifact] = {}
        try:
            self.generated_roots = tuple(root.resolve(strict=True) for root in generated_roots)
            protected_paths = tuple(root.resolve(strict=True) for root in protected_roots)
            for index, root in enumerate(self.generated_roots):
                self._access.register_root(f"generated{index}", root, writable=True)
            for index, canonical in enumerate(protected_paths):
                if canonical.is_dir():
                    self._access.register_root(f"protected{index}", canonical)
                else:
                    self._protected_files.append(canonical)
            self.staging_root = staging_root.resolve(strict=True)
            self.artifacts_root = artifacts_root.resolve(strict=True)
            if any(
                destination.is_relative_to(protected)
                for destination in (self.staging_root, self.artifacts_root)
                for protected in protected_paths
            ):
                raise RunnerError(
                    "invalid_arguments", "Artifact destinations overlap protected roots."
                )
            if self.staging_root.is_relative_to(
                self.artifacts_root
            ) or self.artifacts_root.is_relative_to(self.staging_root):
                raise RunnerError(
                    "invalid_arguments", "Staging and retained roots must be separate."
                )
            self._access.register_root("staging", self.staging_root)
            self._access.register_root("retained", self.artifacts_root, writable=True)
        except OSError:
            raise RunnerError(
                "invalid_arguments", "Artifact roots must exist and be accessible."
            ) from None

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records.values())

    @property
    def retained(self) -> tuple[RetainedArtifact, ...]:
        return tuple(self._retained.values())

    def _generated(self, path: Path) -> Path:
        self.check()
        try:
            _, actual, _ = self._access._resolve(str(path), write=True)
            if not any(actual.is_relative_to(root) for root in self.generated_roots):
                raise RunnerError(
                    "artifact_invalid", "Artifact must be a generated workspace file."
                )
            self._access._protected_alias(actual)
            info = actual.stat()
            for protected in self._protected_files:
                other = protected.stat()
                if (info.st_dev, info.st_ino) == (other.st_dev, other.st_ino):
                    raise RunnerError("artifact_invalid", "Artifact aliases a protected source.")
            with self._access._open(actual) as source:
                if os.fstat(source.fileno()).st_nlink != 1:
                    raise RunnerError(
                        "artifact_invalid", "Artifact candidates cannot have hard-link aliases."
                    )
            return actual
        except RunnerError as error:
            if error.code not in {"file_access_denied", "source_changed"}:
                raise
            raise RunnerError(
                "artifact_invalid", "Artifact is not an approved regular generated file."
            ) from None
        except OSError:
            raise RunnerError(
                "artifact_invalid", "Cannot access a generated artifact safely."
            ) from None

    def register(
        self, path: Path, *, format: str, role: Literal["primary", "secondary"], description: str
    ) -> ArtifactRecord:
        if (
            role not in {"primary", "secondary"}
            or not isinstance(format, str)
            or not format.strip()
            or not isinstance(description, str)
        ):
            raise RunnerError(
                "invalid_arguments", "Artifact requires a format, valid role and description."
            )
        actual = self._generated(path)
        record = ArtifactRecord(uuid.uuid4().hex, actual, format, role, description)
        self._records[record.id] = record
        return record

    def select_primary(self) -> ArtifactRecord | None:
        self.check()
        primaries = [item for item in self._records.values() if item.role == "primary"]
        if len(primaries) > 1:
            raise RunnerError(
                "missing_decision",
                "Choose one primary artifact before publication.",
                details={"artifact_ids": [item.id for item in primaries]},
            )
        return primaries[0] if primaries else None

    def freeze(self, record: ArtifactRecord, *, writers_stopped: bool) -> FrozenArtifact:
        self.check()
        if writers_stopped is not True:
            raise RunnerError(
                "artifact_invalid", "Relevant writers must stop before freezing artifacts."
            )
        if self._records.get(record.id) is not record:
            raise RunnerError(
                "artifact_invalid", "Artifact record does not belong to this registry."
            )
        if record.id in self._frozen:
            frozen = self._frozen[record.id]
            self._verify(frozen.path, frozen.size, frozen.digest)
            return frozen
        source = self._generated(record.path)
        destination = self.staging_root / record.id
        size, digest = self._copy(source, destination)
        frozen = FrozenArtifact(record, destination, size, digest)
        self._frozen[record.id] = frozen
        return frozen

    def retain(self, frozen: FrozenArtifact, *, incomplete: bool = False) -> RetainedArtifact:
        self.check()
        if type(incomplete) is not bool:
            raise RunnerError("invalid_arguments", "Incomplete must be an explicit boolean.")
        if self._frozen.get(frozen.record.id) is not frozen:
            raise RunnerError(
                "artifact_invalid", "Frozen candidate does not belong to this registry."
            )
        previous = self._retained.get(frozen.record.id)
        if previous is not None:
            self._verify(previous.path, previous.size, previous.digest)
            if incomplete and previous.status != "incomplete":
                previous = replace(previous, status="incomplete")
                self._retained[frozen.record.id] = previous
            return previous
        destination = self.artifacts_root / frozen.record.id
        size, digest = self._copy(
            frozen.path, destination, expected_size=frozen.size, expected_digest=frozen.digest
        )
        retained = RetainedArtifact(
            frozen.record, destination, size, digest, "incomplete" if incomplete else "unvalidated"
        )
        self._retained[frozen.record.id] = retained
        return retained

    def _verify(self, path: Path, size: int, digest: str) -> None:
        try:
            with self._access._open(path) as stream:
                if (
                    os.fstat(stream.fileno()).st_size != size
                    or os.fstat(stream.fileno()).st_nlink != 1
                    or self._access._hash(stream) != digest
                ):
                    raise RunnerError("artifact_invalid", "Immutable candidate bytes changed.")
        except RunnerError as error:
            if error.code not in {"file_access_denied", "source_changed"}:
                raise
            raise RunnerError(
                "artifact_invalid", "Immutable candidate changed or disappeared."
            ) from None

    def _used(self, root: Path) -> int:
        used = 0
        for path in self._access._tree(root):
            self.check()
            used += path.stat().st_size
            if used > self.max_bytes:
                raise RunnerError("budget_exhausted", "Artifact storage quota exceeded.")
        return used

    def _copy(
        self,
        source: Path,
        destination: Path,
        *,
        expected_size: int | None = None,
        expected_digest: str | None = None,
    ) -> tuple[int, str]:
        created = False
        created_identity: tuple[int, int] | None = None
        name: str | Path = destination
        try:
            self.check()
            used = self._used(destination.parent)
            with (
                self._access._open(source) as reader,
                self._access._directory(destination.parent) as parent,
            ):
                info = os.fstat(reader.fileno())
                if info.st_nlink != 1:
                    raise RunnerError(
                        "artifact_invalid", "Artifact candidates cannot have hard-link aliases."
                    )
                if info.st_size + used > self.max_bytes:
                    raise RunnerError("budget_exhausted", "Artifact storage quota exceeded.")
                if expected_size is not None and info.st_size != expected_size:
                    raise RunnerError("artifact_invalid", "Frozen candidate size changed.")
                initial_digest = self._access._hash(reader)
                reader.seek(0)
                name = destination.name if parent is not None else destination
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                    0o600,
                    dir_fd=parent,
                )
                created = True
                copied_info = os.fstat(descriptor)
                created_identity = copied_info.st_dev, copied_info.st_ino
                try:
                    digest = hashlib.sha256()
                    size = 0
                    with os.fdopen(descriptor, "wb") as writer:
                        while True:
                            self.check()
                            chunk = reader.read(65_536)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size + used > self.max_bytes:
                                raise RunnerError(
                                    "budget_exhausted", "Artifact storage quota exceeded."
                                )
                            digest.update(chunk)
                            writer.write(chunk)
                        value = digest.hexdigest()
                        if (
                            file_identity(os.fstat(reader.fileno())) != file_identity(info)
                            or os.fstat(reader.fileno()).st_nlink != 1
                            or file_identity(source.stat()) != file_identity(info)
                            or value != initial_digest
                            or (expected_digest is not None and value != expected_digest)
                        ):
                            raise RunnerError(
                                "artifact_invalid", "Artifact changed during copying."
                            )
                        writer.flush()
                        os.fsync(writer.fileno())
                        if os.name == "posix":
                            os.fchmod(writer.fileno(), stat.S_IRUSR)
                    if os.name != "posix":
                        destination.chmod(stat.S_IREAD)
                    if parent is not None:
                        os.fsync(parent)
                    self._access._check_root_identities(destination.parent)
                    installed = destination.lstat()
                    if (installed.st_dev, installed.st_ino) != created_identity:
                        raise RunnerError(
                            "artifact_invalid", "Artifact destination changed during copying."
                        )
                except BaseException:
                    with suppress(OSError):
                        self._unlink_owned(name, parent, created_identity)
                    created = False
                    raise
            return size, value
        except BaseException as error:
            # An enclosing source identity check may fail after the copy has
            # completed. Remove only the destination this call created.
            if created:
                with suppress(OSError, RunnerError):
                    self._access._check_root_identities(destination.parent)
                    self._unlink_owned(destination, None, created_identity)
            if isinstance(error, RunnerError) and error.code in {
                "file_access_denied",
                "source_changed",
            }:
                raise RunnerError("artifact_invalid", "Artifact changed during copying.") from None
            if isinstance(error, OSError):
                raise RunnerError(
                    "artifact_invalid", "Cannot preserve this artifact safely."
                ) from None
            raise

    @staticmethod
    def _unlink_owned(
        name: str | Path, parent: int | None, identity: tuple[int, int] | None
    ) -> None:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity:
            if os.name != "posix":
                os.chmod(name, stat.S_IWRITE | stat.S_IREAD)
            os.unlink(name, dir_fd=parent)
