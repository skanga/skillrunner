"""Workspace file operations; host mode is not a same-user security sandbox.

Paths are ``root-name/relative/path`` or absolute paths beneath registered roots.
Read offsets/lengths count UTF-8 bytes; offsets must start at a codepoint boundary.
Root registration is dynamic, so the coordinator can expose newly activated skills.
POSIX uses descriptor-relative no-follow operations; Windows uses identity rechecks
and cannot promise equivalent protection against concurrent reparse-point changes.
"""

import codecs
import fnmatch
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from skillrunner.domain.errors import RunnerError


@dataclass(frozen=True)
class _Root:
    name: str
    path: Path
    writable: bool
    max_bytes: int | None
    identity: tuple[int, int]


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _version(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _error(code: str, message: str) -> RunnerError:
    return RunnerError(code, message)


def _count(value: int, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise _error(
            "invalid_arguments",
            "Expected a valid nonnegative count." if zero else "Expected a positive count.",
        )


class FileTools:
    def __init__(
        self,
        *,
        max_read_bytes: int = 65_536,
        max_tool_output_bytes: int = 1_048_576,
        check: Callable[[], None] | None = None,
    ) -> None:
        _count(max_read_bytes)
        _count(max_tool_output_bytes)
        self.max_read_bytes = max_read_bytes
        self.max_tool_output_bytes = max_tool_output_bytes
        self.check = check or (lambda: None)
        self._roots: dict[str, _Root] = {}

    def register_root(
        self, name: str, path: Path, *, writable: bool = False, max_bytes: int | None = None
    ) -> None:
        self.check()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in self._roots:
            raise _error("invalid_arguments", "Root names must be unique simple identifiers.")
        if max_bytes is not None:
            _count(max_bytes, zero=True)
        try:
            canonical = path.resolve(strict=True)
            info = canonical.stat()
            if not stat.S_ISDIR(info.st_mode):
                raise OSError
        except (OSError, RuntimeError):
            raise _error(
                "file_access_denied", "An approved root must be an existing directory."
            ) from None
        self._roots[name] = _Root(name, canonical, writable, max_bytes, _identity(info))

    def _root_error(self, message: str) -> RunnerError:
        return RunnerError(
            "file_access_denied",
            message,
            details={
                "workspace_roots": {name: str(root.path) for name, root in self._roots.items()},
                "suggested_action": (
                    "Use a registered root shown in workspace_roots; host commands need its "
                    "absolute filesystem path. For an external input, restart with --input."
                ),
            },
        )

    def _resolve(self, value: str, *, write: bool = False) -> tuple[_Root, Path, str]:
        self.check()
        if not isinstance(value, str) or not value or "\x00" in value:
            raise _error("invalid_arguments", "A valid workspace path is required.")
        path = Path(value)
        if ".." in path.parts:
            raise _error("file_access_denied", "Parent traversal is not allowed.")
        if not path.is_absolute():
            root = self._roots.get(path.parts[0]) if path.parts else None
            if root is None:
                raise self._root_error("Use a registered workspace root.")
            path = root.path.joinpath(*path.parts[1:])
        if not any(path.is_relative_to(root.path) for root in self._roots.values()):
            raise self._root_error("Path is outside approved workspace roots.")
        try:
            canonical = path.resolve(strict=False)
            candidates = [
                root for root in self._roots.values() if canonical.is_relative_to(root.path)
            ]
            if not candidates:
                raise OSError
            root = max(candidates, key=lambda item: len(item.path.parts))
            if _identity(root.path.stat()) != root.identity:
                raise OSError
        except (OSError, RuntimeError):
            raise self._root_error("The path no longer belongs to an approved root.") from None
        if write and (
            not root.writable
            or any(
                not item.writable and canonical.is_relative_to(item.path)
                for item in self._roots.values()
            )
        ):
            raise _error("file_access_denied", "Inputs and skill packages are read-only.")
        logical = root.name + (
            "/" + canonical.relative_to(root.path).as_posix() if canonical != root.path else ""
        )
        return root, canonical, logical

    @contextmanager
    def _directory(self, path: Path) -> Iterator[int | None]:
        if sys.platform == "win32":
            self._check_root_identities(path)
            before = path.stat()
            if not stat.S_ISDIR(before.st_mode) or path.resolve() != path:
                raise _error("file_access_denied", "Directory changed during access.")
            yield None
            self._check_root_identities(path)
            if _identity(path.stat()) != _identity(before):
                raise _error("file_access_denied", "Directory changed during access.")
            return
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            opened = Path(path.anchor)
            for part in path.parts[1:]:
                self.check()
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = child
                opened /= part
                for root in self._roots.values():
                    if opened == root.path and _identity(os.fstat(descriptor)) != root.identity:
                        raise _error("file_access_denied", "Approved root changed during access.")
            yield descriptor
        finally:
            os.close(descriptor)

    def _check_root_identities(self, path: Path) -> None:
        for root in self._roots.values():
            if path.is_relative_to(root.path) and _identity(root.path.stat()) != root.identity:
                raise _error("file_access_denied", "Approved root changed during access.")

    @contextmanager
    def _open(self, path: Path) -> Iterator[BinaryIO]:
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise _error("file_access_denied", "Only regular files can be read or edited.")
            with self._directory(path.parent) as parent:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                descriptor = os.open(
                    path.name if parent is not None else path, flags, dir_fd=parent
                )
                with os.fdopen(descriptor, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise _error(
                            "file_access_denied", "Only regular files can be read or edited."
                        )
                    if _identity(path.stat()) != _identity(before):
                        raise _error("file_access_denied", "File identity changed during access.")
                    yield stream
                    if _version(os.fstat(stream.fileno())) != _version(before) or _version(
                        path.stat()
                    ) != _version(before):
                        raise _error("source_changed", "File changed while it was being read.")
        except OSError:
            raise _error(
                "file_access_denied", "Cannot access this workspace file safely."
            ) from None

    def _hash(self, stream: BinaryIO) -> str:
        stream.seek(0)
        digest = hashlib.sha256()
        while True:
            self.check()
            chunk = stream.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()

    def _fits(self, value: dict[str, Any]) -> bool:
        return (
            len(json.dumps(value, ensure_ascii=True).encode("utf-8")) <= self.max_tool_output_bytes
        )

    def _bounded(self, value: dict[str, Any]) -> dict[str, Any]:
        if not self._fits(value):
            raise _error(
                "budget_exhausted", "Tool output limit cannot fit this result; increase it."
            )
        return value

    def read_text(self, path: str, *, offset: int = 0, length: int | None = None) -> dict[str, Any]:
        """Return a UTF-8 byte page and full-file SHA-256; next_offset preserves codepoints."""
        _count(offset, zero=True)
        if length is not None:
            _count(length)
        _, actual, logical = self._resolve(path)
        with self._open(actual) as stream:
            size = os.fstat(stream.fileno()).st_size
            digest = self._hash(stream)
            stream.seek(offset)
            data = stream.read(min(length or self.max_read_bytes, self.max_read_bytes))
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                if error.reason == "unexpected end of data" and offset + len(data) < size:
                    text = data[: error.start].decode("utf-8")
                else:
                    raise _error(
                        "invalid_arguments", "Read requires UTF-8 text and an aligned byte offset."
                    ) from None
            if data and not text:
                raise _error(
                    "invalid_arguments", "Read length cannot fit the next UTF-8 codepoint."
                )
            result = {
                "path": logical,
                "text": text,
                "offset": offset,
                "next_offset": offset,
                "truncated": False,
                "sha256": digest,
                "size_bytes": size,
            }
            while True:
                result["text"] = text
                end = offset + len(text.encode("utf-8"))
                result["next_offset"] = end
                result["truncated"] = end < size
                if self._fits(result):
                    if data and not text:
                        raise _error(
                            "budget_exhausted", "Tool output limit cannot fit a text page."
                        )
                    return result
                if not text:
                    return self._bounded(result)
                text = text[: max(0, len(text) - max(1, len(text) // 8))]

    def list_files(self, path: str, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        """List immediate children in directory order; pagination assumes an unchanged directory."""
        _count(offset, zero=True)
        _count(limit)
        _, actual, logical = self._resolve(path)
        entries: list[dict[str, str]] = []
        result: dict[str, Any] = {
            "path": logical,
            "entries": entries,
            "next_offset": offset,
            "truncated": False,
        }
        try:
            with (
                self._directory(actual) as descriptor,
                os.scandir(descriptor if descriptor is not None else actual) as scan,
            ):
                for index, entry in enumerate(scan):
                    self.check()
                    if index < offset:
                        continue
                    if len(entries) == limit:
                        result["truncated"] = True
                        break
                    mode = entry.stat(follow_symlinks=False).st_mode
                    kind = (
                        "directory"
                        if stat.S_ISDIR(mode)
                        else "file"
                        if stat.S_ISREG(mode)
                        else "link"
                        if stat.S_ISLNK(mode)
                        else "special"
                    )
                    entries.append({"path": logical + "/" + entry.name, "kind": kind})
                    result["next_offset"] = index + 1
                    result["truncated"] = True
                    if not self._fits(result):
                        entries.pop()
                        result["next_offset"] = index
                        if not entries:
                            raise _error(
                                "budget_exhausted",
                                "Tool output limit cannot fit a directory entry.",
                            )
                        break
                else:
                    result["truncated"] = False
        except OSError:
            raise _error(
                "file_access_denied", "Cannot list this workspace directory safely."
            ) from None
        return self._bounded(result)

    def _tree(self, directory: Path) -> Iterator[Path]:
        """Yield regular entries, never follow links while scanning protected roots or quotas."""
        pending = [directory]
        while pending:
            self.check()
            current = pending.pop()
            with (
                self._directory(current) as descriptor,
                os.scandir(descriptor if descriptor is not None else current) as scan,
            ):
                for entry in scan:
                    self.check()
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        pending.append(current / entry.name)
                    elif stat.S_ISREG(mode):
                        yield current / entry.name

    def search_text(
        self, root: str, query: str, *, glob: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """Search literal text; disclose lines skipped because they exceed max_read_bytes."""
        _count(limit)
        if not isinstance(query, str) or not query:
            raise _error("invalid_arguments", "Search requires a nonempty literal query.")
        _, actual, logical = self._resolve(root)
        matches: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "path": logical,
            "matches": matches,
            "truncated": False,
            "skipped_files": 0,
            "skipped_lines": 0,
        }
        try:
            for file in self._tree(actual):
                self.check()
                if glob is not None and not fnmatch.fnmatchcase(
                    file.relative_to(actual).as_posix(), glob
                ):
                    continue
                _, file, name = self._resolve(str(file))
                with self._open(file) as stream:
                    line_number = 0
                    while True:
                        self.check()
                        line = stream.readline(self.max_read_bytes + 1)
                        if not line:
                            break
                        line_number += 1
                        if len(line) > self.max_read_bytes:
                            result["skipped_lines"] += 1
                            result["truncated"] = True
                            while line and not line.endswith(b"\n"):
                                self.check()
                                line = stream.readline(self.max_read_bytes + 1)
                            continue
                        try:
                            text = line.decode("utf-8")
                        except UnicodeDecodeError:
                            result["skipped_files"] += 1
                            result["truncated"] = True
                            break
                        if query not in text:
                            continue
                        if len(matches) == limit:
                            result["truncated"] = True
                            return self._bounded(result)
                        matches.append(
                            {"path": name, "line": line_number, "text": text.rstrip("\r\n")}
                        )
                        if not self._fits(result):
                            matches.pop()
                            result["truncated"] = True
                            return self._bounded(result)
        except OSError:
            raise _error("file_access_denied", "Workspace changed during search.") from None
        return self._bounded(result)

    def read_media(self, path: str, *, representation: str) -> dict[str, Any]:
        self._resolve(path)
        raise _error(
            "unsupported_capability",
            "Media reads require compatible representation and token accounting, "
            "which are unavailable.",
        )

    def _protected_alias(self, actual: Path) -> None:
        if not actual.exists():
            return
        identity = _identity(actual.stat())
        for root in self._roots.values():
            if not root.writable:
                for path in self._tree(root.path):
                    if _identity(path.stat()) == identity:
                        raise _error(
                            "file_access_denied",
                            "Protected input or skill aliases cannot be written.",
                        )

    def _quota(self, root: _Root, actual: Path, size: int) -> None:
        if root.max_bytes is None:
            return
        used = 0
        for file in self._tree(root.path):
            if file != actual:
                used += file.stat().st_size
            if used + size > root.max_bytes:
                raise _error("budget_exhausted", "Generated workspace storage limit exceeded.")
        if used + size > root.max_bytes:
            raise _error("budget_exhausted", "Generated workspace storage limit exceeded.")

    def write_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Atomically create, or replace only with an explicit current full-file digest."""
        if not isinstance(content, str) or type(overwrite) is not bool:
            raise _error(
                "invalid_arguments", "Writes require UTF-8 text and an explicit overwrite flag."
            )
        if overwrite and (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            raise _error("invalid_arguments", "Overwrite requires the current full-file SHA-256.")
        try:
            data = content.encode("utf-8")
        except UnicodeError:
            raise _error("invalid_arguments", "Content must be valid UTF-8 text.") from None
        return self._install(path, iter((data,)), len(data), overwrite, expected_sha256)

    def _install(
        self,
        path: str,
        chunks: Iterator[bytes],
        size: int,
        overwrite: bool,
        expected_sha256: str | None,
    ) -> dict[str, Any]:
        root, actual, logical = self._resolve(path, write=True)
        result = self._bounded({"path": logical, "sha256": "0" * 64, "size_bytes": size})
        try:
            self._protected_alias(actual)
            previous = None
            if overwrite:
                with self._open(actual) as source:
                    previous = _version(os.fstat(source.fileno()))
                    if self._hash(source) != expected_sha256:
                        raise _error(
                            "source_changed", "Expected digest does not match current file."
                        )
            elif actual.exists() or actual.is_symlink():
                raise _error(
                    "file_exists", "File already exists; provide an expected digest to overwrite."
                )
            for enclosing in self._roots.values():
                if enclosing.writable and actual.is_relative_to(enclosing.path):
                    self._quota(enclosing, actual, size)
            with self._directory(actual.parent) as parent:
                parent_identity = _identity(actual.parent.stat())
                temporary = ".skillrun-" + uuid.uuid4().hex
                temporary_path = actual.parent / temporary
                name: str | Path = temporary if parent is not None else temporary_path
                target: str | Path = actual.name if parent is not None else actual
                descriptor = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent
                )
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        digest = hashlib.sha256()
                        written = 0
                        for chunk in chunks:
                            self.check()
                            for offset in range(0, len(chunk), 65_536):
                                self.check()
                                part = chunk[offset : offset + 65_536]
                                stream.write(part)
                                digest.update(part)
                                written += len(part)
                        if written != size:
                            raise _error("source_changed", "Source size changed during write.")
                        result["sha256"] = digest.hexdigest()
                        stream.flush()
                        os.fsync(stream.fileno())
                    self.check()
                    self._resolve(path, write=True)
                    if _identity(actual.parent.stat()) != parent_identity:
                        raise _error("source_changed", "Parent directory changed during write.")
                    if overwrite:
                        if previous != _version(actual.stat()):
                            raise _error("source_changed", "File changed before replacement.")
                        with self._open(actual) as current:
                            if self._hash(current) != expected_sha256:
                                raise _error("source_changed", "File changed before replacement.")
                        os.replace(name, target, src_dir_fd=parent, dst_dir_fd=parent)
                    else:
                        os.link(
                            name,
                            target,
                            src_dir_fd=parent,
                            dst_dir_fd=parent,
                            follow_symlinks=False,
                        )
                    if parent is not None:
                        os.fsync(parent)
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(name, dir_fd=parent)
        except OSError:
            raise _error(
                "file_write_failed", "Workspace write failed; inspect the path before retrying."
            ) from None
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        return result

    def edit_file(self, path: str, *, expected_sha256: str, old: str, new: str) -> dict[str, Any]:
        """Stream an exact, unambiguous literal replacement, including cross-chunk matches."""
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise _error("invalid_arguments", "Edit requires nonempty old and replacement text.")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise _error("invalid_arguments", "Edit requires the current full-file SHA-256.")
        try:
            before, after = old.encode("utf-8"), new.encode("utf-8")
        except UnicodeError:
            raise _error("invalid_arguments", "Edit text must be valid UTF-8.") from None
        _, actual, _ = self._resolve(path, write=True)
        position: int | None = None
        with self._open(actual) as source:
            if self._hash(source) != expected_sha256:
                raise _error("source_changed", "Expected digest does not match current file.")
            size = os.fstat(source.fileno()).st_size
            source.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8")()
            tail = b""
            consumed = 0
            while True:
                self.check()
                chunk = source.read(65_536)
                try:
                    decoder.decode(chunk, final=not chunk)
                except UnicodeError:
                    raise _error(
                        "invalid_arguments", "Only valid UTF-8 text can be edited."
                    ) from None
                if not chunk:
                    break
                window = tail + chunk
                start = 0
                while (found := window.find(before, start)) >= 0:
                    if position is not None:
                        raise _error(
                            "invalid_arguments", "Replacement matches more than one location."
                        )
                    position = consumed - len(tail) + found
                    start = found + 1
                consumed += len(chunk)
                tail = window[-(len(before) - 1) :] if len(before) > 1 else b""
        if position is None:
            raise _error("invalid_arguments", "Replacement text was not found.")
        return self._install(
            path,
            self._edited_chunks(actual, position, before, after, expected_sha256),
            size - len(before) + len(after),
            True,
            expected_sha256,
        )

    def _edited_chunks(
        self, path: Path, position: int, before: bytes, after: bytes, expected_sha256: str
    ) -> Iterator[bytes]:
        with self._open(path) as source:
            if self._hash(source) != expected_sha256:
                raise _error("source_changed", "File changed before editing.")
            source.seek(0)
            remaining = position
            while remaining:
                self.check()
                chunk = source.read(min(65_536, remaining))
                if not chunk:
                    raise _error("source_changed", "Source changed during editing.")
                remaining -= len(chunk)
                yield chunk
            source.seek(len(before), os.SEEK_CUR)
            yield after
            while True:
                self.check()
                chunk = source.read(65_536)
                if not chunk:
                    break
                yield chunk
