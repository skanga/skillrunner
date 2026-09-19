"""Built-in syntax/container checks, never full document application conformance."""

import codecs
import csv
import io
import json
import lzma
import os
import stat
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from skillrunner.artifacts.zip_stream import member_bytes
from skillrunner.catalog.snapshots import file_identity
from skillrunner.domain.errors import RunnerError

MEDIA_TYPES = {
    "txt": "text/plain",
    "md": "text/markdown",
    "json": "application/json",
    "csv": "text/csv",
    "zip": "application/zip",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
OFFICE_PARTS = {
    "docx": {"[Content_Types].xml", "_rels/.rels", "word/document.xml"},
    "xlsx": {"[Content_Types].xml", "_rels/.rels", "xl/workbook.xml"},
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    validation_level: str
    media_type: str
    issues: tuple[str, ...] = ()


class _CheckpointFailure(Exception):
    def __init__(self, cause: Exception) -> None:
        self.cause = cause


def _reject_constant(value: str) -> None:
    raise ValueError("Nonstandard JSON constant")


def _validate_archive(
    reader: BinaryIO, format: str, expanded_limit: int, check: Callable[[], None]
) -> bool:
    with zipfile.ZipFile(reader) as archive:
        names: set[str] = set()
        declared_size = 0
        for member in archive.infolist():
            check()
            path = PurePosixPath(member.filename)
            if (
                member.orig_filename != member.filename
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or ":" in member.filename
                or member.filename in names
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                return False
            names.add(member.filename)
            declared_size += member.file_size
            if declared_size > expanded_limit:
                raise RunnerError("budget_exhausted", "Archive expanded-byte limit exceeded.")
        if not OFFICE_PARTS.get(format, set()).issubset(names):
            return False
        expanded = 0
        members = sorted(archive.infolist(), key=lambda member: member.header_offset)
        for index, member in enumerate(members):
            boundary = (
                members[index + 1].header_offset if index + 1 < len(members) else archive.start_dir
            )
            expanded += member_bytes(reader, member, boundary, expanded_limit - expanded, check)
        return True


def validate_builtin(
    path: Path,
    format: str,
    *,
    size_limit: int,
    archive_expanded_limit: int,
    check: Callable[[], None] | None = None,
) -> ValidationResult:
    """Validate a staged candidate after writers stop; external formats fail explicitly."""
    for limit in (size_limit, archive_expanded_limit):
        if type(limit) is not int or limit < 0:
            raise ValueError("Validation bounds must be nonnegative integers")
    format = {"markdown": "md", "text": "txt"}.get(format.lower(), format.lower())
    if format not in MEDIA_TYPES:
        raise RunnerError(
            "unsupported_capability", "Output format requires a configured external validator."
        )
    if check:
        check()

    def checkpoint() -> None:
        if check:
            try:
                check()
            except Exception as exc:
                raise _CheckpointFailure(exc) from exc

    level = "container" if format in {"zip", "docx", "xlsx"} else "parsed"
    media_type = MEDIA_TYPES[format]
    valid = False
    try:
        if path.is_symlink():
            raise RunnerError("artifact_invalid", "Candidate must be a regular staged file.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as reader:
            initial = os.fstat(reader.fileno())
            if not stat.S_ISREG(initial.st_mode):
                raise RunnerError("artifact_invalid", "Candidate must be a regular staged file.")
            if initial.st_size > size_limit:
                raise RunnerError("budget_exhausted", "Artifact byte limit exceeded.")
            if level == "container":
                valid = _validate_archive(reader, format, archive_expanded_limit, checkpoint)
            else:
                decoder = codecs.getincrementaldecoder("utf-8")()
                pieces: list[str] = []
                total = 0
                while True:
                    checkpoint()
                    chunk = reader.read(64 * 1024)
                    total += len(chunk)
                    if total > size_limit:
                        raise RunnerError("budget_exhausted", "Artifact byte limit exceeded.")
                    decoded = decoder.decode(chunk, final=not chunk)
                    if format in {"json", "csv"}:
                        pieces.append(decoded)
                    if not chunk:
                        break
                if format == "json":
                    json.loads("".join(pieces), parse_constant=_reject_constant)
                elif format == "csv":
                    for _ in csv.reader(io.StringIO("".join(pieces), newline=""), strict=True):
                        checkpoint()
                valid = True
            if file_identity(os.fstat(reader.fileno())) != file_identity(initial):
                raise RunnerError("artifact_invalid", "Candidate changed during validation.")
    except _CheckpointFailure as exc:
        raise exc.cause from None
    except RunnerError:
        raise
    except (
        OSError,
        UnicodeError,
        csv.Error,
        zipfile.BadZipFile,
        zlib.error,
        lzma.LZMAError,
        RuntimeError,
        ValueError,
    ):
        valid = False
    return ValidationResult(
        valid, level, media_type, () if valid else ("Format validation failed.",)
    )
