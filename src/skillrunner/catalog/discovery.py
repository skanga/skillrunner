"""Discover metadata without executing scripts or copying inactive packages."""

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillrunner.catalog.frontmatter import parse_skill
from skillrunner.catalog.snapshots import Snapshot, file_identity, scan_tree, tree_fingerprint
from skillrunner.domain.errors import RunnerError


@dataclass
class SkillDescriptor:
    name: str
    description: str
    root: Path
    metadata: dict[str, Any]
    instructions: str
    fingerprint: str
    snapshot: Snapshot | None = None


@dataclass(frozen=True)
class RejectedPackage:
    path: Path
    code: str
    message: str


@dataclass
class Catalog:
    skills: dict[str, SkillDescriptor] = field(default_factory=dict)
    rejected: list[RejectedPackage] = field(default_factory=list)

    def require(self, name: str) -> SkillDescriptor:
        if name not in self.skills:
            raise RunnerError(
                "required_skill_unavailable",
                f"Required skill {name!r} is missing or invalid; inspect skills list.",
            )
        return self.skills[name]

    def metadata_for_model(self) -> list[dict[str, str]]:
        return [
            {"name": skill.name, "description": skill.description} for skill in self.skills.values()
        ]


def discover(
    root: Path, *, max_instruction_bytes: int = 536_870_912, check: Callable[[], None] | None = None
) -> Catalog:
    catalog = Catalog()
    boundary = root.resolve()

    def reject(path: Path, error: RunnerError) -> None:
        catalog.rejected.append(RejectedPackage(path, error.code, error.message))

    def visit(path: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        if check:
            check()
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(boundary):
                raise RunnerError("invalid_arguments", "Catalog link escapes the skills directory.")
            info = resolved.stat()
            key = (info.st_dev, info.st_ino)
            if key in ancestors:
                raise RunnerError("invalid_arguments", "Catalog contains a directory-link cycle.")
            instruction = path / "SKILL.md"
            if instruction.exists() or instruction.is_symlink():
                before = scan_tree(path, check=check)
                if not instruction.resolve().is_relative_to(resolved):
                    raise RunnerError("invalid_arguments", "SKILL.md escapes its package root.")
                expected = next(
                    (entry for entry in before if entry.relative_path == "SKILL.md"), None
                )
                if expected is None:
                    raise RunnerError("source_changed", "SKILL.md disappeared during discovery.")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                descriptor = os.open(expected.source, flags)
                with os.fdopen(descriptor, "rb") as reader:
                    opened = os.fstat(reader.fileno())
                    initial = file_identity(opened)
                    if not stat.S_ISREG(opened.st_mode) or initial != expected.identity:
                        raise RunnerError("source_changed", "SKILL.md changed before reading.")
                    content = reader.read(max_instruction_bytes + 1)
                    if file_identity(os.fstat(reader.fileno())) != initial:
                        raise RunnerError(
                            "source_changed",
                            "SKILL.md changed during discovery.",
                            status="failed",
                            exit_code=6,
                        )
                if len(content) > max_instruction_bytes:
                    raise RunnerError(
                        "budget_exhausted", "Skill instructions exceed package bounds."
                    )
                text = content.decode("utf-8")
                metadata = parse_skill(text, path.name)
                if tree_fingerprint(scan_tree(path, check=check)) != tree_fingerprint(before):
                    raise RunnerError(
                        "source_changed",
                        "Package changed during discovery.",
                        status="failed",
                        exit_code=6,
                    )
                name = metadata["name"]
                if name in catalog.skills:
                    raise RunnerError("duplicate_skill_name", f"Duplicate skill name {name!r}.")
                catalog.skills[name] = SkillDescriptor(
                    name, metadata["description"], path, metadata, text, tree_fingerprint(before)
                )
                return
            for child in sorted(path.iterdir()):
                if child.is_dir() or child.is_symlink():
                    visit(child, ancestors | {key})
        except RunnerError as exc:
            if exc.code not in {"invalid_arguments", "source_changed"}:
                raise
            reject(path, exc)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            reject(
                path,
                RunnerError("invalid_arguments", f"Cannot read package: {type(exc).__name__}."),
            )

    if not root.exists():
        return catalog
    if not root.is_dir():
        raise RunnerError("invalid_arguments", "Skills directory must be a directory.")
    visit(root, frozenset())
    return catalog
