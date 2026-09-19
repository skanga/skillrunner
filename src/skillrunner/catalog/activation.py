"""Activate bounded immutable-by-contract copies, once per skill per run."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from skillrunner.catalog.discovery import Catalog, SkillDescriptor
from skillrunner.catalog.snapshots import snapshot_tree
from skillrunner.runtime.cleanup import remove_work_tree


class ActivationService:
    def __init__(
        self,
        catalog: Catalog,
        workspace: Path,
        *,
        max_files: int,
        max_bytes: int,
        check: Callable[[], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.workspace = workspace
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.check = check
        self.active: dict[str, SkillDescriptor] = {}
        self.records: list[dict[str, str | int]] = []
        self.total_files = 0
        self.total_bytes = 0

    def activate(
        self, name: str, *, reason: str, admit_instructions: Callable[[str], None] | None = None
    ) -> SkillDescriptor:
        if name in self.active:
            return self.active[name]
        descriptor = self.catalog.require(name)
        snapshot = snapshot_tree(
            descriptor.root,
            self.workspace / name,
            max_files=self.max_files - self.total_files,
            max_bytes=self.max_bytes - self.total_bytes,
            check=self.check,
            expected_fingerprint=descriptor.fingerprint,
        )
        try:
            instructions = (snapshot.root / "SKILL.md").read_text(encoding="utf-8")
            if admit_instructions:
                admit_instructions(instructions)
        except BaseException:
            remove_work_tree(snapshot.root)
            raise
        descriptor = replace(descriptor, snapshot=snapshot, instructions=instructions)
        self.active[name] = descriptor
        self.total_files += snapshot.file_count
        self.total_bytes += snapshot.total_bytes
        self.records.append(
            {
                "name": name,
                "digest": snapshot.digest,
                "order": len(self.records) + 1,
                "reason": reason,
            }
        )
        return descriptor
