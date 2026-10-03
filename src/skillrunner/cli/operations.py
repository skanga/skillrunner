"""Offline input previews and explicit, conservative bundle maintenance."""

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import typer

from skillrunner.catalog.snapshots import scan_tree
from skillrunner.cli.receipts import fail
from skillrunner.config.sources import resolve_settings
from skillrunner.domain.errors import RunnerError
from skillrunner.recording.bundle import TERMINAL_EXITS
from skillrunner.runtime.cleanup import remove_work_tree


def preview_inputs(
    inputs: list[Path], excludes: list[str], config: Path | None, json_mode: bool
) -> None:
    try:
        settings = resolve_settings(Path.cwd(), {"config": config}, dict(os.environ))
        records = []
        count = size = 0
        for source in inputs:
            entries = scan_tree(
                source,
                exclude=excludes,
                max_files=settings.storage.max_input_files - count,
                max_bytes=settings.storage.max_input_bytes - size,
            )
            files = [entry for entry in entries if not entry.directory]
            count += len(files)
            size += sum(entry.identity[3] for entry in files)
            records.append(
                {
                    "source": str(source.absolute()),
                    "files": [entry.relative_path for entry in files],
                }
            )
        result = {
            "inputs": records,
            "excludes": excludes,
            "file_count": count,
            "bytes": size,
            "note": (
                "No contents were copied or sent. No implicit ignores apply; inspect .env, .git "
                "and virtual environments before running. "
                "Use an output directory outside every input tree."
            ),
        }
    except RunnerError as error:
        fail(error, json_mode)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=None if json_mode else 2))


def _manifest(root: Path) -> dict[str, Any] | None:
    try:
        path = root / "run.json"
        if (
            root.is_symlink()
            or root.is_junction()
            or re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", root.name) is None
            or path.is_symlink()
            or path.stat().st_size > 10_485_760
        ):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("identity"), dict)
            or value["identity"].get("run_id") != root.name
            or value["identity"].get("schema_version") != "1"
            or not isinstance(value.get("lifecycle"), dict)
            or not isinstance(value["lifecycle"].get("status"), str)
        ):
            return None
        return value
    except (OSError, ValueError, RecursionError):
        return None


def _completed(root: Path, manifest: dict[str, Any]) -> bool:
    lifecycle = manifest.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        return False
    cleanup = lifecycle.get("cleanup") or {}
    status = lifecycle.get("status")
    if not isinstance(status, str) or type(lifecycle.get("exit_code")) is not int:
        return False
    return bool(
        not (root / ".active").exists()
        and not (root / ".active").is_symlink()
        and manifest["identity"].get("finished_at")
        and lifecycle.get("exit_code") in TERMINAL_EXITS.get(status, set())
        and isinstance(cleanup, dict)
        and not cleanup.get("owned_pids_remaining")
    )


def _bytes(root: Path) -> int:
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_junction()]
        for name in files:
            info = (Path(folder) / name).lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def runs_command(
    output_dir: Path | None,
    config: Path | None,
    json_mode: bool,
    *,
    clean: bool = False,
    yes: bool = False,
) -> None:
    try:
        settings = resolve_settings(
            Path.cwd(), {"config": config, "output_dir": output_dir}, dict(os.environ)
        )
        root = settings.output_dir
        runs: list[dict[str, Any]] = []
        candidates: list[str] = []
        removed: list[str] = []
        for path in sorted(root.iterdir()) if root.is_dir() else []:
            manifest = _manifest(path) if path.is_dir() else None
            if manifest is None:
                continue
            complete = _completed(path, manifest)
            runs.append(
                {
                    "run_id": path.name,
                    "path": str(path),
                    "status": manifest.get("lifecycle", {}).get("status", "unknown"),
                    "bytes": _bytes(path),
                    "cleanable": complete,
                }
            )
            if clean and complete:
                candidates.append(str(path))
                if yes:
                    before = path.stat()
                    latest = _manifest(path)
                    after = path.stat()
                    if (
                        latest != manifest
                        or not _completed(path, manifest)
                        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    ):
                        raise RunnerError(
                            "invalid_arguments",
                            "Bundle changed during cleanup; retry after its owner exits.",
                        )
                    remove_work_tree(path)
                    removed.append(str(path))
        result = {
            "runs": runs,
            "total_bytes": sum(run["bytes"] for run in runs),
            "candidates": candidates,
            "removed": removed,
            "note": (
                "Preview only; pass --yes to delete completed bundles. "
                "Active, crashed, linked and unknown directories are never candidates."
            )
            if clean and not yes
            else "Published outputs outside bundles are not removed.",
        }
    except RunnerError as error:
        fail(error, json_mode)
    except OSError:
        fail(
            RunnerError(
                "invalid_arguments",
                "Cannot inspect or clean the bundle directory; "
                "check permissions and concurrent activity.",
            ),
            json_mode,
        )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=None if json_mode else 2))
