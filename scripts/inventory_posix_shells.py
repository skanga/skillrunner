"""Inventory shell executable names in one traversal of fixed local drives."""

import json
import os
import sys
import time
from pathlib import Path

SHELL_NAMES = {"bash.exe", "sh.exe", "zsh.exe", "dash.exe", "busybox.exe"}


def inventory(roots: list[str]) -> dict:
    pending = list(roots)
    seen: set[str] = set()
    shells: list[str] = []
    inaccessible: list[dict[str, str]] = []
    last_progress = time.monotonic()
    while pending:
        directory = os.path.normcase(os.path.realpath(pending.pop()))
        if directory in seen:
            continue
        seen.add(directory)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            pending.append(entry.path)
                        elif entry.name.casefold() in SHELL_NAMES and entry.is_file():
                            shells.append(entry.path)
                    except OSError as error:
                        inaccessible.append({"path": entry.path, "error": str(error)})
        except OSError as error:
            inaccessible.append({"path": directory, "error": str(error)})
        if time.monotonic() - last_progress >= 15:
            print(
                f"Scanned {len(seen)} directories; {len(pending)} pending; "
                f"found {len(shells)} shell names",
                file=sys.stderr,
                flush=True,
            )
            last_progress = time.monotonic()
    return {
        "roots": roots,
        "directories_scanned": len(seen),
        "shells": sorted(set(shells)),
        "inaccessible": inaccessible,
    }


if __name__ == "__main__":
    destination, *roots = sys.argv[1:]
    if not roots:
        raise SystemExit("Provide fixed-drive roots to inventory")
    result = inventory(roots)
    Path(destination).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Inventory complete: {result['directories_scanned']} directories, "
        f"{len(result['shells'])} shell names, "
        f"{len(result['inaccessible'])} inaccessible paths",
        flush=True,
    )
