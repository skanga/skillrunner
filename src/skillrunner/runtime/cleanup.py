"""Remove owned snapshots, including Windows read-only file attributes."""

import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any


def remove_work_tree(root: Path) -> None:
    owned = root.resolve()

    def retry(function: Callable[..., Any], path: str, error: BaseException) -> None:
        target = Path(path)
        if not isinstance(error, PermissionError) or not target.resolve().is_relative_to(owned):
            raise error
        info = target.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & stat.S_IWUSR:
            raise error
        target.chmod(info.st_mode | stat.S_IWUSR)
        function(path)

    shutil.rmtree(root, onexc=retry)
