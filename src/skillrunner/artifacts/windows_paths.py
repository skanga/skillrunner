"""Hold ordinary Windows directory components stable during publication."""

# mypy: disable-error-code="attr-defined"
# Windows-only ctypes declarations are absent from POSIX typeshed views.

import ctypes
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path


@contextmanager
def lock_directory_chain(parent: Path) -> Iterator[None]:
    """Deny directory deletion and renaming until staging cleanup has finished.

    Open from the anchor down so earlier components cannot redirect later opens.
    Reparse components are rejected rather than following an unlocked target.
    Child-file creation/replacement remains permitted inside ordinary directories.
    """
    if os.name != "nt":
        raise OSError("Windows directory locking requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handles = []
    try:
        for component in reversed((parent, *parent.parents)):
            handle = kernel.CreateFileW(
                str(component),
                0x80000000,  # GENERIC_READ participates in sharing enforcement.
                3,  # FILE_SHARE_READ | FILE_SHARE_WRITE; deny delete/rename.
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            if handle in (None, ctypes.c_void_p(-1).value):
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            info = component.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_file_attributes & 0x400:
                raise OSError("Publication requires ordinary directory components")
        yield
    finally:
        for handle in reversed(handles):
            kernel.CloseHandle(handle)
