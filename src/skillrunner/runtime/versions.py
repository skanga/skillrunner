"""Explicit runtime probes and bounded, structured version recognition."""

import re
from pathlib import Path

# A basename selects only a probe protocol, never a claimed version.
_VERSION = r"([0-9]{1,4}\.[0-9]{1,4}(?:\.[0-9]{1,4})?(?:(?:a|b|rc)[0-9]{1,4})?)"


def runtime_probe(executable: str) -> tuple[str, tuple[str, ...], str] | None:
    name = Path(executable).name.lower().removesuffix(".exe")
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", name):
        return "python", ("--version",), rf"Python {_VERSION}"
    if name in {"node", "nodejs"}:
        return "node", ("--version",), rf"v{_VERSION}"
    if name == "bash":
        return (
            "shell",
            ("--version",),
            rf"GNU bash, version {_VERSION}(?:\([0-9]+\)-release)?(?: .*)?",
        )
    if name == "zsh":
        return "shell", ("--version",), rf"zsh {_VERSION}(?: .*)?"
    if name in {"sh", "dash", "ash", "ksh"}:
        return "shell", (), ""
    return None


def parse_version(pattern: str, stdout: bytes, stderr: bytes) -> str | None:
    """Only a recognized first line yields a numeric version; discard all else."""
    for output in (stdout, stderr):
        first = output[:1024].split(b"\n", 1)[0].decode("ascii", errors="replace").strip()
        match = re.fullmatch(pattern, first)
        if match:
            return match.group(1)
    return None
