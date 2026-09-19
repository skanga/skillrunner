"""Minimal child environments; callers supply only policy-approved references.

The baseline carries PATH, temporary-directory and locale settings. Windows also
needs SYSTEMROOT, WINDIR, COMSPEC and PATHEXT. HOME, language module paths, loader
overrides and application credentials require explicit command/server mappings.
This limits accidental inheritance; it does not contain host processes.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from skillrunner.config.models import environment_reference
from skillrunner.domain.errors import RunnerError

BASELINE = frozenset({"PATH", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE"})
WINDOWS_BASELINE = frozenset({"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"})


@dataclass(repr=False)
class ChildEnvironment:
    values: dict[str, str] = field(repr=False)
    references: dict[str, str]

    def __repr__(self) -> str:
        return f"ChildEnvironment(names={sorted(self.values)!r})"


def build_child_environment(
    environ: Mapping[str, str],
    *,
    references: Mapping[str, str],
    platform: str = os.name,
) -> ChildEnvironment:
    """Resolve an already-authorized mapping, without reading global environment."""
    windows = platform == "nt"
    baseline = BASELINE | WINDOWS_BASELINE if windows else BASELINE
    source = {(name.upper() if windows else name): value for name, value in environ.items()}
    values = {name: value for name, value in source.items() if name in baseline}
    resolved: dict[str, str] = {}
    for target, reference in references.items():
        try:
            environment_reference(target)
            environment_reference(reference)
        except ValueError as exc:
            raise RunnerError(
                "invalid_configuration", "Invalid child environment reference."
            ) from exc
        destination = target.upper() if windows else target
        lookup = reference.upper() if windows else reference
        if destination in resolved:
            raise RunnerError("invalid_configuration", "Duplicate child environment variable.")
        if lookup not in source:
            raise RunnerError(
                "missing_credential", f"Required environment reference {reference} is missing."
            )
        values[destination] = source[lookup]
        resolved[destination] = reference
    if any("\x00" in value for value in values.values()):
        raise RunnerError("invalid_configuration", "Child environment contains an invalid value.")
    return ChildEnvironment(values, resolved)
