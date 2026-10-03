"""Conservative recognition of an explicitly named output file, not input mentions."""

import re
from pathlib import Path


def requested_output_filename(prompt: str) -> str | None:
    pattern = (
        r"(?:\b(?:write|save|publish|output|deliver|create|produce)\b"
        r"(?:\s+(?:the|a|an|final|primary|output|result|report|file|artifact|deliverable)){0,5}"
        r"\s+(?:to|as|at|named|called)\s+"
        r"|\b(?:output|deliverable|file)\s+(?:should|must|will|shall)\s+be\s+"
        r"(?:written|saved|published)\s+(?:to|as|at|in)\s+)"
        r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([^\s,;:!?)}\]]+))"
    )
    for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
        raw = next(value for value in match.groups() if value is not None).strip()
        if match.group(4) is not None:
            raw = raw.rstrip(".")
        name = re.split(r"[/\\]", raw)[-1]
        if (
            name not in {"", ".", ".."}
            and not re.search(r'[<>:"|?*\x00]', name)
            and (Path(name).suffix or raw != name or match.group(4) is None)
        ):
            return name
    return None
