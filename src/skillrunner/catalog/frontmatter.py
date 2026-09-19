"""Parse standard SKILL.md metadata without executable YAML constructors."""

import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from skillrunner.domain.errors import RunnerError


class SkillMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=1024)
    license: str | None = None
    compatibility: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")

    @field_validator("name")
    @classmethod
    def name_format(cls, value: str) -> str:
        value = unicodedata.normalize("NFKC", value.strip())
        if (
            not value
            or len(value) > 64
            or value != value.lower()
            or value.startswith("-")
            or value.endswith("-")
            or "--" in value
        ):
            raise ValueError("Invalid skill name")
        if any(not (char.isalnum() or char == "-") for char in value):
            raise ValueError("Invalid skill name")
        return value

    @field_validator("description")
    @classmethod
    def nonblank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Description must not be blank")
        return value


def parse_skill(text: str, directory_name: str) -> dict[str, Any]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise RunnerError("invalid_arguments", "SKILL.md must start with YAML frontmatter.")
    end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        raise RunnerError("invalid_arguments", "SKILL.md frontmatter is not terminated.")
    yaml = YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    try:
        raw = yaml.load("".join(lines[1:end]))
        model = SkillMetadata.model_validate(raw)
        if model.name != unicodedata.normalize("NFKC", directory_name):
            raise RunnerError("invalid_arguments", "Skill name must match its directory name.")
        return model.model_dump(by_alias=True, exclude_none=True)
    except (YAMLError, ValidationError, RecursionError, TypeError) as exc:
        raise RunnerError("invalid_arguments", "Invalid SKILL.md metadata or unsafe YAML.") from exc
