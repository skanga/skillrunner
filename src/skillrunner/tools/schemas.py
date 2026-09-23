"""Strict argument contracts for the approved built-in tools."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

Nonempty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ToolArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ActivateSkillArgs(ToolArgs):
    name: Nonempty
    reason: Nonempty


class ListFilesArgs(ToolArgs):
    path: Nonempty
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, gt=0)


class ReadTextArgs(ToolArgs):
    path: Nonempty
    offset: int = Field(default=0, ge=0)
    length: int | None = Field(default=None, gt=0)


class SearchTextArgs(ToolArgs):
    root: Nonempty
    query: Nonempty
    glob: str | None = None
    limit: int = Field(default=100, gt=0)


class ReadMediaArgs(ToolArgs):
    path: Nonempty
    representation: Nonempty


class InspectImageArgs(ToolArgs):
    path: Nonempty
    question: Nonempty


class WriteFileArgs(ToolArgs):
    path: Nonempty
    content: str
    overwrite: bool = Field(
        default=False, description="False for new files; true requires the current expected_sha256."
    )
    expected_sha256: Digest | None = Field(
        default=None,
        description=(
            "Null for a new file. For replacement, use the 64-character lowercase "
            "SHA-256 from read_text."
        ),
    )

    @model_validator(mode="after")
    def require_overwrite_digest(self) -> "WriteFileArgs":
        if self.overwrite and self.expected_sha256 is None:
            raise PydanticCustomError(
                "overwrite_digest_required", "Overwrite requires an expected digest."
            )
        return self


class EditFileArgs(ToolArgs):
    path: Nonempty
    expected_sha256: Digest
    old: Nonempty
    new: str


class RunCommandArgs(ToolArgs):
    executable: Nonempty
    argv: list[str] = Field(default_factory=list)
    cwd: Nonempty | None = Field(
        default=None,
        description=(
            "Absolute generated-root directory for output work; omitted means the skill package."
        ),
    )
    env_refs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional child environment references: each key is a child variable name "
            "such as PATH, and each value is its configured host reference name. "
            "Omit env_refs to use the executable's configured environment. "
            "Executable paths are not variable names."
        ),
    )
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class RegisterArtifactArgs(ToolArgs):
    path: Nonempty
    format: Nonempty
    role: Literal["primary", "secondary"]
    description: str


class FinishRunArgs(ToolArgs):
    outcome: Literal["succeeded", "no_matching_skill", "needs_input", "blocked", "failed"]
    report: str
    primary_artifact_id: str | None = Field(
        default=None,
        description="ID returned by register_artifact, or null for a text-only report. "
        "Never put an external page or record ID here; include it in report instead.",
    )
    secondary_ids: list[str] = Field(
        default_factory=list,
        description="Only IDs returned by register_artifact for secondary generated files.",
    )
    missing_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class MCPArgs(BaseModel):
    """Only JSON object shape is local; the remote server owns schema validation."""

    model_config = ConfigDict(extra="allow", strict=True, hide_input_in_errors=True)
