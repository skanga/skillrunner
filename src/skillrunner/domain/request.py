"""Validated task invocation independent of CLI presentation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

FORMAT_ALIASES = {"markdown": "md", "text": "txt"}
KNOWN_FORMATS = {
    "md",
    "txt",
    "json",
    "csv",
    "zip",
    "docx",
    "xlsx",
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "svg",
}


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    prompt: str
    invocation_directory: Path
    prompt_source: Literal["argument", "file", "stdin"] = "argument"
    inputs: list[Path] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    output: Path | None = None
    format: str | None = None
    overwrite: bool = False
    _output_is_directory: bool = PrivateAttr(default=False)

    @property
    def output_is_directory(self) -> bool:
        return self._output_is_directory

    @field_validator("prompt")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value.encode("utf-8", errors="strict")
        if not value.strip():
            raise ValueError("Prompt must contain non-whitespace text.")
        return value

    @model_validator(mode="after")
    def paths_and_format(self) -> "RunRequest":
        self.invocation_directory = self.invocation_directory.resolve()
        self.inputs = [(self.invocation_directory / path).absolute() for path in self.inputs]
        if self.output is not None:
            self.output = (self.invocation_directory / self.output).absolute()
            self._output_is_directory = self.output.is_dir()
        suffix = (
            self.output.suffix.lstrip(".").lower()
            if self.output and not self.output_is_directory
            else ""
        )
        suffix = FORMAT_ALIASES.get(suffix, suffix)
        if self.format is not None:
            self.format = FORMAT_ALIASES.get(self.format.lower(), self.format.lower())
            if not self.format or not self.format.isalnum():
                raise ValueError("Format must be a simple format name.")
            if suffix in KNOWN_FORMATS and suffix != self.format:
                raise ValueError("Output extension and declared format conflict.")
        else:
            if suffix and suffix not in KNOWN_FORMATS:
                raise ValueError("Unknown output extension requires an explicit format.")
            self.format = suffix or "md"
        return self
