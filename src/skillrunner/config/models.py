"""Strict external configuration contracts; credentials are references only."""

import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


def environment_reference(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Invalid environment reference")
    return value


EnvironmentReference = Annotated[str, AfterValidator(environment_reference)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


def duration(value: Any, *, allow_zero: bool = False) -> float:
    if not isinstance(value, str):
        raise ValueError("Duration requires explicit units")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value)
    if match is None:
        raise ValueError("Duration requires explicit units")
    seconds = float(match[1]) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError("Invalid duration")
    return seconds


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoint requires an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("Endpoint must not contain credentials, query, or fragment")
    if any(c.isspace() for c in value) or parsed.port == 0:
        raise ValueError("Malformed endpoint")
    return value


class RunLimits(StrictModel):
    timeout: float = 600
    max_steps: PositiveInt = 40
    max_tool_calls: PositiveInt = 100
    max_tokens: PositiveInt = 100_000
    model_transport_retries: NonnegativeInt = 1
    shutdown_grace: float = 5

    @field_validator("timeout", "shutdown_grace", mode="before")
    @classmethod
    def parse_duration(cls, value: Any, info: Any) -> float:
        return duration(value, allow_zero=info.field_name == "shutdown_grace")


class StorageLimits(StrictModel):
    max_input_files: NonnegativeInt = 10_000
    max_input_bytes: NonnegativeInt = 1_073_741_824
    max_package_files: NonnegativeInt = 20_000
    max_package_bytes: NonnegativeInt = 536_870_912
    max_artifact_bytes: NonnegativeInt = 2_147_483_648
    max_scratch_bytes: NonnegativeInt = 4_294_967_296
    max_tool_output_bytes: NonnegativeInt = 1_048_576
    max_event_log_bytes: NonnegativeInt = 10_485_760
    max_read_bytes: NonnegativeInt = 65_536
    max_archive_expanded_bytes: NonnegativeInt = 268_435_456


class Discovery(StrictModel):
    path: str
    context_window_field: str | None = None
    max_output_field: str | None = None
    input_modalities_field: str | None = None

    @field_validator("path")
    @classmethod
    def relative_resource(cls, value: str) -> str:
        if not value or value.startswith("/") or ":" in value or ".." in value.split("/"):
            raise ValueError("Discovery requires an endpoint-relative resource")
        return value


class DirectModel(StrictModel):
    auth_mode: Literal["bearer", "none"] = "bearer"
    api_key_env: EnvironmentReference | None = None
    context_window_tokens: PositiveInt | None = None
    max_output_tokens: PositiveInt | None = None
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    image_accounting: Literal["openai-patch-high-v1", "gemma4-image-max-v1"] | None = None
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    request_options: dict[str, Any] = Field(default_factory=dict)
    discovery: Discovery | None = None

    @field_validator("request_options")
    @classmethod
    def options(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "temperature",
            "top_p",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "reasoning_effort",
        }
        if value.keys() - allowed:
            raise ValueError("Unsupported inference option")
        return value


class ModelProfile(DirectModel):
    base_url: str
    model: Annotated[str, Field(min_length=1)]
    _endpoint = field_validator("base_url")(endpoint)


class Policy(StrictModel):
    allowed_executables: list[str] = Field(default_factory=list)
    allowed_env: list[EnvironmentReference] = Field(default_factory=list)
    command_env: dict[str, dict[EnvironmentReference, EnvironmentReference]] = Field(
        default_factory=dict
    )
    executable_identities: dict[str, tuple[int, int, int, int]] = Field(
        default_factory=dict, exclude=True
    )
    unresolved_executables: list[str] = Field(default_factory=list, exclude=True)


class MCPConfig(StrictModel):
    transport: Literal["stdio", "streamable-http"]
    allowed_tools: list[str] = Field(default_factory=list)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env: dict[EnvironmentReference, EnvironmentReference] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, EnvironmentReference] = Field(default_factory=dict)

    @model_validator(mode="after")
    def connection(self) -> "MCPConfig":
        if self.transport == "stdio":
            if not self.command or self.url or self.headers:
                raise ValueError("Stdio requires command and no HTTP fields")
        elif not self.url or self.command or self.args or self.cwd or self.env:
            raise ValueError("HTTP requires url and no process fields")
        if self.url:
            endpoint(self.url)
        return self


class ExternalValidator(StrictModel):
    command: Annotated[str, Field(min_length=1)]
    args: list[str]

    @field_validator("args")
    @classmethod
    def path_argument(cls, value: list[str]) -> list[str]:
        if sum(item.count("{path}") for item in value) != 1:
            raise ValueError("Validator args require exactly one {path} substitution")
        return value


class Artifacts(StrictModel):
    validators: dict[str, ExternalValidator] = Field(default_factory=dict)


class Acceptance(StrictModel):
    checks: dict[str, ExternalValidator] = Field(default_factory=dict)
    max_repairs: NonnegativeInt = 2


class Diagnostics(StrictModel):
    log_content: bool = False
    retain_work: bool = False
    redact_env: list[str] = Field(default_factory=list)


class FileSettings(StrictModel):
    schema_version: Literal[1] = 1
    skills_dir: str | Path = "skills"
    output_dir: str | Path = "outputs"
    default_model: str | None = None
    image_inspector: Annotated[str, Field(min_length=1)] | None = None
    models: dict[str, ModelProfile] = Field(default_factory=dict)
    direct_model: DirectModel = Field(default_factory=DirectModel)
    limits: RunLimits = Field(default_factory=RunLimits)
    storage: StorageLimits = Field(default_factory=StorageLimits)
    policy: Policy = Field(default_factory=Policy)
    mcp: dict[str, MCPConfig] = Field(default_factory=dict)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    acceptance: Acceptance = Field(default_factory=Acceptance)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @model_validator(mode="after")
    def inspector_alias(self) -> "FileSettings":
        if self.image_inspector is not None and self.image_inspector not in self.models:
            raise ValueError("image_inspector must name a configured model profile")
        return self

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("Schema version must be an integer")
        return value


class ResolvedSettings(FileSettings):
    skills_dir: Path = Path("skills")
    output_dir: Path = Path("outputs")
    config_path: Path | None = None
    selected_model: str | None = None
    base_url: str | None = None
    sources: dict[str, str] = Field(default_factory=dict)
