"""Resolve credentials only immediately before their authorized use."""

from collections.abc import Mapping

from pydantic import SecretStr

from skillrunner.config.models import ModelProfile
from skillrunner.domain.errors import RunnerError


def resolve_api_key(profile: ModelProfile, environ: Mapping[str, str]) -> SecretStr | None:
    if profile.auth_mode == "none":
        return None
    name = profile.api_key_env or "OPENAI_API_KEY"
    value = environ.get(name)
    if not value or not value.strip():
        raise RunnerError(
            "missing_credential",
            f"Set {name} for the selected model and rerun.",
            details={
                "credential_reference": name,
                "suggested_action": f"Set {name} and rerun.",
            },
        )
    return SecretStr(value)
