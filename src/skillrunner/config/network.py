"""Explicit TLS trust, independent of ambient proxy and certificate variables."""

import ssl
from pathlib import Path

from skillrunner.domain.errors import RunnerError


def tls_verify(ca_bundle: str | Path | None) -> ssl.SSLContext | bool:
    if ca_bundle is None:
        return True
    try:
        return ssl.create_default_context(cafile=str(ca_bundle))
    except (OSError, ValueError):
        raise RunnerError(
            "invalid_configuration",
            "Cannot load configured ca_bundle as a PEM trust store.",
            details={
                "suggested_action": (
                    "Check the ca_bundle file path and PEM certificate contents; "
                    "TLS verification cannot be disabled."
                )
            },
        ) from None
