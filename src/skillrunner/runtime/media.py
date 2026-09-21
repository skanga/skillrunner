"""Validate an isolated image snapshot before allowing media into model context."""

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path

from skillrunner.artifacts.external import ExternalValidationResult, validate_external
from skillrunner.config.models import ExternalValidator, ModelProfile
from skillrunner.domain.errors import RunnerError
from skillrunner.model.media import IMAGE_ACCOUNTING_CONTRACTS, MediaAttachment, png_dimensions
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import ChildEnvironment
from skillrunner.runtime.processes import ProcessSupervisor
from skillrunner.runtime.storage import monitor_operation
from skillrunner.tools.files import FileTools


async def read_png(
    files: FileTools,
    path: str,
    representation: str,
    *,
    profile: ModelProfile,
    validator: ExternalValidator | None,
    supervisor: ProcessSupervisor,
    environment: ChildEnvironment,
    deadline: Deadline,
    staging: Path,
    monitor: Callable[[], None],
) -> MediaAttachment:
    files._resolve(path)
    if (
        representation != "image"
        or "image" not in profile.input_modalities
        or profile.image_accounting not in IMAGE_ACCOUNTING_CONTRACTS
        or validator is None
    ):
        raise RunnerError(
            "unsupported_capability",
            "PNG reading requires image input, a supported explicit image_accounting contract, "
            "representation=image, and an allowlisted artifacts.validators.png parser.",
        )
    logical, data = files.read_binary_snapshot(path)
    width, height = png_dimensions(data)
    digest = hashlib.sha256(data).hexdigest()
    # Only this copy is exposed to the validator, never the original user's file.
    with tempfile.TemporaryDirectory(prefix="media-", dir=staging) as temporary:
        candidate = Path(temporary) / "image.png"
        candidate.write_bytes(data)
        monitor()

        async def validate() -> ExternalValidationResult:
            return await validate_external(
                candidate,
                validator,
                supervisor=supervisor,
                environment=environment,
                deadline=deadline,
                expected_digest=digest,
                expected_size=len(data),
                writers_stopped=True,
            )

        checked = await monitor_operation(validate, monitor)
        if not checked.validation.valid:
            raise RunnerError("artifact_invalid", "Configured PNG validator rejected the image.")
    assert profile.image_accounting is not None
    return MediaAttachment(
        data=data,
        path=logical,
        width=width,
        height=height,
        image_accounting=profile.image_accounting,
    )
