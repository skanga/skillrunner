import hashlib
import importlib
import os
from dataclasses import replace

import pytest

from skillrunner.domain.errors import RunnerError


def setup_registry(tmp_path, max_bytes=100, check=None):

    assert importlib.util.find_spec("skillrunner.artifacts.registry") is not None, (
        "Artifact registry is not implemented"
    )
    module = importlib.import_module("skillrunner.artifacts.registry")
    paths = {}
    for name in ("generated", "staging", "artifacts", "inputs", "skills"):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        paths[name] = path
    registry = module.ArtifactRegistry(
        generated_roots=[paths["generated"]],
        staging_root=paths["staging"],
        artifacts_root=paths["artifacts"],
        protected_roots=[paths["inputs"], paths["skills"]],
        max_bytes=max_bytes,
        check=check,
    )
    return registry, paths


def record(registry, paths, content="content", role="primary", name="out.txt", format="txt"):
    source = paths["generated"] / name
    source.write_text(content)
    return registry.register(source, format=format, role=role, description="Result")


def test_freeze_requires_stopped_writers(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    with pytest.raises(RunnerError):
        registry.freeze(item, writers_stopped=False)
    assert list(paths["staging"].iterdir()) == []


def test_freeze_and_retain_are_independent_immutable_copies(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    frozen = registry.freeze(item, writers_stopped=True)
    assert frozen.size == 7 and frozen.digest == hashlib.sha256(b"content").hexdigest()
    item.path.write_text("changed source")
    assert frozen.path.read_text() == "content"
    retained = registry.retain(frozen)
    assert retained.path.parent == paths["artifacts"]
    assert retained.status == "unvalidated"
    assert retained.path.stat().st_ino != frozen.path.stat().st_ino
    assert retained.path.read_text() == "content"


def test_incomplete_retention_stays_explicit(tmp_path):
    registry, paths = setup_registry(tmp_path)
    retained = registry.retain(
        registry.freeze(record(registry, paths), writers_stopped=True), incomplete=True
    )
    assert retained.status == "incomplete"
    assert retained.path.is_file()


def test_primary_selection_requires_resolution_for_multiple(tmp_path):
    registry, paths = setup_registry(tmp_path)
    assert registry.select_primary() is None
    first = record(registry, paths)
    assert registry.select_primary() is first
    record(registry, paths, name="other")
    with pytest.raises(RunnerError) as exc:
        registry.select_primary()
    assert exc.value.status == "needs_input"


def test_secondary_does_not_change_primary_selection(tmp_path):
    registry, paths = setup_registry(tmp_path)
    first = record(registry, paths)
    record(registry, paths, role="secondary", name="secondary")
    assert registry.select_primary() is first


@pytest.mark.parametrize("kind", ["outside", "input", "skill", "symlink", "hardlink", "fifo"])
def test_registration_rejects_non_generated_or_protected_sources(tmp_path, kind):
    registry, paths = setup_registry(tmp_path)
    protected = paths["inputs"] / "secret"
    protected.write_text("secret")
    source = paths["generated"] / "bad"
    if kind == "outside":
        source = tmp_path / "outside"
        source.write_text("outside")
    elif kind == "input":
        source = protected
    elif kind == "skill":
        source = paths["skills"] / "SKILL.md"
        source.write_text("instructions")
    elif kind == "symlink":
        source.symlink_to(protected)
    elif kind == "hardlink":
        os.link(protected, source)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is unavailable on this platform")
        os.mkfifo(source)
    with pytest.raises(RunnerError):
        registry.register(source, format="txt", role="primary", description="Bad")


def test_sources_revalidated_when_freezing(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    protected = paths["inputs"] / "secret"
    protected.write_text("protected")
    item.path.unlink()
    os.link(protected, item.path)
    with pytest.raises(RunnerError):
        registry.freeze(item, writers_stopped=True)
    assert not list(paths["staging"].iterdir())


def test_untrusted_format_never_becomes_a_path(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths, format="../../escaped")
    retained = registry.retain(registry.freeze(item, writers_stopped=True))
    assert retained.path.parent == paths["artifacts"]
    assert not (tmp_path / "escaped").exists()


def test_forged_and_foreign_records_are_rejected(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    with pytest.raises(RunnerError):
        registry.freeze(replace(item, description="Forged"), writers_stopped=True)
    frozen = registry.freeze(item, writers_stopped=True)
    with pytest.raises(RunnerError):
        registry.retain(replace(frozen, digest="0" * 64))


def test_changed_frozen_bytes_cannot_be_retained(tmp_path):
    registry, paths = setup_registry(tmp_path)
    frozen = registry.freeze(record(registry, paths), writers_stopped=True)
    frozen.path.chmod(0o600)
    frozen.path.write_text("changed")
    with pytest.raises(RunnerError, match="artifact_invalid"):
        registry.retain(frozen)
    assert not list(paths["artifacts"].iterdir())


def test_total_retained_and_staging_quota(tmp_path):
    registry, paths = setup_registry(tmp_path, max_bytes=10)
    first = registry.freeze(record(registry, paths, content="123456"), writers_stopped=True)
    registry.retain(first)
    item = record(registry, paths, content="abcdef", name="second")
    with pytest.raises(RunnerError, match="budget_exhausted"):
        registry.freeze(item, writers_stopped=True)
    first.path.chmod(0o600)
    first.path.unlink()
    second = registry.freeze(item, writers_stopped=True)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        registry.retain(second)
    assert len(list(paths["artifacts"].iterdir())) == 1


def test_cancellation_removes_partial_copy(tmp_path):
    registry, paths = setup_registry(tmp_path, max_bytes=200_000)
    item = record(registry, paths, content="x" * 100_000)

    def check():
        if list(paths["staging"].iterdir()):
            raise RunnerError("cancelled", "Cancelled")

    registry.check = check
    with pytest.raises(RunnerError, match="cancelled"):
        registry.freeze(item, writers_stopped=True)
    assert not list(paths["staging"].iterdir())


def test_source_change_during_freeze_is_rejected(tmp_path):
    registry, paths = setup_registry(tmp_path, max_bytes=200_000)
    item = record(registry, paths, content="x" * 100_000)
    changed = False

    def check():
        nonlocal changed
        if list(paths["staging"].iterdir()) and not changed:
            changed = True
            item.path.write_text("y" * 100_000)

    registry.check = check
    with pytest.raises(RunnerError):
        registry.freeze(item, writers_stopped=True)
    assert not list(paths["staging"].iterdir())


def test_generated_file_already_in_artifact_root_can_be_registered(tmp_path):
    _, paths = setup_registry(tmp_path)
    from skillrunner.artifacts.registry import ArtifactRegistry

    registry = ArtifactRegistry(
        generated_roots=[paths["artifacts"]],
        staging_root=paths["staging"],
        artifacts_root=paths["artifacts"],
        max_bytes=100,
    )
    source = paths["artifacts"] / "model-result.txt"
    source.write_text("generated")
    item = registry.register(source, format="txt", role="primary", description="Result")
    retained = registry.retain(registry.freeze(item, writers_stopped=True))
    assert retained.path != source and retained.path.read_text() == "generated"


def test_copy_destinations_cannot_overlap_protected_roots(tmp_path):
    _, paths = setup_registry(tmp_path)
    from skillrunner.artifacts.registry import ArtifactRegistry

    with pytest.raises(RunnerError):
        ArtifactRegistry(
            generated_roots=[paths["generated"]],
            staging_root=paths["inputs"],
            artifacts_root=paths["artifacts"],
            protected_roots=[paths["inputs"]],
            max_bytes=100,
        )


def test_cancellation_respects_destination_ownership(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    competing = paths["staging"] / item.id

    def check():
        if competing.exists():
            if os.name == "nt":
                # Windows forbids renaming the open writer; cancellation cleans its own file.
                with pytest.raises(PermissionError):
                    competing.rename(paths["staging"] / "moved-original")
                raise RunnerError("cancelled", "Cancelled")
            competing.rename(paths["staging"] / "moved-original")
            competing.write_text("competitor")
            raise RunnerError("cancelled", "Cancelled")

    registry.check = check
    with pytest.raises(RunnerError, match="cancelled"):
        registry.freeze(item, writers_stopped=True)
    if os.name == "nt":
        assert not competing.exists()
        assert not (paths["staging"] / "moved-original").exists()
    else:
        assert competing.read_text() == "competitor"


def test_repeat_retention_cannot_promote_incomplete_artifact(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    frozen = registry.freeze(item, writers_stopped=True)
    assert registry.freeze(item, writers_stopped=True) is frozen
    retained = registry.retain(frozen, incomplete=True)
    assert registry.retain(frozen).status == "incomplete"
    assert len(registry.retained) == 1 and registry.retained[0].path == retained.path


def test_zero_byte_artifact_allowed_by_zero_quota(tmp_path):
    registry, paths = setup_registry(tmp_path, max_bytes=0)
    retained = registry.retain(
        registry.freeze(record(registry, paths, content=""), writers_stopped=True)
    )
    assert retained.size == 0 and retained.path.read_bytes() == b""


def test_destination_root_replacement_during_copy_is_rejected(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    changed = False

    def check():
        nonlocal changed
        if list(paths["staging"].iterdir()) and not changed:
            changed = True
            paths["staging"].rename(tmp_path / "old-staging")
            paths["staging"].mkdir()

    registry.check = check
    with pytest.raises(RunnerError, match="artifact_invalid"):
        registry.freeze(item, writers_stopped=True)
    assert not list(paths["staging"].iterdir())


def test_unrelated_host_hardlink_is_not_a_generated_artifact(tmp_path):
    registry, paths = setup_registry(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("unrelated")
    candidate = paths["generated"] / "linked"
    os.link(outside, candidate)
    with pytest.raises(RunnerError, match="artifact_invalid"):
        registry.register(candidate, format="txt", role="primary", description="Linked")
    assert outside.read_text() == "unrelated"


def test_unrelated_hardlink_added_after_registration_blocks_freeze(tmp_path):
    registry, paths = setup_registry(tmp_path)
    item = record(registry, paths)
    os.link(item.path, tmp_path / "outside-link")
    with pytest.raises(RunnerError, match="artifact_invalid"):
        registry.freeze(item, writers_stopped=True)
    assert not list(paths["staging"].iterdir())
