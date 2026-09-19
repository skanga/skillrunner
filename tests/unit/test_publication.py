import hashlib
import importlib
import os
import shutil
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.artifacts.publication") is not None, (
        "Atomic primary-output publication is not implemented"
    )
    return importlib.import_module("skillrunner.artifacts.publication")


def candidate(tmp_path):
    path = tmp_path / "validated"
    path.write_bytes(b"validated output")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_verified_bytes_into_new_parent(tmp_path):
    source, digest = candidate(tmp_path)
    destination = tmp_path / "new/result.txt"
    target = api().PublicationTarget.prepare(destination, overwrite=False)
    published = target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert destination.read_bytes() == b"validated output"
    assert source.read_bytes() == destination.read_bytes()
    assert published.path == destination
    assert published.digest == digest
    destination.parent.rename(tmp_path / "renamed-output")


def test_existing_destination_refused_without_overwrite(tmp_path):
    destination = tmp_path / "result"
    destination.write_text("keep")
    with pytest.raises(ValueError, match="publication_failed"):
        api().PublicationTarget.prepare(destination, overwrite=False)
    assert destination.read_text() == "keep"


def test_overwrite_replaces_only_authorized_unchanged_destination(tmp_path):
    source, digest = candidate(tmp_path)
    destination = tmp_path / "result"
    destination.write_text("old")
    target = api().PublicationTarget.prepare(destination, overwrite=True)
    target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert destination.read_bytes() == b"validated output"


def test_destination_changed_since_preflight_is_preserved(tmp_path):
    source, digest = candidate(tmp_path)
    destination = tmp_path / "result"
    destination.write_text("old")
    target = api().PublicationTarget.prepare(destination, overwrite=True)
    destination.write_text("another writer's content")
    with pytest.raises(ValueError, match="publication_failed"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert destination.read_text() == "another writer's content"


def test_no_clobber_install_wins_race_safely(tmp_path, monkeypatch):
    module = api()
    source, digest = candidate(tmp_path)
    destination = tmp_path / "result"
    target = module.PublicationTarget.prepare(destination, overwrite=False)
    original = module.os.link

    def raced(*args, **kwargs):
        destination.write_text("competing writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.os, "link", raced)
    with pytest.raises(ValueError, match="publication_failed"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert destination.read_text() == "competing writer"
    assert not list(tmp_path.glob(".result-*"))


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_protected_sources_cannot_be_publication_targets(tmp_path, alias):
    protected = tmp_path / "input"
    protected.write_text("protected")
    destination = tmp_path / "result"
    if alias == "direct":
        destination = protected
    elif alias == "symlink":
        destination.symlink_to(protected)
    else:
        os.link(protected, destination)
    with pytest.raises(ValueError, match="publication_failed"):
        api().PublicationTarget.prepare(destination, overwrite=True, protected_roots=[protected])
    assert protected.read_text() == "protected"


def test_candidate_digest_mismatch_never_commits(tmp_path):
    source, digest = candidate(tmp_path)
    destination = tmp_path / "result"
    target = api().PublicationTarget.prepare(destination, overwrite=False)
    source.write_text("changed after validation")
    with pytest.raises(ValueError, match="artifact_invalid"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert not destination.exists()
    assert not list(tmp_path.glob(".result-*"))


def test_cancellation_preserves_existing_output(tmp_path):
    source, digest = candidate(tmp_path)
    destination = tmp_path / "result"
    destination.write_text("keep")
    target = api().PublicationTarget.prepare(destination, overwrite=True)

    def cancel():
        raise ValueError("cancelled")

    with pytest.raises(ValueError, match="cancelled"):
        target.publish(
            source, expected_digest=digest, expected_size=16, max_bytes=1000, check=cancel
        )
    assert destination.read_text() == "keep"


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory descriptors")
def test_parent_swap_cannot_redirect_install_into_protected_directory(tmp_path, monkeypatch):
    module = api()
    source, digest = candidate(tmp_path)
    parent = tmp_path / "output"
    parent.mkdir()
    protected = tmp_path / "inputs"
    protected.mkdir()
    (protected / "result").write_text("protected")
    target = module.PublicationTarget.prepare(parent / "result", protected_roots=[protected])
    original = module.os.link

    def raced(*args, **kwargs):
        parent.rename(tmp_path / "old-output")
        parent.symlink_to(protected, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.os, "link", raced)
    with pytest.raises(ValueError, match="Output parent changed"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert (protected / "result").read_text() == "protected"
    assert not list((tmp_path / "old-output").glob(".result-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows publication path locking")
@pytest.mark.parametrize("swap_ancestor", [False, True])
def test_windows_parent_swap_cannot_overwrite_protected_file(tmp_path, monkeypatch, swap_ancestor):
    module = api()
    source, digest = candidate(tmp_path)
    ancestor = tmp_path / "ancestor"
    parent = ancestor / "output"
    parent.mkdir(parents=True)
    destination = parent / "result"
    destination.write_text("old output")
    protected = tmp_path / "protected"
    protected_parent = protected / "output" if swap_ancestor else protected
    protected_parent.mkdir(parents=True)
    protected_file = protected_parent / "result"
    protected_file.write_text("protected input")
    target = module.PublicationTarget.prepare(
        destination, overwrite=True, protected_roots=[protected]
    )
    original = module.os.replace
    attempted = []

    def raced(staging, output, **kwargs):
        attempted.append(True)
        pivot = ancestor if swap_ancestor else parent
        moved = tmp_path / "moved"
        pivot.rename(moved)
        staged_path = moved / Path(staging).relative_to(pivot)
        shutil.copyfile(staged_path, protected_parent / Path(staging).name)
        pivot.symlink_to(protected, target_is_directory=True)
        return original(staging, output, **kwargs)

    monkeypatch.setattr(module.os, "replace", raced)
    with pytest.raises(ValueError, match="publication_failed"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert attempted
    assert protected_file.read_text() == "protected input"
    assert destination.read_text() == "old output"
    assert not list(parent.glob(".result-*"))
    ancestor.rename(tmp_path / "released-after-failure")


@pytest.mark.skipif(os.name != "nt", reason="Windows directory reparse handling")
def test_windows_directory_guard_rejects_reparse_and_releases_ancestors(tmp_path):
    from skillrunner.artifacts.windows_paths import lock_directory_chain

    parent = tmp_path / "parent"
    parent.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    link = parent / "link"
    link.symlink_to(protected, target_is_directory=True)
    with (
        pytest.raises(OSError, match="ordinary directory components"),
        lock_directory_chain(link),
    ):
        pytest.fail("Reparse directory must not be admitted")
    parent.rename(tmp_path / "released-after-acquisition-failure")
    assert not list(protected.iterdir())


def test_staging_cleanup_failure_preserves_original_stop(tmp_path, monkeypatch):
    module = api()
    from skillrunner.domain.errors import RunnerError

    source, digest = candidate(tmp_path)
    target = module.PublicationTarget.prepare(tmp_path / "result")
    count = 0

    def stop():
        nonlocal count
        count += 1
        if count == 2:
            raise RunnerError("budget_exhausted", "Time limit reached.")

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(module.os, "unlink", denied)
    with pytest.raises(RunnerError) as failure:
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000, check=stop)
    assert failure.value.code == "budget_exhausted"
    assert failure.value.details["cleanup_errors"]
    assert not target.path.exists()


def test_postcommit_cleanup_failure_reports_committed_output(tmp_path, monkeypatch):
    module = api()
    from skillrunner.domain.errors import RunnerError

    source, digest = candidate(tmp_path)
    target = module.PublicationTarget.prepare(tmp_path / "result")

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(module.os, "unlink", denied)
    with pytest.raises(RunnerError) as failure:
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert failure.value.details["committed"] is True
    assert target.path.read_bytes() == b"validated output"


def test_cleanup_error_is_not_attached_to_callers_handled_exception(tmp_path, monkeypatch):
    module = api()
    from skillrunner.domain.errors import RunnerError

    source, digest = candidate(tmp_path)
    target = module.PublicationTarget.prepare(tmp_path / "result")

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(module.os, "unlink", denied)
    try:
        raise ValueError("earlier caller error")
    except ValueError as earlier:
        with pytest.raises(RunnerError) as failure:
            target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
        assert failure.value.details["committed"] is True
        assert not getattr(earlier, "__notes__", [])


@pytest.mark.skipif(os.name != "posix", reason="POSIX exclusive staging")
def test_staging_name_collision_does_not_delete_unowned_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = api()
    source, digest = candidate(tmp_path)
    target = module.PublicationTarget.prepare(tmp_path / "result")
    other = tmp_path / ".result-fixed"
    other.write_text("another writer")
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    with pytest.raises(ValueError, match="publication_failed"):
        target.publish(source, expected_digest=digest, expected_size=16, max_bytes=1000)
    assert other.read_text() == "another writer"
