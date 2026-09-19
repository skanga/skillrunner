import importlib
import os

import pytest


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_identity_uses_consistent_windows_creation_time_and_posix_change_time(
    monkeypatch, platform
):
    from types import SimpleNamespace

    from skillrunner.catalog import snapshots

    monkeypatch.setattr(snapshots, "os", SimpleNamespace(name=platform))
    fields = dict(
        st_dev=1, st_ino=2, st_mode=0o100644, st_size=5, st_mtime_ns=200, st_birthtime_ns=100
    )
    path = SimpleNamespace(**fields, st_ctime_ns=100)
    opened = SimpleNamespace(**fields, st_ctime_ns=200)
    assert (snapshots.file_identity(path) == snapshots.file_identity(opened)) is (platform == "nt")
    changed = SimpleNamespace(**{**fields, "st_mtime_ns": 201}, st_ctime_ns=200)
    assert snapshots.file_identity(path) != snapshots.file_identity(changed)


def snapshot_api():
    assert importlib.util.find_spec("skillrunner.catalog.snapshots") is not None, (
        "Stable snapshot copying has not been implemented"
    )
    return importlib.import_module("skillrunner.catalog.snapshots")


def test_snapshot_preserves_nested_files_and_records_digests(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    (source / "extra/resources").mkdir(parents=True)
    (source / "extra/resources/a.txt").write_text("original")
    snapshot = api.snapshot_tree(source, tmp_path / "copy", max_files=10, max_bytes=100)
    assert snapshot.file_count == 1
    assert snapshot.total_bytes == 8
    assert snapshot.files[0].relative_path == "extra/resources/a.txt"
    assert len(snapshot.digest) == 64
    (source / "extra/resources/a.txt").write_text("changed!")
    assert (snapshot.root / "extra/resources/a.txt").read_text() == "original"


def test_single_file_input_copied(tmp_path):
    api = snapshot_api()
    source = tmp_path / "notes.txt"
    source.write_text("notes")
    result = api.snapshot_tree(source, tmp_path / "copy", max_files=1, max_bytes=5)
    assert (result.root / "notes.txt").read_text() == "notes"


@pytest.mark.parametrize("max_files,max_bytes", [(0, 100), (100, 4)])
def test_limit_failure_leaves_no_partial_snapshot(tmp_path, max_files, max_bytes):
    api = snapshot_api()
    source = tmp_path / "notes.txt"
    source.write_text("notes")
    with pytest.raises(ValueError, match="budget_exhausted"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=max_files, max_bytes=max_bytes)
    assert not (tmp_path / "copy").exists()


def test_internal_symlink_is_materialized(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    (source / "original").write_text("hello")
    (source / "alias").symlink_to("original")
    result = api.snapshot_tree(source, tmp_path / "copy", max_files=2, max_bytes=10)
    assert result.file_count == 2
    assert result.total_bytes == 10
    assert (result.root / "alias").read_text() == "hello"
    assert not (result.root / "alias").is_symlink()
    assert any(entry.materialized_link for entry in result.files)


def test_external_symlink_rejected(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "secret").write_text("outside")
    (source / "escape").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="invalid_arguments"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=10, max_bytes=100)
    assert not (tmp_path / "copy").exists()


def test_symlink_cycle_rejected(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    (source / "cycle").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="invalid_arguments"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=10, max_bytes=100)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special file")
def test_fifo_is_rejected_without_opening(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "fifo")
    with pytest.raises(ValueError, match="invalid_arguments"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=10, max_bytes=100)


def test_changed_source_detected_during_copy(tmp_path):
    api = snapshot_api()
    source = tmp_path / "notes.txt"
    source.write_bytes(b"A" * 300_000)
    checks = [0]

    def check():
        checks[0] += 1
        if checks[0] == 3:
            source.write_bytes(b"B" * 300_001)

    with pytest.raises(ValueError, match="source_changed"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=1, max_bytes=1_000_000, check=check)
    assert not (tmp_path / "copy").exists()


@pytest.mark.parametrize("interruption", ["cancelled", "directory_change"])
def test_interrupted_snapshot_removes_readonly_copies_and_preserves_source(tmp_path, interruption):
    from skillrunner.domain.errors import RunnerError

    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    readonly = source / "a-readonly.txt"
    readonly.write_bytes(b"original readonly input")
    readonly.chmod(0o444)
    original_mode = readonly.stat().st_mode
    large = source / "b-large.txt"
    large.write_bytes(b"x" * 200_000)
    destination = tmp_path / "copy"
    interrupted = False

    def check():
        nonlocal interrupted
        partial = destination / large.name
        if interrupted or not partial.exists() or partial.stat().st_size == 0:
            return
        interrupted = True
        assert (destination / readonly.name).read_bytes() == readonly.read_bytes()
        if interruption == "cancelled":
            raise RunnerError("cancelled", "Snapshot cancelled by the caller.")
        (source / "new-file.txt").write_bytes(b"concurrent addition")

    try:
        with pytest.raises(RunnerError) as raised:
            api.snapshot_tree(source, destination, max_files=3, max_bytes=1_000_000, check=check)
        assert raised.value.code == (
            "cancelled" if interruption == "cancelled" else "source_changed"
        )
        assert interrupted
        assert not destination.exists()
        assert readonly.read_bytes() == b"original readonly input"
        assert readonly.stat().st_mode == original_mode
        assert large.read_bytes() == b"x" * 200_000
        if interruption == "directory_change":
            assert (source / "new-file.txt").read_bytes() == b"concurrent addition"
    finally:
        readonly.chmod(0o644)


def test_existing_destination_never_removed(tmp_path):
    api = snapshot_api()
    source = tmp_path / "notes.txt"
    source.write_text("notes")
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("keep")
    with pytest.raises(ValueError):
        api.snapshot_tree(source, target, max_files=1, max_bytes=10)
    assert (target / "keep").read_text() == "keep"


def test_overlap_check_accounts_for_resolved_links(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="invalid_arguments"):
        api.validate_output_locations([source], alias / "outputs", None)
    api.validate_output_locations([source], tmp_path / "outside", None)


def test_source_deleted_before_copy_is_reported_as_change(tmp_path):
    api = snapshot_api()
    source = tmp_path / "notes.txt"
    source.write_text("notes")
    calls = [0]

    def check():
        calls[0] += 1
        if calls[0] == 2:
            source.unlink()

    with pytest.raises(ValueError, match="source_changed"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=1, max_bytes=10, check=check)
    assert not (tmp_path / "copy").exists()


def test_snapshot_stops_enumerating_when_file_limit_is_exceeded(tmp_path):
    api = snapshot_api()
    source = tmp_path / "source"
    source.mkdir()
    for i in range(100):
        (source / str(i)).write_text("x")
    calls = [0]

    def check():
        calls[0] += 1

    with pytest.raises(ValueError, match="budget_exhausted"):
        api.snapshot_tree(source, tmp_path / "copy", max_files=0, max_bytes=0, check=check)
    assert calls[0] <= 2
