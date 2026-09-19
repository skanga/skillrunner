import os
import stat

import pytest


def api():
    import importlib

    assert importlib.util.find_spec("skillrunner.runtime.cleanup") is not None
    return importlib.import_module("skillrunner.runtime.cleanup")


def test_cleanup_retries_owned_readonly_file_without_touching_outside(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    file = root / "snapshot"
    file.write_text("data")
    file.chmod(0o444)
    seen = []

    def unlink(path):
        assert os.stat(path).st_mode & stat.S_IWUSR
        seen.append(path)

    def rmtree(path, *, onexc):
        onexc(unlink, str(file), PermissionError("Read-only file"))

    monkeypatch.setattr(api().shutil, "rmtree", rmtree)
    api().remove_work_tree(root)
    assert seen == [str(file)]


def test_cleanup_does_not_retry_unowned_path(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("data")
    outside.chmod(0o444)
    before = outside.stat().st_mode

    def rmtree(path, *, onexc):
        onexc(os.unlink, str(outside), PermissionError("Not owned"))

    monkeypatch.setattr(api().shutil, "rmtree", rmtree)
    with pytest.raises(PermissionError):
        api().remove_work_tree(root)
    assert outside.stat().st_mode == before
    assert outside.read_text() == "data"
