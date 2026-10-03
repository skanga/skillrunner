"""Explicit input selection, transport settings and non-destructive run management."""

import json
from pathlib import Path

from typer.testing import CliRunner

from skillrunner.catalog.snapshots import snapshot_tree
from skillrunner.cli.app import app
from skillrunner.config.sources import resolve_settings
from skillrunner.recording.bundle import RunBundle


def test_snapshot_exclusions_are_explicit_and_stable(tmp_path):
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    (source / ".git/secret").write_text("not shared")
    (source / "notes.txt").write_text("shared")
    result = snapshot_tree(
        source, tmp_path / "copy", max_files=10, max_bytes=1000, exclude=[".git"]
    )
    assert [entry.relative_path for entry in result.files] == ["notes.txt"]
    assert not (result.root / ".git").exists()


def test_input_preview_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / ".env").write_text("private")
    result = CliRunner().invoke(app, ["inputs", "preview", "-i", ".", "--exclude", ".env", "-j"])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.stdout)
    assert shown["inputs"][0]["files"] == ["notes.txt"]
    assert shown["excludes"] == [".env"]
    assert not (tmp_path / "outputs").exists()


def test_transport_paths_resolve_against_configuration(tmp_path):
    config = tmp_path / "team/config.toml"
    config.parent.mkdir()
    config.write_text("""[models.local]
base_url="https://test.invalid/v1"
model="fake"
proxy_url="http://localhost:8080"
ca_bundle="certs/root.pem"
[mcp.remote]
transport="streamable-http"
url="https://mcp.invalid"
proxy_url="http://localhost:8080"
ca_bundle="certs/root.pem"
""")
    settings = resolve_settings(tmp_path, {"config": config}, {})
    assert Path(settings.models["local"].ca_bundle) == config.parent / "certs/root.pem"
    assert Path(settings.mcp["remote"].ca_bundle) == config.parent / "certs/root.pem"


def test_run_listing_and_cleanup_preview_skip_active_bundles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    completed = RunBundle.create(tmp_path / "outputs", invocation_directory=tmp_path)
    completed.finalize(status="succeeded", exit_code=0, answer="done")
    active = RunBundle.create(tmp_path / "outputs", invocation_directory=tmp_path)
    try:
        listed = CliRunner().invoke(app, ["runs", "list", "-j"])
        assert listed.exit_code == 0, listed.output
        assert len(json.loads(listed.stdout)["runs"]) == 2
        preview = CliRunner().invoke(app, ["runs", "clean", "-j"])
        assert preview.exit_code == 0, preview.output
        assert json.loads(preview.stdout)["removed"] == []
        assert completed.root.is_dir() and active.root.is_dir()
        deleted = CliRunner().invoke(app, ["runs", "clean", "--yes", "-j"])
        assert deleted.exit_code == 0, deleted.output
        assert not completed.root.exists()
        assert active.root.exists()
    finally:
        if active.events:
            active.events.close()


def test_cleanup_does_not_follow_bundle_links(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = RunBundle.create(tmp_path / "elsewhere", invocation_directory=tmp_path)
    outside.finalize(status="succeeded", exit_code=0, answer="keep")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    link = outputs / outside.root.name
    try:
        link.symlink_to(outside.root, target_is_directory=True)
    except OSError:
        import pytest

        pytest.skip("symlink privileges unavailable")
    result = CliRunner().invoke(app, ["runs", "clean", "--yes", "-j"])
    assert result.exit_code == 0
    assert outside.root.exists()
