"""Full-package corpus preparation cannot silently rewrite or omit source files."""

import hashlib
import importlib
import json
import os

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.qualification.corpus") is not None
    return importlib.import_module("skillrunner.qualification.corpus")


def fixture(tmp_path):
    checkout = tmp_path / "checkout"
    root = checkout / "skills/writer"
    root.mkdir(parents=True)
    blobs = {"SKILL.md": b"instructions\r\n", "LICENSE": b"notice", "scripts/run.py": b"print(1)\n"}
    files = []
    for name, content in blobs.items():
        target = root / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o644)
        files.append(
            {
                "path": name,
                "mode": "100644",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    files.sort(key=lambda item: item["path"])
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    document = {
        "corpus_version": "test",
        "packages": [
            {
                "id": "upstream/writer",
                "name": "writer",
                "repository": "upstream",
                "path": "skills/writer",
                "classification": "candidate",
                "files": files,
                "package_sha256": digest,
            }
        ],
    }
    return document, {"upstream": checkout}, root


def test_prepares_complete_unchanged_packages_without_running_scripts(tmp_path):
    document, checkouts, source = fixture(tmp_path)
    document["packages"].append({"classification": "license-excluded", "id": "excluded"})
    destination = tmp_path / "catalog"
    result = api().prepare_catalog(document, checkouts, destination)
    for entry in document["packages"][0]["files"]:
        assert (destination / "writer" / entry["path"]).read_bytes() == (
            source / entry["path"]
        ).read_bytes()
    assert result["packages"][0]["package_sha256"] == document["packages"][0]["package_sha256"]
    assert result["case_prerequisites_verified"] is False
    assert sorted(p.name for p in destination.iterdir()) == ["writer"]
    with pytest.raises(FileExistsError):
        api().prepare_catalog(document, checkouts, destination)


@pytest.mark.parametrize(
    "change", ["modified", "missing", "extra", "manifest_digest", "unsafe_path"]
)
def test_mismatch_rejected_before_creating_catalog(tmp_path, change):
    document, checkouts, source = fixture(tmp_path)
    if change == "modified":
        (source / "LICENSE").write_text("changed")
    elif change == "missing":
        (source / "LICENSE").unlink()
    elif change == "extra":
        (source / "generated.cache").write_text("extra")
    elif change == "manifest_digest":
        document["packages"][0]["package_sha256"] = "0" * 64
    else:
        document["packages"][0]["path"] = "../outside"
    destination = tmp_path / "catalog"
    with pytest.raises(ValueError):
        api().prepare_catalog(document, checkouts, destination)
    assert not destination.exists()


def test_duplicate_destination_names_rejected(tmp_path):
    document, checkouts, _ = fixture(tmp_path)
    document["packages"].append(dict(document["packages"][0], id="another/writer"))
    with pytest.raises(ValueError, match="Duplicate"):
        api().prepare_catalog(document, checkouts, tmp_path / "catalog")


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable-bit preservation")
def test_executable_mode_mismatch_is_not_silently_normalized(tmp_path):
    document, checkouts, source = fixture(tmp_path)
    (source / "scripts/run.py").chmod(0o755)
    with pytest.raises(ValueError, match="mode"):
        api().prepare_catalog(document, checkouts, tmp_path / "catalog")


def test_catalog_cannot_be_created_inside_its_source(tmp_path, monkeypatch):
    document, checkouts, source = fixture(tmp_path)
    destination = source / "catalog"
    monkeypatch.setattr(
        api().shutil, "copytree", lambda *args, **kwargs: pytest.fail("copy started")
    )
    with pytest.raises(ValueError, match="overlap"):
        api().prepare_catalog(document, checkouts, destination)
    assert not destination.exists()


def test_internal_symlink_bytes_preserved_and_escape_rejected(tmp_path):
    document, checkouts, source = fixture(tmp_path)
    link = source / "notice-link"
    try:
        link.symlink_to("LICENSE")
    except OSError:
        pytest.skip("Symlinks unavailable")
    package = document["packages"][0]
    package["files"].append(
        {
            "path": "notice-link",
            "mode": "120000",
            "size": 7,
            "sha256": hashlib.sha256(b"LICENSE").hexdigest(),
        }
    )
    package["package_sha256"] = hashlib.sha256(
        json.dumps(
            sorted(package["files"], key=lambda item: item["path"]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    destination = tmp_path / "catalog"
    api().prepare_catalog(document, checkouts, destination)
    assert (destination / "writer/notice-link").is_symlink()
    assert os.readlink(destination / "writer/notice-link") == "LICENSE"
    link.unlink()
    link.symlink_to("../../outside")
    with pytest.raises(ValueError, match="escapes"):
        api().prepare_catalog(document, checkouts, tmp_path / "second")


def test_change_during_copy_leaves_partial_evidence_but_never_returns_success(
    tmp_path, monkeypatch
):
    document, checkouts, _ = fixture(tmp_path)
    module = api()
    copytree = module.shutil.copytree

    def changed_copy(source, target, **kwargs):
        # Copying occurs once per package; avoid patching recursive shutil calls.
        with monkeypatch.context() as context:
            context.setattr(module.shutil, "copytree", copytree)
            copytree(source, target, **kwargs)
        (target / "LICENSE").write_text("corrupted")

    monkeypatch.setattr(module.shutil, "copytree", changed_copy)
    destination = tmp_path / "catalog"
    with pytest.raises(ValueError, match="bytes mismatch"):
        module.prepare_catalog(document, checkouts, destination)
    assert destination.exists()
