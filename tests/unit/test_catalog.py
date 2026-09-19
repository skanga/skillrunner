import importlib
from pathlib import Path

import pytest


def catalog_api():
    assert importlib.util.find_spec("skillrunner.catalog.discovery") is not None, (
        "Skill discovery has not been implemented"
    )
    return importlib.import_module("skillrunner.catalog.discovery")


def package(root: Path, name: str = "sample", extra: str = "", body: str = "Instructions."):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Example skill\n{extra}---\n{body}", encoding="utf-8"
    )
    return folder


def test_discovery_retains_unknown_metadata_and_full_instructions(tmp_path):
    api = catalog_api()
    body = "# Instructions\n" + "Keep every line.\n" * 1000
    package(tmp_path, extra="x-custom:\n  version: 9\n", body=body)
    catalog = api.discover(tmp_path)
    assert list(catalog.skills) == ["sample"]
    assert catalog.skills["sample"].metadata["x-custom"] == {"version": 9}
    assert catalog.skills["sample"].instructions == (
        tmp_path / "sample/SKILL.md"
    ).read_bytes().decode("utf-8")
    assert catalog.rejected == []


def test_unicode_uncased_names_and_nfkc_directory_matching(tmp_path):
    api = catalog_api()
    package(tmp_path, "你好")
    (tmp_path / "你好/SKILL.md").write_text(
        "---\nname: 你好\ndescription: 中文技能\n---\nInstructions.", encoding="utf-8"
    )
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample/SKILL.md").write_text(
        "---\nname: ｓａｍｐｌｅ\ndescription: Fullwidth name\n---\nInstructions.",
        encoding="utf-8",
    )

    catalog = api.discover(tmp_path)

    assert set(catalog.skills) == {"你好", "sample"}
    assert catalog.require("sample").name == "sample"


def test_name_length_is_checked_after_nfkc_normalization(tmp_path):
    api = catalog_api()
    canonical = "é" + "a" * 63
    raw = "e\u0301" + "a" * 63
    assert len(raw) == 65 and len(canonical) == 64
    folder = tmp_path / canonical
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: {raw}\ndescription: Normalize first\n---\nInstructions.",
        encoding="utf-8",
    )

    catalog = api.discover(tmp_path)
    assert list(catalog.skills) == [canonical]


def test_name_whitespace_is_trimmed_before_nfkc_directory_comparison(tmp_path):
    api = catalog_api()
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        '---\nname: " sample "\ndescription: Trim name\n---\nInstructions.',
        encoding="utf-8",
    )
    assert list(api.discover(tmp_path).skills) == ["sample"]


def test_nested_package_is_resource_not_separate_skill(tmp_path):
    api = catalog_api()
    outer = package(tmp_path)
    package(outer / "resources", "nested")
    assert list(api.discover(tmp_path).skills) == ["sample"]


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: bad\nname: bad\ndescription: duplicate",
        "name: bad\ndescription: !!python/object/apply:os.system ['echo bad']",
        "name: BAD\ndescription: bad",
        "name: bad--name\ndescription: bad",
        "name: mismatch\ndescription: bad",
        "name: bad\ndescription: ''",
        "name: bad\ndescription: test\ncompatibility: ''",
        "name: bad\ndescription: test\nmetadata:\n  count: 7",
    ],
)
def test_malformed_unrelated_packages_are_reported(tmp_path, frontmatter):
    api = catalog_api()
    package(tmp_path, "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nInstructions")
    catalog = api.discover(tmp_path)
    assert list(catalog.skills) == ["good"]
    assert len(catalog.rejected) == 1
    with pytest.raises(ValueError, match="required_skill_unavailable"):
        catalog.require("bad")


def test_duplicate_declared_names_fail_whole_catalog(tmp_path):
    api = catalog_api()
    package(tmp_path / "a", "same")
    package(tmp_path / "b", "same")
    with pytest.raises(ValueError, match="duplicate_skill_name"):
        api.discover(tmp_path)


def test_discovery_does_not_copy_resources(tmp_path):
    api = catalog_api()
    package(tmp_path)
    assert api.discover(tmp_path).skills["sample"].snapshot is None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sample"]


def test_activation_snapshots_and_reuses_original_copy(tmp_path):
    api = catalog_api()
    source = package(tmp_path / "catalog")
    catalog = api.discover(tmp_path / "catalog")
    activation = importlib.import_module("skillrunner.catalog.activation")
    service = activation.ActivationService(
        catalog, tmp_path / "work", max_files=20, max_bytes=10000
    )
    active = service.activate("sample", reason="Explicit request")
    assert active.snapshot is not None
    original = (active.snapshot.root / "SKILL.md").read_text()
    (source / "SKILL.md").write_text("changed after activation")
    assert service.activate("sample", reason="repeat") is active
    assert (active.snapshot.root / "SKILL.md").read_text() == original
    assert len(service.records) == 1


def test_unexpected_edit_between_discovery_and_activation_rejected(tmp_path):
    api = catalog_api()
    source = package(tmp_path / "catalog")
    resource = source / "resource.txt"
    resource.write_text("first")
    catalog = api.discover(tmp_path / "catalog")
    resource.write_text("changed")
    activation = importlib.import_module("skillrunner.catalog.activation")
    service = activation.ActivationService(
        catalog, tmp_path / "work", max_files=20, max_bytes=10000
    )
    with pytest.raises(ValueError, match="source_changed"):
        service.activate("sample", reason="automatic")
    assert service.records == []


def test_package_bounds_accumulate_only_on_activation(tmp_path):
    api = catalog_api()
    package(tmp_path / "catalog", "first")
    package(tmp_path / "catalog", "second")
    catalog = api.discover(tmp_path / "catalog")
    activation = importlib.import_module("skillrunner.catalog.activation")
    service = activation.ActivationService(catalog, tmp_path / "work", max_files=1, max_bytes=10000)
    service.activate("first", reason="automatic")
    with pytest.raises(ValueError, match="budget_exhausted"):
        service.activate("second", reason="automatic")
    assert list(service.active) == ["first"]


def test_instruction_capacity_failure_does_not_mark_active(tmp_path):
    api = catalog_api()
    package(tmp_path / "catalog")
    catalog = api.discover(tmp_path / "catalog")
    activation = importlib.import_module("skillrunner.catalog.activation")
    service = activation.ActivationService(catalog, tmp_path / "work", max_files=2, max_bytes=10000)

    def reject(instructions):
        raise ValueError("context_capacity_exceeded")

    with pytest.raises(ValueError, match="context_capacity_exceeded"):
        service.activate("sample", reason="automatic", admit_instructions=reject)
    assert not service.active
    assert not service.records
    assert service.total_files == 0


def test_instruction_rejection_cleans_readonly_package_without_marking_active(tmp_path):
    from skillrunner.catalog.activation import ActivationService
    from skillrunner.domain.errors import RunnerError

    source = package(tmp_path / "catalog")
    resource = source / "reference.txt"
    resource.write_bytes(b"original reference")
    resource.chmod(0o444)
    original_mode = resource.stat().st_mode
    catalog = catalog_api().discover(tmp_path / "catalog")
    service = ActivationService(catalog, tmp_path / "work", max_files=10, max_bytes=10000)
    rejected = RunnerError("context_capacity_exceeded", "Instructions do not fit.")

    def reject(instructions):
        assert instructions == (source / "SKILL.md").read_text()
        assert (tmp_path / "work/sample/reference.txt").read_bytes() == b"original reference"
        raise rejected

    try:
        with pytest.raises(RunnerError) as raised:
            service.activate("sample", reason="Required skill", admit_instructions=reject)
        assert raised.value is rejected
        assert not service.active
        assert not service.records
        assert service.total_files == 0 and service.total_bytes == 0
        assert not (tmp_path / "work/sample").exists()
        assert resource.read_bytes() == b"original reference"
        assert resource.stat().st_mode == original_mode
    finally:
        resource.chmod(0o644)


def test_package_link_outside_catalog_rejected(tmp_path):
    api = catalog_api()
    outside = package(tmp_path / "outside")
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "sample").symlink_to(outside, target_is_directory=True)
    catalog = api.discover(root)
    assert not catalog.skills
    assert len(catalog.rejected) == 1


def test_cancellation_inside_package_scan_propagates(tmp_path):
    api = catalog_api()
    package(tmp_path)
    from skillrunner.domain.errors import RunnerError

    calls = [0]

    def check():
        calls[0] += 1
        if calls[0] == 3:
            raise RunnerError("cancelled", "Interrupted")

    with pytest.raises(RunnerError, match="cancelled"):
        api.discover(tmp_path, check=check)


def test_malformed_link_alias_does_not_hide_canonical_package(tmp_path):
    api = catalog_api()
    source = package(tmp_path, "z")
    (tmp_path / "a").symlink_to(source, target_is_directory=True)
    catalog = api.discover(tmp_path)
    assert list(catalog.skills) == ["z"]
    assert len(catalog.rejected) == 1


def test_package_removed_after_discovery_is_source_change(tmp_path):
    api = catalog_api()
    source = package(tmp_path / "catalog")
    catalog = api.discover(tmp_path / "catalog")
    (source / "SKILL.md").unlink()
    source.rmdir()
    activation = importlib.import_module("skillrunner.catalog.activation")
    service = activation.ActivationService(catalog, tmp_path / "work", max_files=2, max_bytes=10000)
    with pytest.raises(ValueError, match="source_changed"):
        service.activate("sample", reason="automatic")
    assert not service.active
    assert not service.records


def test_activation_services_keep_independent_snapshots(tmp_path):
    api = catalog_api()
    package(tmp_path / "catalog")
    catalog = api.discover(tmp_path / "catalog")
    activation = importlib.import_module("skillrunner.catalog.activation")
    first = activation.ActivationService(catalog, tmp_path / "a", max_files=2, max_bytes=10000)
    second = activation.ActivationService(catalog, tmp_path / "b", max_files=2, max_bytes=10000)
    active = first.activate("sample", reason="first run")
    second.activate("sample", reason="second run")
    assert first.activate("sample", reason="repeat").snapshot.root == tmp_path / "a/sample"
    assert active.snapshot.root == tmp_path / "a/sample"
    assert catalog.require("sample").snapshot is None


def test_fifo_replacement_during_discovery_does_not_block(tmp_path):
    import os
    import subprocess
    import sys

    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO")
    package(tmp_path)
    script = """
import os
import sys
from pathlib import Path
from skillrunner.catalog import discovery
original = discovery.scan_tree
def replace_after_scan(source, **kwargs):
    result = original(source, **kwargs)
    instruction = source / 'SKILL.md'
    instruction.unlink()
    os.mkfifo(instruction)
    return result
discovery.scan_tree = replace_after_scan
catalog = discovery.discover(Path(sys.argv[1]))
assert not catalog.skills
assert len(catalog.rejected) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=3
    )
    assert result.returncode == 0, result.stderr.decode()


def test_instruction_removed_before_enumeration_is_reported(tmp_path, monkeypatch):
    api = catalog_api()
    package(tmp_path)
    original = api.scan_tree

    def remove_before_scan(source, **kwargs):
        (source / "SKILL.md").unlink()
        return original(source, **kwargs)

    monkeypatch.setattr(api, "scan_tree", remove_before_scan)
    catalog = api.discover(tmp_path)
    assert not catalog.skills
    assert len(catalog.rejected) == 1
    assert catalog.rejected[0].code == "source_changed"
