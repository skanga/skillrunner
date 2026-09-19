import hashlib
import importlib
import os

import pytest

from skillrunner.domain.errors import RunnerError


def setup_tools(tmp_path, **kwargs):
    from skillrunner import tools

    assert hasattr(tools, "FileTools"), "Workspace file tools are not implemented"
    instance = tools.FileTools(**kwargs)
    for name in ("inputs", "skills", "scratch", "artifacts"):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        instance.register_root(
            name,
            root,
            writable=name in ("scratch", "artifacts"),
            max_bytes=32 if name == "scratch" else None,
        )
    return instance


def digest(content):
    return hashlib.sha256(content.encode()).hexdigest()


def test_write_read_and_unicode_byte_paging(tmp_path):
    tools = setup_tools(tmp_path, max_read_bytes=5)
    result = tools.write_file("scratch/a.txt", "ééé")
    assert result["sha256"] == digest("ééé")
    page = tools.read_text("scratch/a.txt")
    assert page["text"] == "éé"
    assert page["truncated"] and page["next_offset"] == 4
    assert tools.read_text("scratch/a.txt", offset=4)["text"] == "é"
    with pytest.raises(RunnerError):
        tools.read_text("scratch/a.txt", offset=1)


def test_protected_paths_outside_links_and_special_files(tmp_path):
    tools = setup_tools(tmp_path)
    protected = tmp_path / "inputs" / "private.txt"
    protected.write_text("protected")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (tmp_path / "scratch" / "outside").symlink_to(outside)
    (tmp_path / "scratch" / "alias").symlink_to(protected)
    os.link(protected, tmp_path / "scratch" / "hardlink")
    for path in ("scratch/outside", str(outside), "scratch/../outside.txt"):
        with pytest.raises(RunnerError):
            tools.read_text(path)
    for path in ("inputs/private.txt", "scratch/alias", "scratch/hardlink"):
        with pytest.raises(RunnerError):
            tools.write_file(path, "changed", overwrite=True, expected_sha256=digest("protected"))
    assert protected.read_text() == "protected"
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "scratch" / "fifo")
        with pytest.raises(RunnerError):
            tools.read_text("scratch/fifo")


def test_atomic_creation_does_not_overwrite_racing_creator(tmp_path, monkeypatch):
    tools = setup_tools(tmp_path)
    module = importlib.import_module("skillrunner.tools.files")
    original = module.os.link

    def race(source, destination, *args, **kwargs):
        (tmp_path / "scratch" / "new.txt").write_text("competitor")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "link", race)
    with pytest.raises(RunnerError):
        tools.write_file("scratch/new.txt", "ours")
    assert (tmp_path / "scratch" / "new.txt").read_text() == "competitor"
    assert sorted(p.name for p in (tmp_path / "scratch").iterdir()) == ["new.txt"]


def test_digest_checked_edit_rejects_stale_and_ambiguous(tmp_path):
    tools = setup_tools(tmp_path)
    tools.write_file("scratch/a", "hello hello")
    with pytest.raises(RunnerError):
        tools.edit_file("scratch/a", expected_sha256=digest("wrong"), old="hello", new="bye")
    with pytest.raises(RunnerError):
        tools.edit_file("scratch/a", expected_sha256=digest("hello hello"), old="hello", new="bye")
    result = tools.edit_file(
        "scratch/a", expected_sha256=digest("hello hello"), old="hello hello", new="bye"
    )
    assert result["sha256"] == digest("bye")
    assert tools.read_text("scratch/a")["text"] == "bye"


def test_list_pagination_and_literal_unicode_search(tmp_path):
    tools = setup_tools(tmp_path)
    for name in ("one.txt", "two.txt", "skip.csv"):
        (tmp_path / "inputs" / name).write_text("a.b é\naxb\n", encoding="utf-8")
    first = tools.list_files("inputs", limit=2)
    assert len(first["entries"]) == 2 and first["truncated"]
    second = tools.list_files("inputs", offset=first["next_offset"], limit=2)
    assert len(second["entries"]) == 1 and not second["truncated"]
    result = tools.search_text("inputs", "a.b é", glob="*.txt")
    assert len(result["matches"]) == 2
    assert all(
        item["line"] == 1 and item["path"].startswith("inputs/") for item in result["matches"]
    )


@pytest.mark.parametrize("path", [".", "./", "././"])
def test_dot_only_paths_require_a_registered_root(tmp_path, path):
    tools = setup_tools(tmp_path)
    with pytest.raises(RunnerError, match="file_access_denied") as error:
        tools.list_files(path)
    assert "registered workspace root" in str(error.value)
    assert tools.list_files("scratch/.")["entries"] == []


def test_quota_overwrite_accounting_and_cancellation(tmp_path):
    checks = []

    def check():
        checks.append(True)

    tools = setup_tools(tmp_path, check=check)
    tools.write_file("scratch/a", "a" * 20)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        tools.write_file("scratch/b", "b" * 20)
    tools.write_file("scratch/a", "a" * 30, overwrite=True, expected_sha256=digest("a" * 20))
    assert checks

    def cancelled():
        raise RunnerError("cancelled", "Cancelled")

    tools.check = cancelled
    with pytest.raises(RunnerError, match="cancelled"):
        tools.list_files("scratch")


def test_output_bounds_media_and_dynamic_roots(tmp_path):
    tools = setup_tools(tmp_path, max_tool_output_bytes=240, max_read_bytes=1024)
    (tmp_path / "inputs" / "large").write_text('"' * 1000)
    result = tools.read_text("inputs/large")
    import json

    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 240
    assert result["truncated"]
    with pytest.raises(RunnerError, match="unsupported_capability"):
        tools.read_media("inputs/large", representation="image")
    active = tmp_path / "active"
    active.mkdir()
    (active / "SKILL.md").write_text("instructions")
    tools.register_root("newskill", active)
    assert tools.read_text("newskill/SKILL.md")["text"] == "instructions"


def test_edit_is_independent_of_display_page_limits_and_rejects_overlap(tmp_path):
    tools = setup_tools(tmp_path, max_read_bytes=5, max_tool_output_bytes=240)
    content = "x" * 65534 + "target" + "z" * 20
    (tmp_path / "artifacts" / "large").write_text(content)
    result = tools.edit_file(
        "artifacts/large", expected_sha256=digest(content), old="target", new="updated"
    )
    assert result["sha256"] == digest(content.replace("target", "updated"))
    tools.write_file("scratch/overlap", "aaa")
    with pytest.raises(RunnerError, match="invalid_arguments"):
        tools.edit_file("scratch/overlap", expected_sha256=digest("aaa"), old="aa", new="b")


def test_read_rejects_file_replaced_during_hash(tmp_path):
    tools = setup_tools(tmp_path)
    target = tmp_path / "inputs" / "a"
    target.write_text("original")
    # Trigger replacement as soon as the file descriptor is open, inside hashing.
    original_hash = tools._hash

    def replacing_hash(stream):
        replacement = target.with_name("replacement")
        replacement.write_text("changed")
        replacement.replace(target)
        return original_hash(stream)

    tools._hash = replacing_hash
    # Windows prevents replacement of this open handle; POSIX detects the new identity.
    expected = "file_access_denied" if os.name == "nt" else "source_changed"
    with pytest.raises(RunnerError, match=expected):
        tools.read_text("inputs/a")
    assert target.read_text() == ("original" if os.name == "nt" else "changed")


def test_nested_writable_root_cannot_bypass_parent_quota(tmp_path):
    tools = setup_tools(tmp_path)
    nested = tmp_path / "scratch" / "nested"
    nested.mkdir()
    tools.register_root("nested", nested, writable=True, max_bytes=100)
    with pytest.raises(RunnerError, match="budget_exhausted"):
        tools.write_file("nested/large", "x" * 40)


def test_cancelled_edit_preserves_original_and_removes_temporary(tmp_path):
    tools = setup_tools(tmp_path)
    content = "a target z"
    tools.write_file("scratch/file", content)

    def check():
        if list((tmp_path / "scratch").glob(".skillrun-*")):
            raise RunnerError("cancelled", "Cancelled")

    tools.check = check
    with pytest.raises(RunnerError, match="cancelled"):
        tools.edit_file("scratch/file", expected_sha256=digest(content), old="target", new="new")
    assert (tmp_path / "scratch" / "file").read_text() == content
    assert sorted(p.name for p in (tmp_path / "scratch").iterdir()) == ["file"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor-relative race injection")
def test_changed_root_and_directory_link_race_are_denied(tmp_path, monkeypatch):
    tools = setup_tools(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a").write_text("private")
    scratch = tmp_path / "scratch"
    original = os.open
    switched = False

    def race(path, *args, **kwargs):
        nonlocal switched
        if path == "scratch" and not switched:
            switched = True
            scratch.rename(tmp_path / "old-scratch")
            scratch.symlink_to(outside, target_is_directory=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", race)
    with pytest.raises(RunnerError):
        tools.write_file("scratch/a", "changed")
    assert (outside / "a").read_text() == "private"
    with pytest.raises(RunnerError):
        tools.read_text("scratch/a")


def test_search_discloses_skipped_long_and_binary_lines(tmp_path):
    tools = setup_tools(tmp_path, max_read_bytes=10)
    (tmp_path / "inputs" / "long").write_text("x" * 30 + "\nfind\n")
    (tmp_path / "inputs" / "binary").write_bytes(b"\xff\n")
    result = tools.search_text("inputs", "find")
    assert result["truncated"]
    assert result["skipped_lines"] == 1 and result["skipped_files"] == 1
    assert result["matches"][0]["line"] == 2


def test_read_pages_include_digest_of_whole_file(tmp_path):
    tools = setup_tools(tmp_path, max_read_bytes=3)
    (tmp_path / "inputs" / "file").write_text("abcdef")
    result = tools.read_text("inputs/file")
    assert result["text"] == "abc" and result["sha256"] == digest("abcdef")
    result = tools.read_text("inputs/file", offset=3)
    assert result["text"] == "def" and not result["truncated"]
    assert result["sha256"] == digest("abcdef")


def test_non_ascii_paths_results_fit_default_json_encoding(tmp_path):
    tools = setup_tools(tmp_path, max_read_bytes=300, max_tool_output_bytes=260)
    tools.write_file("scratch/é", "é" * 15)
    result = tools.read_text("scratch/é")
    import json

    assert len(json.dumps(result).encode()) <= 260


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor-relative race injection")
def test_registered_root_identity_checked_on_descriptor_open(tmp_path, monkeypatch):
    tools = setup_tools(tmp_path)
    original_root = tmp_path / "inputs"
    (original_root / "a").write_text("approved")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "a").write_text("unapproved")
    original_open = os.open
    swapped = False

    def race(path, *args, **kwargs):
        nonlocal swapped
        if path == "inputs" and not swapped:
            swapped = True
            original_root.rename(tmp_path / "old-inputs")
            replacement.rename(original_root)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", race)
    with pytest.raises(RunnerError, match="file_access_denied"):
        tools.read_text("inputs/a")


def test_overwrite_rechecks_digest_when_metadata_is_unchanged(tmp_path):
    tools = setup_tools(tmp_path)
    tools.write_file("scratch/a", "first")
    target = tmp_path / "scratch" / "a"
    before = target.stat()
    changed = False

    def check():
        nonlocal changed
        if list((tmp_path / "scratch").glob(".skillrun-*")) and not changed:
            changed = True
            target.write_text("other")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    tools.check = check
    with pytest.raises(RunnerError, match="source_changed"):
        tools.write_file("scratch/a", "ours!", overwrite=True, expected_sha256=digest("first"))
    assert target.read_text() == "other"
