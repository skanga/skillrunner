import importlib
import zipfile

import pytest


def api():
    assert importlib.util.find_spec("skillrunner.artifacts") is not None, (
        "Artifact validation has not been implemented"
    )
    return importlib.import_module("skillrunner.artifacts.validation")


@pytest.mark.parametrize(
    "format,content",
    [("md", "# Answer"), ("txt", "hello"), ("json", '{"value": 1}'), ("csv", "a,b\n1,2\n")],
)
def test_valid_builtin_document(tmp_path, format, content):
    path = tmp_path / "candidate"
    path.write_text(content, encoding="utf-8")
    result = api().validate_builtin(path, format, size_limit=1000, archive_expanded_limit=1000)
    assert result.valid
    assert result.validation_level == "parsed"


@pytest.mark.parametrize(
    "format,content",
    [("txt", b"\xff"), ("json", b"{broken"), ("json", b"NaN"), ("csv", b'a,"unterminated')],
)
def test_invalid_content_is_not_validated_by_extension(tmp_path, format, content):
    path = tmp_path / f"candidate.{format}"
    path.write_bytes(content)
    result = api().validate_builtin(path, format, size_limit=1000, archive_expanded_limit=1000)
    assert not result.valid
    assert "unterminated" not in repr(result)


def test_unsupported_format_requires_external_validator(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"%PDF-fake")
    with pytest.raises(ValueError, match="unsupported_capability"):
        api().validate_builtin(path, "pdf", size_limit=1000, archive_expanded_limit=1000)


def test_candidate_size_checked_before_parsing(tmp_path):
    path = tmp_path / "candidate"
    path.write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="budget_exhausted"):
        api().validate_builtin(path, "txt", size_limit=10, archive_expanded_limit=1000)


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "C:/absolute", "dir\\escape", "dir\x00escape"]
)
def test_unsafe_zip_member_rejected_without_extraction(tmp_path, member):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        # ZipInfo normalizes backslashes on Windows; preserve the hostile wire name.
        info = zipfile.ZipInfo("placeholder")
        info.filename = member
        archive.writestr(info, b"content")
    with zipfile.ZipFile(path) as archive:
        assert [item.orig_filename for item in archive.infolist()] == [member]
    result = api().validate_builtin(path, "zip", size_limit=1000, archive_expanded_limit=1000)
    assert not result.valid
    assert not (tmp_path / "escape").exists()


def test_compressed_archive_expansion_is_bounded(tmp_path):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large", b"x" * 10000)
    with pytest.raises(ValueError, match="budget_exhausted"):
        api().validate_builtin(path, "zip", size_limit=1000, archive_expanded_limit=100)


@pytest.mark.parametrize(
    "format,part", [("docx", "word/document.xml"), ("xlsx", "xl/workbook.xml")]
)
def test_office_archive_reports_container_validation(tmp_path, format, part):
    path = tmp_path / "candidate"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ["[Content_Types].xml", "_rels/.rels", part]:
            archive.writestr(name, "<root/>")
    result = api().validate_builtin(path, format, size_limit=1000, archive_expanded_limit=1000)
    assert result.valid
    assert result.validation_level == "container"


def test_zip_renamed_docx_missing_required_parts_fails(tmp_path):
    path = tmp_path / "candidate.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("text.txt", "ordinary zip")
    result = api().validate_builtin(path, "docx", size_limit=1000, archive_expanded_limit=1000)
    assert not result.valid


def test_validation_calls_deadline_check(tmp_path):
    path = tmp_path / "candidate"
    path.write_text("text")

    def stop():
        raise ValueError("cancelled")

    with pytest.raises(ValueError, match="cancelled"):
        api().validate_builtin(
            path, "txt", size_limit=1000, archive_expanded_limit=1000, check=stop
        )


def test_corrupt_deflate_stream_is_invalid_artifact(tmp_path):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("file", "hello" * 10)
    content = bytearray(path.read_bytes())
    content[34] = 0xFF  # Invalid deflate block type, after the local header and name.
    path.write_bytes(content)
    result = api().validate_builtin(path, "zip", size_limit=1000, archive_expanded_limit=1000)
    assert not result.valid


def test_later_checkpoint_failure_propagates(tmp_path):
    path = tmp_path / "candidate"
    path.write_text("text")
    checks = 0

    def stop():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("cancelled")

    with pytest.raises(ValueError, match="cancelled"):
        api().validate_builtin(
            path, "txt", size_limit=1000, archive_expanded_limit=1000, check=stop
        )


def test_corrupt_lzma_stream_is_invalid_artifact(tmp_path):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as archive:
        archive.writestr("x", "hello" * 100)
    content = bytearray(path.read_bytes())
    content[35] = 0xFF
    path.write_bytes(content)
    result = api().validate_builtin(path, "zip", size_limit=1000, archive_expanded_limit=1000)
    assert not result.valid


@pytest.mark.parametrize(
    "compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA]
)
def test_forged_member_size_cannot_hide_expanded_bytes(tmp_path, compression):
    import struct
    import zlib

    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("x", b"x" * 10000)
    content = bytearray(path.read_bytes())
    central = content.index(b"PK\x01\x02")
    for offset in [14, central + 16]:
        struct.pack_into("<I", content, offset, zlib.crc32(b"x"))
    for offset in [22, central + 24]:
        struct.pack_into("<I", content, offset, 1)
    path.write_bytes(content)
    with pytest.raises(ValueError, match="budget_exhausted"):
        api().validate_builtin(path, "zip", size_limit=20000, archive_expanded_limit=100)


@pytest.mark.parametrize(
    "compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA]
)
def test_genuine_zip_streams_validate(tmp_path, compression):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("x", b"hello" * 10000)
    result = api().validate_builtin(
        path, "zip", size_limit=100000, archive_expanded_limit=16 * 1024 * 1024
    )
    assert result.valid


def test_deflate_drains_buffered_output_after_compressed_input(tmp_path):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x", b"x" * 65537)
    result = api().validate_builtin(path, "zip", size_limit=100000, archive_expanded_limit=100000)
    assert result.valid
