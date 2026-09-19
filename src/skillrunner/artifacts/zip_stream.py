"""Validate actual ZIP member streams without trusting declared expanded sizes."""

import bz2
import lzma
import os
import struct
import zipfile
import zlib
from collections.abc import Callable
from typing import Any, BinaryIO

from skillrunner.domain.errors import RunnerError


def member_bytes(
    reader: BinaryIO,
    member: zipfile.ZipInfo,
    boundary: int,
    remaining: int,
    check: Callable[[], None],
) -> int:
    reader.seek(member.header_offset)
    header = reader.read(30)
    if len(header) != 30:
        raise ValueError("Truncated ZIP header")
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack(
        "<4s5H3I2H", header
    )
    if signature != b"PK\x03\x04" or method != member.compress_type or flags != member.flag_bits:
        raise ValueError("ZIP header mismatch")
    if flags & 0x41:
        raise RunnerError(
            "unsupported_capability", "Encrypted archives require an external validator."
        )
    name = reader.read(name_size).decode("utf-8" if flags & 0x800 else "cp437")
    if name != member.orig_filename:
        raise ValueError("ZIP member name mismatch")
    reader.seek(extra_size, os.SEEK_CUR)
    compressed_left = member.compress_size
    if reader.tell() + compressed_left > boundary:
        raise ValueError("Overlapping ZIP members")
    decoder: Any = None
    if method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
    elif method == zipfile.ZIP_BZIP2:
        decoder = bz2.BZ2Decompressor()
    elif method == zipfile.ZIP_LZMA:
        properties = reader.read(min(compressed_left, 9))
        compressed_left -= len(properties)
        if len(properties) != 9 or int.from_bytes(properties[2:4], "little") != 5:
            raise ValueError("Unsupported ZIP LZMA properties")
        prop = properties[4]
        if prop >= 225:
            raise ValueError("Invalid LZMA properties")
        dictionary = int.from_bytes(properties[5:9], "little")
        # Raw LZMA's dictionary allocation is requested by the untrusted file.
        if dictionary > remaining:
            raise RunnerError(
                "budget_exhausted", "Archive decoder dictionary exceeds archive byte limit."
            )
        decoder = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "dict_size": dictionary,
                    "lc": prop % 9,
                    "lp": (prop // 9) % 5,
                    "pb": prop // 45,
                }
            ],
        )
    elif method != zipfile.ZIP_STORED:
        raise RunnerError(
            "unsupported_capability", "ZIP compression requires an external validator."
        )
    expanded = 0
    crc = 0
    while compressed_left:
        check()
        compressed = reader.read(min(64 * 1024, compressed_left))
        if not compressed:
            raise ValueError("Truncated ZIP stream")
        compressed_left -= len(compressed)
        pending = compressed
        while True:
            check()
            if decoder is None:
                data = pending
            else:
                output_limit = min(64 * 1024, remaining - expanded + 1)
                data = decoder.decompress(pending, max_length=output_limit)
            expanded += len(data)
            if expanded > remaining:
                raise RunnerError("budget_exhausted", "Archive expanded-byte limit exceeded.")
            crc = zlib.crc32(data, crc)
            if decoder is None:
                break
            if decoder.eof:
                if decoder.unused_data or compressed_left:
                    raise ValueError("Trailing compressed member data")
                break
            if method == zipfile.ZIP_DEFLATED:
                pending = decoder.unconsumed_tail
                if not pending and len(data) < output_limit:
                    break
            else:
                if decoder.needs_input:
                    break
                pending = b""
    if decoder is not None and not decoder.eof:
        raise ValueError("Incomplete compressed stream")
    if expanded != member.file_size or crc != member.CRC:
        raise ValueError("ZIP size or checksum mismatch")
    return expanded
