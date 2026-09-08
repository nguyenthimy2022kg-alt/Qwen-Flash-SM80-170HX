"""Compact, arithmetic metadata for immutable GDS PLE artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any

from .manifest import (
    _align,
    _fsync_dir,
    _fsync_file,
    _json_dump,
    _sha256,
    _source_files,
    _validate_spec,
)

COMPACT_SCHEMA_VERSION = "ple-gds-compact-v1"


def _copy_exact(src: Path, src_offset: int, out: Any, dst_offset: int, size: int) -> None:
    remaining = size
    with src.open("rb", buffering=0) as source:
        source.seek(src_offset)
        out.seek(dst_offset)
        while remaining:
            chunk = source.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise OSError(f"short read from {src} at {src_offset + size - remaining}")
            written = out.write(chunk)
            if written != len(chunk):
                raise OSError(f"short write to artifact at {dst_offset + size - remaining}")
            remaining -= len(chunk)


def _source_identity(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def slice_source_spec(source_spec: dict[str, Any], row_count: int) -> dict[str, Any]:
    """Return a prefix-only source spec without copying any source payload."""
    total = int(source_spec["row_count"])
    if row_count <= 0 or row_count > total:
        raise ValueError(f"row_count must be in [1, {total}]")
    result = json.loads(json.dumps(source_spec))
    result["row_count"] = row_count
    for component in result["components"]:
        if component.get("global", False):
            continue
        kept = []
        for shard in component["shards"]:
            start = int(shard["row_start"])
            if start >= row_count:
                break
            item = dict(shard)
            item["rows"] = min(int(shard["rows"]), row_count - start)
            kept.append(item)
        component["shards"] = kept
    return result


def convert_compact(
    source_spec: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    block_rows: int = 512,
    alignment: int = 4096,
    include_source_hash: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build a compact immutable generation with arithmetic row addressing."""
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    row_count, row_components, global_components = _validate_spec(source_spec, alignment)
    block_count = (row_count + block_rows - 1) // block_rows
    tail_rows = row_count - (block_count - 1) * block_rows
    cursor = 0
    components = []
    for component in row_components:
        row_bytes = int(component["row_bytes"])
        block_payload_bytes = block_rows * row_bytes
        block_stride = _align(block_payload_bytes, alignment)
        base_offset = _align(cursor, alignment)
        region_size = block_count * block_stride
        components.append({
            "name": component["name"],
            "dtype": component.get("dtype", "UNKNOWN"),
            "row_bytes": row_bytes,
            "base_offset": base_offset,
            "block_stride": block_stride,
            "block_payload_bytes": block_payload_bytes,
            "block_count": block_count,
            "tail_rows": tail_rows,
            "tail_payload_bytes": tail_rows * row_bytes,
            "region_size": region_size,
        })
        cursor = base_offset + region_size
    globals_index = []
    for component in global_components:
        shard = component["shards"][0]
        offset = _align(cursor, alignment)
        size = int(shard["rows"]) * int(shard["row_bytes"])
        globals_index.append({
            "name": component["name"],
            "dtype": component.get("dtype", "UNKNOWN"),
            "offset": offset,
            "size": size,
        })
        cursor = _align(offset + size, alignment)
    source_files = _source_files(source_spec, include_source_hash)
    generation_id = "gen-" + uuid.uuid4().hex
    metadata = {
        "schema": COMPACT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "format": "component-block-stride-v1",
        "data_file": "ple-gds-data.bin",
        "metadata_file": "ple-gds-metadata.json",
        "alignment": alignment,
        "row_count": row_count,
        "block_rows": block_rows,
        "block_count": block_count,
        "tail_block": block_count - 1,
        "tail_rows": tail_rows,
        "components": components,
        "globals": globals_index,
        "data_size": cursor,
        "source_schema": source_spec.get("schema"),
        "source_files": source_files,
        "source_identity_sha256": _source_identity(source_files),
    }
    if dry_run:
        return metadata

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generations = output / "generations"
    generations.mkdir(exist_ok=True)
    temp = generations / (".tmp-" + uuid.uuid4().hex)
    generation = generations / generation_id
    temp.mkdir()
    data_path = temp / metadata["data_file"]
    metadata_path = temp / metadata["metadata_file"]
    try:
        with data_path.open("w+b", buffering=0) as out:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(out.fileno(), 0, cursor)
            else:
                out.truncate(cursor)
            for component, layout in zip(row_components, components, strict=True):
                row_bytes = int(layout["row_bytes"])
                if int(layout["block_stride"]) == int(layout["block_payload_bytes"]):
                    for shard in component["shards"]:
                        size = int(shard["rows"]) * row_bytes
                        destination = int(layout["base_offset"]) + int(shard["row_start"]) * row_bytes
                        _copy_exact(Path(shard["path"]), int(shard["offset"]), out, destination, size)
                else:
                    for block_id in range(block_count):
                        start = block_id * block_rows
                        count = min(block_rows, row_count - start)
                        from .manifest import _read_rows
                        payload = _read_rows(component, start, count)
                        out.seek(int(layout["base_offset"]) + block_id * int(layout["block_stride"]))
                        if out.write(payload) != len(payload):
                            raise OSError(f"short write for component {component['name']}")
            for item, component in zip(globals_index, global_components, strict=True):
                shard = component["shards"][0]
                _copy_exact(Path(shard["path"]), int(shard["offset"]), out,
                            int(item["offset"]), int(item["size"]))
        _fsync_file(data_path)
        metadata["data_sha256"] = _sha256(data_path)
        _json_dump(metadata_path, metadata)
        _fsync_file(metadata_path)
        validate_compact_metadata(metadata_path, check_sources=True, check_data=True)
        _fsync_dir(temp)
        os.replace(temp, generation)
        os.chmod(generation / metadata["data_file"], 0o444)
        os.chmod(generation / metadata["metadata_file"], 0o444)
        os.chmod(generation, 0o555)
        _fsync_dir(generation)
        _fsync_dir(generations)
        current_tmp = output / (".CURRENT.tmp-" + uuid.uuid4().hex)
        try:
            current_tmp.write_text(generation_id + "\n", encoding="ascii")
            _fsync_file(current_tmp)
            os.replace(current_tmp, output / "CURRENT")
            _fsync_dir(output)
        finally:
            current_tmp.unlink(missing_ok=True)
    finally:
        if temp.exists():
            for path in temp.iterdir():
                path.unlink(missing_ok=True)
            temp.rmdir()
    return metadata


def validate_compact_metadata(
    metadata_path: str | os.PathLike[str], *, check_sources: bool = True, check_data: bool = True
) -> dict[str, Any]:
    path = Path(metadata_path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != COMPACT_SCHEMA_VERSION:
        raise ValueError("unsupported compact PLE GDS schema")
    alignment = int(metadata["alignment"])
    row_count = int(metadata["row_count"])
    block_rows = int(metadata["block_rows"])
    block_count = int(metadata["block_count"])
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two")
    if row_count <= 0 or block_rows <= 0 or block_count != (row_count + block_rows - 1) // block_rows:
        raise ValueError("invalid compact row/block geometry")
    data = path.parent / metadata["data_file"]
    if data.stat().st_size != int(metadata["data_size"]):
        raise ValueError("compact data size mismatch")
    cursor = 0
    names: set[str] = set()
    for component in metadata["components"]:
        name = str(component["name"])
        if not name or name in names:
            raise ValueError("duplicate compact component")
        names.add(name)
        base = int(component["base_offset"])
        stride = int(component["block_stride"])
        payload = block_rows * int(component["row_bytes"])
        if base % alignment or base < cursor or stride % alignment or stride < payload:
            raise ValueError(f"invalid compact geometry for {name}")
        if int(component["block_count"]) != block_count:
            raise ValueError(f"block count mismatch for {name}")
        cursor = base + block_count * stride
    for item in metadata.get("globals", []):
        offset, size = int(item["offset"]), int(item["size"])
        if item["name"] in names or offset % alignment or offset < cursor or size <= 0:
            raise ValueError(f"invalid global {item['name']}")
        names.add(item["name"])
        cursor = _align(offset + size, alignment)
    if cursor != int(metadata["data_size"]):
        raise ValueError("compact data layout does not cover data_size")
    if check_data and _sha256(data) != metadata.get("data_sha256"):
        raise ValueError("compact data SHA-256 mismatch")
    if check_sources:
        for source in metadata["source_files"]:
            stat = Path(source["path"]).stat()
            if stat.st_size != int(source["size"]) or stat.st_mtime_ns != int(source["mtime_ns"]):
                raise ValueError(f"source changed since conversion: {source['path']}")
            if source.get("sha256") and _sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"source hash changed since conversion: {source['path']}")
    return metadata


def load_current_compact(
    output_dir: str | os.PathLike[str], *, check_sources: bool = False, check_data: bool = False,
    require_immutable: bool = True,
) -> tuple[dict[str, Any], Path]:
    output = Path(output_dir).resolve()
    generation_id = (output / "CURRENT").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"gen-[0-9a-f]{32}", generation_id):
        raise ValueError("invalid CURRENT generation")
    generation = output / "generations" / generation_id
    metadata_path = generation / "ple-gds-metadata.json"
    metadata = validate_compact_metadata(metadata_path, check_sources=check_sources, check_data=check_data)
    if metadata["generation_id"] != generation_id:
        raise ValueError("CURRENT does not match compact metadata")
    data_path = generation / metadata["data_file"]
    if require_immutable:
        for immutable_path in (generation, metadata_path, data_path):
            if stat.S_IMODE(immutable_path.stat().st_mode) & 0o222:
                raise PermissionError(f"compact generation is writable: {immutable_path}")
    identity = metadata.get("data_sha256")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("compact metadata has no valid data identity")
    return metadata, generation


def plan_page_requests(
    row_ids: Any, metadata: dict[str, Any], component_name: str
) -> dict[str, Any]:
    """Plan minimal aligned page ranges and GPU gather offsets."""
    ids = [int(value) for value in row_ids]
    row_count = int(metadata["row_count"])
    for value in ids:
        if value < 0 or value >= row_count:
            raise IndexError(f"row id {value} outside [0, {row_count})")
    component = next((item for item in metadata["components"] if item["name"] == component_name), None)
    if component is None:
        raise KeyError(f"unknown component {component_name!r}")
    alignment = int(metadata["alignment"])
    block_rows = int(metadata["block_rows"])
    row_bytes = int(component["row_bytes"])

    def row_offset(row_id: int) -> int:
        block_id, within = divmod(row_id, block_rows)
        return (int(component["base_offset"]) + block_id * int(component["block_stride"])
                + within * row_bytes)

    unique = sorted(set(ids))
    pages: set[int] = set()
    absolute_offsets = {}
    for row_id in unique:
        offset = row_offset(row_id)
        absolute_offsets[row_id] = offset
        first = offset // alignment
        last = (offset + row_bytes - 1) // alignment
        pages.update(range(first, last + 1))
    sorted_pages = sorted(pages)
    ranges = []
    for page in sorted_pages:
        if ranges and page == ranges[-1]["last_page"] + 1:
            ranges[-1]["last_page"] = page
            ranges[-1]["size"] += alignment
        else:
            ranges.append({"first_page": page, "last_page": page,
                           "file_offset": page * alignment, "size": alignment})
    staging_cursor = 0
    page_to_staging: dict[int, int] = {}
    for item in ranges:
        item["staging_offset"] = staging_cursor
        for page in range(int(item["first_page"]), int(item["last_page"]) + 1):
            page_to_staging[page] = staging_cursor + (page - int(item["first_page"])) * alignment
        staging_cursor += int(item["size"])

    def staging_offset(row_id: int) -> int:
        absolute = absolute_offsets[row_id]
        page, within_page = divmod(absolute, alignment)
        result = page_to_staging.get(page)
        if result is None:
            raise AssertionError(f"row {row_id} is not covered by a planned page")
        if within_page + row_bytes > alignment:
            next_page = page_to_staging.get(page + 1)
            if next_page != result + alignment:
                raise AssertionError(f"cross-page row {row_id} is not contiguous in staging")
        return result + within_page

    gather_offsets = [staging_offset(value) for value in ids]
    requested_bytes = len(ids) * row_bytes
    return {
        "requested_count": len(ids),
        "unique_count": len(unique),
        "unique_row_ids": unique,
        "row_bytes": row_bytes,
        "unique_page_count": len(sorted_pages),
        "merged_io_count": len(ranges),
        "ranges": ranges,
        "gather_offsets": gather_offsets,
        "requested_bytes": requested_bytes,
        "read_bytes": staging_cursor,
        "io_amplification": (staging_cursor / requested_bytes if requested_bytes else 0.0),
        "staging_bytes": staging_cursor,
    }
