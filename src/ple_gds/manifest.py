"""Versioned, aligned PLE artifact builder.

The builder deliberately uses only the Python standard library.  The input is
a small source-spec JSON file describing row-wise tensor shards and optional
global tensors.  Source bytes are copied into an aligned, block-addressable
artifact; the original checkpoint is never changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Any

SOURCE_SCHEMA_VERSION = "ple-gds-source-v1"
SCHEMA_VERSION = "ple-gds-v1"
_KNOWN_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "BF16": 2, "F16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, size: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(offset)
        remaining = size
        while remaining:
            chunk = f.read(min(remaining, chunk_size))
            if not chunk:
                raise OSError(f"short read while hashing {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _dtype_itemsize(dtype: str) -> int | None:
    return _KNOWN_ITEMSIZE.get(dtype.upper())


def _safe_generation_id(value: str) -> bool:
    return bool(re.fullmatch(r"gen-[0-9a-f]{32}", value))


def plan_row_requests(row_ids: Any, *, row_count: int, block_rows: int) -> dict[str, Any]:
    """Plan deduplicated block reads while retaining output-order positions.

    The planner intentionally returns metadata only.  It is usable by a
    future cuFile reader for empty, duplicate, unordered, and cross-block ID
    batches without allocating a copy of the PLE table.
    """
    if row_count < 0 or block_rows <= 0:
        raise ValueError("row_count must be non-negative and block_rows positive")
    ids = [int(value) for value in row_ids]
    for value in ids:
        if value < 0 or value >= row_count:
            raise IndexError(f"row id {value} outside [0, {row_count})")
    unique = sorted(set(ids))
    blocks: dict[int, dict[str, Any]] = {}
    for value in unique:
        block_id = value // block_rows
        block = blocks.setdefault(block_id, {"block_id": block_id,
                                             "row_start": block_id * block_rows,
                                             "row_count": min(block_rows, row_count - block_id * block_rows),
                                             "row_ids": []})
        block["row_ids"].append(value)
    positions = {value: [] for value in unique}
    for position, value in enumerate(ids):
        positions[value].append(position)
    return {"requested_count": len(ids), "unique_count": len(unique),
            "unique_row_ids": unique, "positions": positions,
            "blocks": list(blocks.values())}


def _parse_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 100 * 1024 * 1024:
            raise ValueError(f"{path}: unreasonable safetensors header length {header_len}")
        header = json.loads(f.read(header_len))
        if not isinstance(header, dict):
            raise ValueError(f"{path}: safetensors header is not an object")
    return header, 8 + header_len


def load_safetensors_source_spec(index_path: str | os.PathLike[str], mapping_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Expand a tensor-name mapping and safetensors index into a source spec.

    Mapping format::

      {"row_count": 1000, "components": [
        {"name": "packed", "dtype": "U8", "shards": [
          {"tensor": "...shard_0.weight", "row_start": 0}]},
        {"name": "global_scale", "global": true, "shards": [
          {"tensor": "...weight_scale_2"}]}
      ]}
    """
    index_path = Path(index_path).resolve()
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index must contain an object weight_map")
    row_count = int(mapping["row_count"])
    if row_count <= 0:
        raise ValueError("row_count must be positive")

    header_cache: dict[str, tuple[dict[str, Any], int]] = {}
    components = []
    for component in mapping.get("components", []):
        name = str(component["name"])
        shards = []
        for shard in component["shards"]:
            tensor = str(shard["tensor"])
            filename = weight_map.get(tensor)
            if filename is None and not shard.get("path"):
                raise ValueError(f"tensor {tensor!r} is absent from weight_map")
            source_name = shard.get("path", filename)
            source_path = Path(source_name)
            if not source_path.is_absolute():
                source_path = (index_path.parent / source_path).resolve()
            else:
                source_path = source_path.resolve()
            key = str(source_path)
            if key not in header_cache:
                header_cache[key] = _parse_safetensors_header(source_path)
            header, data_start = header_cache[key]
            entry = header.get(tensor)
            if not isinstance(entry, dict):
                raise ValueError(f"tensor {tensor!r} absent from {source_path}")
            offsets = entry.get("data_offsets")
            shape = entry.get("shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or not isinstance(shape, list):
                raise ValueError(f"tensor {tensor!r} has invalid safetensors metadata")
            begin, end = map(int, offsets)
            if begin < 0 or end < begin:
                raise ValueError(f"tensor {tensor!r} has invalid offsets")
            shape = [int(x) for x in shape]
            header_dtype = str(entry.get("dtype", "UNKNOWN"))
            mapped_dtype = component.get("dtype")
            if mapped_dtype is not None and str(mapped_dtype) != header_dtype:
                raise ValueError(
                    f"tensor {tensor!r} dtype mismatch: header={header_dtype}, mapping={mapped_dtype}"
                )
            dtype = header_dtype
            rows = int(shard.get("rows", shape[0] if shape else 1))
            if not component.get("global", False) and len(shape) < 1:
                raise ValueError(f"row component {name!r} tensor {tensor!r} is scalar")
            if rows <= 0 or (not component.get("global", False) and rows != shape[0]):
                raise ValueError(f"tensor {tensor!r} rows do not match shape {shape}")
            itemsize = _dtype_itemsize(dtype)
            element_count = 1
            for dim in shape:
                if dim < 0:
                    raise ValueError(f"tensor {tensor!r} has negative shape")
                element_count *= dim
            if itemsize is None:
                raise ValueError(f"tensor {tensor!r} has unsupported dtype {dtype!r}")
            if (end - begin) != element_count * itemsize:
                raise ValueError(f"tensor {tensor!r} byte length disagrees with shape/dtype")
            row_bytes = (end - begin) // rows
            expected_row_bytes = itemsize
            for dim in shape[1:]:
                expected_row_bytes *= dim
            if row_bytes != expected_row_bytes:
                raise ValueError(f"tensor {tensor!r} row_bytes disagrees with shape/dtype")
            if "row_bytes" in component and int(component["row_bytes"]) != row_bytes:
                raise ValueError(f"tensor {tensor!r} row_bytes mismatch in mapping")
            shards.append({
                "path": str(source_path),
                "offset": data_start + begin,
                "rows": rows,
                "row_bytes": row_bytes,
                "row_start": int(shard.get("row_start", 0)),
                "tensor": tensor,
                "dtype": dtype,
                "shape": shape,
            })
        item = {"name": name, "dtype": str(component.get("dtype", shards[0]["dtype"])), "shards": shards}
        if component.get("global", False):
            item["global"] = True
        else:
            item["row_bytes"] = int(component.get("row_bytes", shards[0]["row_bytes"]))
        components.append(item)
    return {"schema": SOURCE_SCHEMA_VERSION, "row_count": row_count, "components": components}


def load_source_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    spec_path = Path(path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != SOURCE_SCHEMA_VERSION:
        raise ValueError(f"source spec schema must be {SOURCE_SCHEMA_VERSION!r}")
    if int(spec.get("row_count", 0)) <= 0:
        raise ValueError("source spec row_count must be positive")
    components = spec.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("source spec must contain components")
    names = [str(component.get("name", "")) for component in components]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("source spec component names must be globally unique")
    result = json.loads(json.dumps(spec))
    for component in result["components"]:
        if not component.get("name") or not isinstance(component.get("shards"), list):
            raise ValueError("each component needs name and shards")
        for shard in component["shards"]:
            shard["path"] = str((spec_path.parent / shard["path"]).resolve())
            fields = ("offset", "rows", "row_bytes")
            if not component.get("global", False):
                fields += ("row_start",)
            for field in fields:
                if field not in shard:
                    raise ValueError(f"source shard missing {field}")
                shard[field] = int(shard[field])
            shard.setdefault("row_start", 0)
    return result


def _validate_spec(spec: dict[str, Any], alignment: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    row_count = int(spec["row_count"])
    row_components: list[dict[str, Any]] = []
    global_components: list[dict[str, Any]] = []
    names = [str(component.get("name", "")) for component in spec["components"]]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("source spec component names must be globally unique")
    for component in spec["components"]:
        shards = sorted(component["shards"], key=lambda s: int(s.get("row_start", 0)))
        if not shards:
            raise ValueError(f"component {component['name']!r} has no shards")
        if component.get("global", False):
            if len(shards) != 1 or int(shards[0]["rows"]) <= 0:
                raise ValueError(f"global component {component['name']!r} needs one non-empty shard")
            shard = shards[0]
            path = Path(shard["path"])
            end = int(shard["offset"]) + int(shard["rows"]) * int(shard["row_bytes"])
            if int(shard["offset"]) < 0 or int(shard["row_bytes"]) <= 0 or end > path.stat().st_size:
                raise ValueError(f"global source span for {component['name']!r} exceeds {path}")
            global_components.append(component)
            continue
        row_bytes = int(component.get("row_bytes", shards[0]["row_bytes"]))
        if row_bytes <= 0:
            raise ValueError(f"component {component['name']!r} row_bytes must be positive")
        itemsize = _dtype_itemsize(str(component.get("dtype", "")))
        shape = component.get("shape")
        if itemsize is not None and isinstance(shape, list) and shape:
            expected_row_bytes = itemsize
            for dim in shape[1:]:
                expected_row_bytes *= int(dim)
            if row_bytes != expected_row_bytes:
                raise ValueError(f"component {component['name']!r} row_bytes disagrees with dtype/shape")
        expected = 0
        for shard in shards:
            if int(shard["row_start"]) != expected:
                raise ValueError(f"component {component['name']!r} shards do not cover rows contiguously")
            if int(shard["rows"]) <= 0 or int(shard["row_bytes"]) != row_bytes:
                raise ValueError(f"component {component['name']!r} has inconsistent shard geometry")
            path = Path(shard["path"])
            size = path.stat().st_size
            end = int(shard["offset"]) + int(shard["rows"]) * row_bytes
            if int(shard["offset"]) < 0 or end > size:
                raise ValueError(f"source span for {component['name']!r} exceeds {path}")
            expected += int(shard["rows"])
        if expected != row_count:
            raise ValueError(f"component {component['name']!r} covers {expected} rows, expected {row_count}")
        row_components.append({**component, "shards": shards, "row_bytes": row_bytes})
    if not row_components:
        raise ValueError("source spec needs at least one row component")
    return row_count, row_components, global_components


def _read_rows(component: dict[str, Any], start: int, count: int) -> bytes:
    row_bytes = int(component["row_bytes"])
    wanted_start, wanted_end = start, start + count
    chunks: list[bytes] = []
    for shard in component["shards"]:
        shard_start = int(shard["row_start"])
        shard_end = shard_start + int(shard["rows"])
        lo, hi = max(wanted_start, shard_start), min(wanted_end, shard_end)
        if lo >= hi:
            continue
        with Path(shard["path"]).open("rb") as f:
            f.seek(int(shard["offset"]) + (lo - shard_start) * row_bytes)
            data = f.read((hi - lo) * row_bytes)
        if len(data) != (hi - lo) * row_bytes:
            raise OSError(f"short read from {shard['path']}")
        chunks.append(data)
    result = b"".join(chunks)
    expected = count * row_bytes
    if len(result) != expected:
        raise ValueError(f"source rows [{start}, {wanted_end}) produced {len(result)} bytes, expected {expected}")
    return result


def _source_files(spec: dict[str, Any], include_hash: bool) -> list[dict[str, Any]]:
    paths = sorted({str(Path(s["path"]).resolve()) for c in spec["components"] for s in c["shards"]})
    files = []
    for raw in paths:
        path = Path(raw)
        stat = path.stat()
        item: dict[str, Any] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if include_hash:
            item["sha256"] = _sha256(path)
        files.append(item)
    return files


def convert(source_spec: dict[str, Any], output_dir: str | os.PathLike[str], *, block_rows: int = 256,
            alignment: int = 4096, include_source_hash: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Build an aligned artifact in an immutable generation and publish CURRENT."""
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    row_count, row_components, global_components = _validate_spec(source_spec, alignment)
    blocks = []
    cursor = 0
    for block_id, start in enumerate(range(0, row_count, block_rows)):
        count = min(block_rows, row_count - start)
        block_start = _align(cursor, alignment)
        entries = []
        cursor = block_start
        for component in row_components:
            offset = _align(cursor, alignment)
            size = count * int(component["row_bytes"])
            entries.append({"name": component["name"], "offset": offset, "size": size,
                            "row_bytes": int(component["row_bytes"])})
            cursor = offset + size
        block_end = _align(cursor, alignment)
        blocks.append({"block_id": block_id, "row_start": start, "row_count": count,
                       "offset": block_start, "size": block_end - block_start,
                       "components": entries})
        cursor = block_end
    globals_index = []
    for component in global_components:
        shard = component["shards"][0]
        offset = _align(cursor, alignment)
        size = int(shard["rows"]) * int(shard["row_bytes"])
        globals_index.append({"name": component["name"], "dtype": component.get("dtype", "UNKNOWN"),
                              "offset": offset, "size": size,
                              "source": {key: shard[key] for key in
                                         ("path", "offset", "rows", "row_bytes", "tensor", "shape")
                                         if key in shard}})
        cursor = _align(offset + size, alignment)
    generation_id = "gen-" + uuid.uuid4().hex
    manifest = {
        "schema": SCHEMA_VERSION,
        "generation_id": generation_id,
        "format": "aligned-block-rows-v1",
        "alignment": alignment,
        "block_rows": block_rows,
        "row_count": row_count,
        "components": [{
            "name": c["name"],
            "dtype": c.get("dtype", "UNKNOWN"),
            "row_bytes": int(c["row_bytes"]),
            "sources": [{key: shard[key] for key in
                          ("path", "offset", "rows", "row_bytes", "row_start", "tensor", "shape")
                          if key in shard} for shard in c["shards"]],
        } for c in row_components],
        "globals": globals_index,
        "data_file": "ple-gds-data.bin",
        "index_file": "ple-gds-index.json",
        "data_size": cursor,
        "source_schema": source_spec.get("schema"),
        "source_files": _source_files(source_spec, include_source_hash),
        "blocks": blocks,
    }
    if dry_run:
        return manifest

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generations = output / "generations"
    generations.mkdir(exist_ok=True)
    temp_generation = generations / (".tmp-" + uuid.uuid4().hex)
    generation = generations / generation_id
    temp_generation.mkdir()
    data_path = temp_generation / manifest["data_file"]
    index_path = temp_generation / manifest["index_file"]
    manifest_path = temp_generation / "ple-gds-manifest.json"
    try:
        with data_path.open("wb") as out:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(out.fileno(), 0, cursor)
            else:
                out.truncate(cursor)
            for block in blocks:
                for entry in block["components"]:
                    component = next(c for c in row_components if c["name"] == entry["name"])
                    data = _read_rows(component, block["row_start"], block["row_count"])
                    out.seek(entry["offset"])
                    out.write(data)
            for item in globals_index:
                component = next(c for c in global_components if c["name"] == item["name"])
                shard = component["shards"][0]
                with Path(shard["path"]).open("rb") as src:
                    src.seek(int(shard["offset"]))
                    data = src.read(item["size"])
                if len(data) != item["size"]:
                    raise OSError(f"short read for global component {item['name']}")
                out.seek(item["offset"])
                out.write(data)
        _fsync_file(data_path)
        manifest["data_sha256"] = _sha256(data_path)
        for block in manifest["blocks"]:
            block["sha256"] = _sha256_range(data_path, int(block["offset"]), int(block["size"]))
        index = {"schema": SCHEMA_VERSION, "generation_id": generation_id,
                 "data_file": manifest["data_file"],
                 "alignment": alignment, "block_rows": block_rows, "row_count": row_count,
                 "blocks": blocks, "globals": globals_index}
        _json_dump(index_path, index)
        _json_dump(manifest_path, manifest)
        _fsync_file(index_path)
        _fsync_file(manifest_path)
        validate_manifest(manifest_path, data_path=data_path, check_sources=True, check_data=True)
        _fsync_dir(temp_generation)
        os.replace(temp_generation, generation)
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
        if temp_generation.exists():
            for path in temp_generation.iterdir():
                path.unlink(missing_ok=True)
            temp_generation.rmdir()
    return manifest


def validate_manifest(manifest_path: str | os.PathLike[str], *, data_path: str | os.PathLike[str] | None = None,
                      check_sources: bool = True, check_data: bool = True) -> dict[str, Any]:
    """Strictly validate layout geometry, data binding, and source freshness."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("unsupported PLE GDS manifest schema")
    generation_id = str(manifest.get("generation_id", ""))
    if not _safe_generation_id(generation_id):
        raise ValueError("manifest has invalid generation_id")
    alignment = int(manifest["alignment"])
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("manifest alignment must be a power of two")
    data = Path(data_path).resolve() if data_path else path.parent / manifest["data_file"]
    if data.stat().st_size != int(manifest["data_size"]):
        raise ValueError(f"data size mismatch: {data.stat().st_size} != {manifest['data_size']}")
    if check_data:
        expected_hash = manifest.get("data_sha256")
        if not expected_hash or _sha256(data) != expected_hash:
            raise ValueError("derived data SHA-256 mismatch")
    expected_rows = 0
    expected_block_id = 0
    component_names = {str(c["name"]) for c in manifest.get("components", [])}
    if len(component_names) != len(manifest.get("components", [])):
        raise ValueError("manifest contains duplicate component names")
    last_end = 0
    for block in manifest["blocks"]:
        offset, size = int(block["offset"]), int(block["size"])
        if int(block["block_id"]) != expected_block_id or int(block["row_start"]) != expected_rows:
            raise ValueError(f"invalid block row/id sequence at block {block['block_id']}")
        if offset % alignment or size % alignment or offset < last_end:
            raise ValueError(f"invalid block alignment/order for block {block['block_id']}")
        if offset + size > data.stat().st_size:
            raise ValueError("block extends beyond data file")
        if check_data and (not block.get("sha256") or _sha256_range(data, offset, size) != block["sha256"]):
            raise ValueError(f"derived block hash mismatch for block {block['block_id']}")
        seen_components = set()
        for item in block["components"]:
            item_offset, item_size = int(item["offset"]), int(item["size"])
            if item["name"] not in component_names or item["name"] in seen_components:
                raise ValueError(f"invalid/duplicate component in block {block['block_id']}")
            if item_offset % alignment or item_offset < offset or item_offset + item_size > offset + size:
                raise ValueError(f"invalid component range in block {block['block_id']}")
            if item_size != int(block["row_count"]) * int(item["row_bytes"]):
                raise ValueError(f"component size mismatch in block {block['block_id']}")
            seen_components.add(item["name"])
        if seen_components != component_names:
            raise ValueError(f"block {block['block_id']} does not contain all row components")
        expected_rows += int(block["row_count"])
        expected_block_id += 1
        last_end = offset + size
    if expected_rows != int(manifest["row_count"]):
        raise ValueError("manifest blocks do not cover row_count")
    global_names = set()
    for item in manifest.get("globals", []):
        item_offset, item_size = int(item["offset"]), int(item["size"])
        if item["name"] in global_names or item["name"] in component_names:
            raise ValueError(f"duplicate global/component name {item['name']}")
        if item_offset % alignment or item_offset < last_end or item_offset + item_size > data.stat().st_size:
            raise ValueError(f"invalid global range for {item['name']}")
        global_names.add(item["name"])
    if check_sources:
        for source in manifest.get("source_files", []):
            source_path = Path(source["path"])
            stat = source_path.stat()
            if stat.st_size != int(source["size"]) or stat.st_mtime_ns != int(source["mtime_ns"]):
                raise ValueError(f"source changed since conversion: {source_path}")
            if source.get("sha256") and _sha256(source_path) != source["sha256"]:
                raise ValueError(f"source hash changed since conversion: {source_path}")
    return manifest


def load_current_manifest(output_dir: str | os.PathLike[str], *, check_data: bool = True,
                          check_sources: bool = True) -> dict[str, Any]:
    """Resolve and validate the single atomic CURRENT generation pointer."""
    output = Path(output_dir).resolve()
    current = output / "CURRENT"
    generation_id = current.read_text(encoding="ascii").strip()
    if not _safe_generation_id(generation_id):
        raise ValueError("CURRENT contains an invalid generation id")
    generation = output / "generations" / generation_id
    if not generation.is_dir():
        raise FileNotFoundError(f"CURRENT generation is missing: {generation}")
    manifest_path = generation / "ple-gds-manifest.json"
    manifest = validate_manifest(manifest_path, check_data=check_data, check_sources=check_sources)
    if manifest.get("generation_id") != generation_id:
        raise ValueError("CURRENT generation does not match manifest generation_id")
    index = json.loads((generation / manifest["index_file"]).read_text(encoding="utf-8"))
    if index.get("generation_id") != generation_id:
        raise ValueError("CURRENT generation does not match index generation_id")
    return manifest
