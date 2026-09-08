#!/usr/bin/env python3
"""Atomically create an auditable FP8-PLE checkpoint view for GDS loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "qwen38-gds-fp8-checkpoint-view-v1"
PLE_PREFIX = "model-plefp8-"
CONFIG_BACKUP = "config.json.plefp8.bak"
INDEX_BACKUP = "model.safetensors.index.json.plefp8.bak"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def expected_manifest(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = source / CONFIG_BACKUP
    index_path = source / INDEX_BACKUP
    config = read_json(config_path)
    index = read_json(index_path)
    text_config = config.get("text_config", config)
    if text_config.get("ple_embedding_dtype") != "float8_e4m3fn":
        raise RuntimeError("backup config is not the required float8_e4m3fn PLE layout")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("backup index has no weight_map object")
    names = tuple(str(name) for name in weight_map)
    shard_weights = [
        name
        for name in names
        if ".ngram_embedding.shard_" in name and name.endswith(".weight")
    ]
    shard_scales = [
        name
        for name in names
        if ".ngram_embedding.shard_" in name and name.endswith(".weight_scale")
    ]
    global_scales = [name for name in names if name.endswith("ngram_embedding.weight_scale")]
    if len(shard_weights) != 128 or shard_scales or len(global_scales) != 1:
        raise RuntimeError(
            "backup index is not the expected FP8 PLE layout: "
            f"weights={len(shard_weights)} shard_scales={len(shard_scales)} "
            f"global_scales={len(global_scales)}"
        )

    referenced = sorted({str(value) for value in weight_map.values()})
    ple_files = [name for name in referenced if name.startswith(PLE_PREFIX)]
    if len(ple_files) != 10:
        raise RuntimeError(f"expected 10 FP8 PLE files, found {len(ple_files)}")

    sources: dict[str, str] = {}
    for name in referenced:
        if Path(name).name != name:
            raise RuntimeError(f"index filename must be a basename: {name!r}")
        path = source / "_off_plefp8" / name if name.startswith(PLE_PREFIX) else source / name
        if not path.is_file():
            raise RuntimeError(f"index source file is missing: {path}")
        sources[name] = str(path.resolve())

    # Tokenizer and processor metadata are linked too, while the authoritative
    # config and index are copied from the FP8 backups below.
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        CONFIG_BACKUP,
        INDEX_BACKUP,
    }
    for path in sorted(source.iterdir()):
        if (
            path.is_file()
            and path.name not in excluded
            and path.suffix != ".safetensors"
            and path.name not in sources
        ):
            sources[path.name] = str(path.resolve())

    manifest = {
        "schema": SCHEMA,
        "source": str(source.resolve()),
        "config_backup": CONFIG_BACKUP,
        "index_backup": INDEX_BACKUP,
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "weight_count": len(weight_map),
        "referenced_file_count": len(referenced),
        "ple_file_count": len(ple_files),
        "ple_files": ple_files,
        "links": [
            {"name": name, "source": sources[name]}
            for name in sorted(sources)
        ],
    }
    return manifest, sources


def verify(target: Path, manifest: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    if sha256(target / "config.json") != manifest["config_sha256"]:
        raise RuntimeError("view config.json does not match FP8 backup")
    if sha256(target / "model.safetensors.index.json") != manifest["index_sha256"]:
        raise RuntimeError("view index does not match FP8 backup")

    dangling: list[str] = []
    wrong_target: list[str] = []
    for name, source in sources.items():
        link = target / name
        if not link.is_symlink():
            wrong_target.append(name)
            continue
        if not link.exists():
            dangling.append(name)
            continue
        if link.resolve() != Path(source):
            wrong_target.append(name)
    index = read_json(target / "model.safetensors.index.json")
    missing = sorted(
        {
            str(filename)
            for filename in index["weight_map"].values()
            if not (target / str(filename)).is_file()
        }
    )
    result = {
        "schema": SCHEMA,
        "target": str(target.resolve()),
        "missing_index_files": missing,
        "dangling_links": dangling,
        "wrong_link_targets": wrong_target,
        "link_count": len(sources),
        "config_sha256": sha256(target / "config.json"),
        "index_sha256": sha256(target / "model.safetensors.index.json"),
        "valid": not missing and not dangling and not wrong_target,
    }
    if not result["valid"]:
        raise RuntimeError(f"checkpoint view verification failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.absolute()
    manifest, sources = expected_manifest(source)
    if target.exists() or target.is_symlink():
        existing = read_json(target / "VIEW_MANIFEST.json")
        if existing != manifest:
            raise RuntimeError(f"refusing to overwrite unknown existing view: {target}")
        result = verify(target, manifest, sources)
        result["created"] = False
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            shutil.copy2(source / CONFIG_BACKUP, temporary / "config.json")
            shutil.copy2(source / INDEX_BACKUP, temporary / "model.safetensors.index.json")
            for name, source_name in sources.items():
                relative = os.path.relpath(source_name, temporary)
                (temporary / name).symlink_to(relative)
            with (temporary / "VIEW_MANIFEST.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            result = verify(temporary, manifest, sources)
            os.replace(temporary, target)
            result["target"] = str(target.resolve())
            result["created"] = True
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
