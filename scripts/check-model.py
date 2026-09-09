#!/usr/bin/env python3
"""检查固定模型版本、分片完整性与 PLE/MTP 文件头，不加载权重至内存或 GPU。"""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ple_gds.manifest import _parse_safetensors_header, _KNOWN_ITEMSIZE


def inspect_checkpoint(model, source):
    model = Path(model)
    for filename, expected in source["metadata_sha256"].items():
        actual = hashlib.sha256((model / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{filename} 与固定模型版本不一致；请使用 config/model-source.json 指定的 revision")
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    files = defaultdict(list)
    for key, filename in weights.items():
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"索引中的分片路径无效：{filename}")
        files[filename].append(key)
    total_bytes = 0
    mtp_count = 0
    ple_parts = {}
    scales = []
    for filename, names in files.items():
        path = model / filename
        header, start = _parse_safetensors_header(path)
        size = path.stat().st_size
        total_bytes += size
        for name in names:
            entry = header.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"{filename} 缺少索引张量 {name}")
            offsets = entry.get("data_offsets")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
                raise ValueError(f"{filename}: {name} data_offsets 必须为两个整数")
            begin, end = offsets
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError(f"{filename}: {name} shape 无效")
            if not isinstance(dtype, str) or dtype not in _KNOWN_ITEMSIZE:
                raise ValueError(f"{filename}: {name} dtype 无效或不支持：{dtype}")
            expected = math.prod(shape) * _KNOWN_ITEMSIZE[dtype]
            if begin < 0 or end - begin != expected or start + end > size:
                raise ValueError(f"{filename}: {name} 长度无效或文件未下载完整")
            if name.startswith("mtp."):
                mtp_count += 1
                if dtype != "BF16":
                    raise ValueError(f"MTP 张量必须为 BF16：{name}")
            if "ngram_embedding.shard_" in name and name.endswith(".weight"):
                part = int(name.rsplit("shard_", 1)[1].split(".", 1)[0])
                if part in ple_parts or dtype != source["ple_dtype"] or len(shape) != 2 or shape[1] != source["ple_columns"]:
                    raise ValueError(f"PLE 分片格式不符：{name}")
                ple_parts[part] = shape[0]
            if name.endswith("ngram_embedding.weight_scale"):
                scales.append((dtype, expected))
    if len(files) != source["indexed_safetensors_files"] or total_bytes != source["indexed_safetensors_bytes"]:
        raise ValueError("模型分片数量或文件总大小与固定版本不符")
    if sorted(ple_parts) != list(range(128)) or sum(ple_parts.values()) != source["ple_rows"]:
        raise ValueError("PLE 必须为 128 分片，且总行数与固定版本一致")
    if scales != [("BF16", 2)] or mtp_count != source["mtp_tensors"]:
        raise ValueError("PLE scale 或 MTP 张量不符合固定版本")
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (model / filename).is_file():
            raise FileNotFoundError(f"缺少推理所需文件：{filename}")
    return {"revision": source["revision"], "indexed_files": len(files),
            "indexed_bytes": total_bytes, "ple_rows": sum(ple_parts.values()),
            "mtp_tensors": mtp_count, "check": "metadata_and_headers",
            "weight_payload_sha256_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((ROOT / "config/model-source.json").read_text())
    print(json.dumps(inspect_checkpoint(args.model, source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
