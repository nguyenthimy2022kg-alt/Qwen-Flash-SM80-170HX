#!/usr/bin/env python3
"""Create or dry-run an aligned PLE artifact for a future cuFile backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ple_gds.compact import convert_compact, slice_source_spec  # noqa: E402
from ple_gds.manifest import convert, load_safetensors_source_spec, load_source_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-spec", help="versioned source-spec JSON")
    source.add_argument("--checkpoint-index", help="model.safetensors.index.json")
    parser.add_argument("--mapping", help="tensor mapping JSON, required with --checkpoint-index")
    parser.add_argument("--output", required=True, help="new derived artifact directory")
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--compact", action="store_true", help="emit compact arithmetic metadata")
    parser.add_argument("--max-rows", type=int, help="build a prefix-only test artifact")
    parser.add_argument("--hash-source", action="store_true", help="record full source SHA-256 (costly for large files)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_index and not args.mapping:
        parser.error("--mapping is required with --checkpoint-index")
    spec = (load_source_spec(args.source_spec) if args.source_spec else
            load_safetensors_source_spec(args.checkpoint_index, args.mapping))
    if args.max_rows is not None:
        spec = slice_source_spec(spec, args.max_rows)
    builder = convert_compact if args.compact else convert
    manifest = builder(spec, args.output, block_rows=args.block_rows, alignment=args.alignment,
                       include_source_hash=args.hash_source, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
