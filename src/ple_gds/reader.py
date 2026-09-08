"""Strict cuFile/P2PDMA PLE reader orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .compact import load_current_compact, plan_page_requests


class GdsPleReader:
    """Read selected rows with strict cuFile and gather them on the GPU."""

    def __init__(self, artifact: str | os.PathLike[str], *, component: str,
                 max_staging_bytes: int = 64 * 1024 * 1024,
                 max_output_bytes: int = 64 * 1024 * 1024,
                 offline_integrity_audit: bool = False,
                 expected_gpu_uuid: str = "", expected_gpu_bdf: str = "") -> None:
        self.metadata, generation = load_current_compact(
            artifact,
            check_sources=offline_integrity_audit,
            check_data=offline_integrity_audit,
        )
        self.data_path = generation / self.metadata["data_file"]
        self.component = component
        self.max_staging_bytes = int(max_staging_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self.expected_gpu_uuid = expected_gpu_uuid
        self.expected_gpu_bdf = expected_gpu_bdf
        if self.max_staging_bytes <= 0 or self.max_output_bytes <= 0:
            raise ValueError("staging and output limits must be positive")

    def plan(self, row_ids: Any) -> dict[str, Any]:
        plan = plan_page_requests(row_ids, self.metadata, self.component)
        if int(plan["staging_bytes"]) > self.max_staging_bytes:
            raise MemoryError(
                f"request needs {plan['staging_bytes']} staging bytes, limit is {self.max_staging_bytes}"
            )
        if int(plan["requested_bytes"]) > self.max_output_bytes:
            raise MemoryError(
                f"request needs {plan['requested_bytes']} output bytes, limit is {self.max_output_bytes}"
            )
        return plan

    def read_and_compare(self, row_ids: Any, expected: bytes) -> dict[str, Any]:
        plan = self.plan(row_ids)
        if len(expected) != int(plan["requested_bytes"]):
            raise ValueError(f"expected payload has {len(expected)} bytes, need {plan['requested_bytes']}")
        from . import _native
        native = _native.read_and_compare(
            str(self.data_path),
            [(r["file_offset"], r["size"], r["staging_offset"]) for r in plan["ranges"]],
            plan["gather_offsets"],
            int(plan["row_bytes"]),
            expected,
            int(plan["staging_bytes"]),
            self.max_staging_bytes,
            self.expected_gpu_uuid,
            self.expected_gpu_bdf,
        )
        return {**plan, **native, "data_path": str(self.data_path)}

    def read_global_and_compare(self, name: str, expected: bytes) -> dict[str, Any]:
        item = next((value for value in self.metadata["globals"] if value["name"] == name), None)
        if item is None:
            raise KeyError(f"unknown global {name!r}")
        size = int(item["size"])
        if len(expected) != size:
            raise ValueError(f"expected global has {len(expected)} bytes, need {size}")
        alignment = int(self.metadata["alignment"])
        offset = int(item["offset"])
        page = offset // alignment * alignment
        plan = {
            "ranges": [(page, alignment, 0)],
            "gather_offsets": [offset - page],
            "staging_bytes": alignment,
        }
        from . import _native
        return _native.read_and_compare(
            str(self.data_path), plan["ranges"], plan["gather_offsets"], size, expected,
            alignment, self.max_staging_bytes, self.expected_gpu_uuid, self.expected_gpu_bdf,
        )
