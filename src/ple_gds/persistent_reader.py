"""Persistent torch-owned strict GDS reader for production PLE lookups."""

from __future__ import annotations

import array
import operator
import queue
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .compact import load_current_compact


class ChunkedReadRequired(ValueError):
    """The request is valid but must be consumed through ``iter_read``."""


def _canonical_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def _aligned_cuda_uint8(capacity: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    owner = torch.empty(capacity + 4095, dtype=torch.uint8, device=device)
    offset = (-owner.data_ptr()) % 4096
    view = owner[offset:offset + capacity]
    if view.data_ptr() % 4096 or view.numel() != capacity or not view.is_contiguous():
        raise RuntimeError("failed to create a 4096-byte-aligned torch CUDA buffer")
    return owner, view


@dataclass
class ReadMetrics:
    ids_preparation_seconds: float
    planning_seconds: float
    slot_acquire_seconds: float
    native_call_seconds: float
    submission_seconds: float
    io_wait_seconds: float
    metadata_copy_seconds: float
    gather_launch_seconds: float
    gather_gpu_seconds: float
    end_to_end_seconds: float
    requested_rows: int
    requested_bytes: int
    read_bytes: int
    io_count: int
    batch_chunks: int
    mode: str
    slot: int

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class ReadLease:
    """A view into one fixed output slot, owned until explicitly released."""

    def __init__(self, reader: PersistentGdsPleReader, slot: int | None,
                 tensor: torch.Tensor, metrics: ReadMetrics) -> None:
        self._reader = reader
        self.slot = slot
        self.tensor = tensor
        self.metrics = metrics
        self._released = slot is None

    @property
    def released(self) -> bool:
        return self._released

    def release(self, stream: torch.cuda.Stream | None = None) -> None:
        if self._released:
            return
        assert self.slot is not None
        self._reader.release(self.slot, stream)
        self._released = True

    def __enter__(self) -> ReadLease:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class PersistentGdsPleReader:
    """Bounded persistent NVMe-to-GPU reader with fixed torch CUDA slots."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        component: str = "ngram_embedding",
        expected_gpu_uuid: str,
        expected_gpu_bdf: str,
        slot_count: int = 3,
        staging_bytes: int = 32 * 1024 * 1024,
        output_bytes: int = 32 * 1024 * 1024,
        batch_capacity: int = 128,
        worker_count: int = 16,
        pin_workers: bool = False,
        poll_mode: bool = False,
        max_gap_pages: int = 0,
        max_range_bytes: int = 1024 * 1024,
        cache_capacity_rows: int = 0,
        enable_diagnostic_batch: bool = False,
        enable_diagnostic_async: bool = False,
        offline_integrity_audit: bool = False,
        external_output_slots: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if slot_count < 1 or slot_count > 32:
            raise ValueError("slot_count must be in [1, 32]")
        if staging_bytes <= 0 or output_bytes <= 0:
            raise ValueError("slot capacities must be positive")
        if cache_capacity_rows != 0:
            raise NotImplementedError("GPU row cache is reserved but not implemented; use cache_capacity_rows=0")
        if worker_count < 1 or worker_count > 128:
            raise ValueError("worker_count must be in [1, 128]")
        if enable_diagnostic_batch and worker_count < slot_count:
            raise ValueError("diagnostic Batch requires one independent worker file handle per slot")
        if max_gap_pages < 0 or max_gap_pages > 16:
            raise ValueError("max_gap_pages must be in [0, 16]")
        if max_range_bytes < 4096 or max_range_bytes > 1024 * 1024 or max_range_bytes % 4096:
            raise ValueError("max_range_bytes must be a 4 KiB multiple in [4 KiB, 1 MiB]")
        self.metadata, generation = load_current_compact(
            artifact,
            check_sources=offline_integrity_audit,
            check_data=offline_integrity_audit,
        )
        self.data_path = generation / self.metadata["data_file"]
        self.component = component
        layout = next((item for item in self.metadata["components"] if item["name"] == component), None)
        if layout is None:
            raise KeyError(f"unknown component {component!r}")
        self.row_bytes = int(layout["row_bytes"])
        self.alignment = int(self.metadata["alignment"])
        self.row_count = int(self.metadata["row_count"])
        self.base_offset = int(layout["base_offset"])
        self.block_rows = int(self.metadata["block_rows"])
        self.block_stride = int(layout["block_stride"])
        self.staging_bytes = int(staging_bytes)
        self.output_bytes = int(output_bytes)
        self.cache_capacity_rows = cache_capacity_rows
        self.enable_diagnostic_batch = enable_diagnostic_batch
        self.enable_diagnostic_async = enable_diagnostic_async
        self.worker_count = int(worker_count)
        self.pin_workers = bool(pin_workers)
        self.poll_mode = bool(poll_mode)
        self.max_gap_pages = int(max_gap_pages)
        self.max_range_bytes = int(max_range_bytes)
        self.expected_gpu_uuid = expected_gpu_uuid
        self.expected_gpu_bdf = expected_gpu_bdf
        self._closed = False
        self._native: Any = None
        self._active_slots: set[int] = set()
        self._slot_lock = threading.Lock()
        self._free_slots: queue.Queue[int] = queue.Queue(maxsize=slot_count)

        target_index = self._find_visible_device(expected_gpu_uuid)
        torch.cuda.set_device(target_index)
        self.device = torch.device("cuda", target_index)
        torch.cuda.init()
        self._staging_owners: list[torch.Tensor] = []
        self._staging_slots: list[torch.Tensor] = []
        self._output_owners: list[torch.Tensor] = []
        self._output_slots: list[torch.Tensor] = []
        if external_output_slots is not None and len(external_output_slots) != slot_count:
            raise ValueError(
                "external output slot count must match slot_count: "
                f"{len(external_output_slots)} != {slot_count}"
            )
        for slot in range(slot_count):
            staging_owner, staging = _aligned_cuda_uint8(self.staging_bytes, self.device)
            self._staging_owners.append(staging_owner)
            self._staging_slots.append(staging)
            if external_output_slots is None:
                output_owner, output = _aligned_cuda_uint8(
                    self.output_bytes, self.device
                )
                self._output_owners.append(output_owner)
            else:
                external = external_output_slots[slot]
                if (
                    not isinstance(external, torch.Tensor)
                    or external.device != self.device
                    or not external.is_contiguous()
                ):
                    raise ValueError(
                        f"external output slot {slot} must be contiguous on {self.device}"
                    )
                output = external.view(torch.uint8).reshape(-1)
                if output.numel() < self.output_bytes:
                    raise ValueError(
                        f"external output slot {slot} is too small: "
                        f"{output.numel()} < {self.output_bytes}"
                    )
                output = output[: self.output_bytes]
                if output.data_ptr() % 4096:
                    raise ValueError(
                        f"external output slot {slot} is not 4096-byte aligned"
                    )
            self._output_slots.append(output)
            self._free_slots.put_nowait(slot)

        # A densely repeated request can fit many more output rows than random pages.
        max_spans = self.output_bytes // self.row_bytes
        if max_spans <= 0:
            raise ValueError("output slot is smaller than one row")
        from . import _persistent_native
        self._native = _persistent_native.PersistentReader(
            str(self.data_path),
            [tensor.data_ptr() for tensor in self._staging_slots],
            self.staging_bytes,
            [tensor.data_ptr() for tensor in self._output_slots],
            self.output_bytes,
            expected_gpu_uuid,
            expected_gpu_bdf,
            max_spans,
            batch_capacity,
            self.worker_count,
            self.pin_workers,
            self.poll_mode,
        )
        self.global_scale = self._load_global_scale()

    @staticmethod
    def _find_visible_device(expected_uuid: str) -> int:
        wanted = _canonical_uuid(expected_uuid)
        for index in range(torch.cuda.device_count()):
            actual = str(torch.cuda.get_device_properties(index).uuid).lower()
            if actual == wanted:
                return index
        raise RuntimeError(f"target GPU {expected_uuid} is not visible to PyTorch")

    @staticmethod
    def _prepare_ids(row_ids: Sequence[int] | Iterable[int] | torch.Tensor) -> tuple[Any, int]:
        if isinstance(row_ids, torch.Tensor):
            if row_ids.is_cuda:
                raise TypeError("production PLE row IDs must be prepared on CPU before lookup")
            if row_ids.ndim != 1:
                raise TypeError("production PLE row ID tensor must be one-dimensional")
            if row_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("production PLE row ID tensor must be signed int32 or int64")
            if not row_ids.is_contiguous():
                raise TypeError("production PLE row ID tensor must be contiguous")
            return row_ids.numpy(), row_ids.numel()
        if isinstance(row_ids, array.array):
            if row_ids.typecode not in ("i", "l", "q") or row_ids.itemsize not in (4, 8):
                raise TypeError("row ID array must contain signed int32 or int64 values")
            return row_ids, len(row_ids)
        copied = array.array("q", (operator.index(value) for value in row_ids))
        return copied, len(copied)

    def _load_global_scale(self) -> torch.Tensor:
        item = next((entry for entry in self.metadata.get("globals", [])
                     if entry["name"] == "weight_scale"), None)
        if item is None or int(item["size"]) != 2 or item.get("dtype") != "BF16":
            raise ValueError("artifact must contain one BF16 weight_scale value")
        offset = int(item["offset"])
        page = offset // self.alignment * self.alignment
        stream = torch.cuda.current_stream(self.device)
        native = self._native.read(
            0,
            [(page, self.alignment, 0)],
            [offset - page],
            2,
            2,
            stream.cuda_stream,
            "sync",
            False,
        )
        del native
        scale = self._output_slots[0][:2].view(torch.bfloat16).clone()
        self._native.release(0, stream.cuda_stream)
        return scale

    def _make_metrics(self, *, ids_seconds: float, acquire_seconds: float,
                      native_seconds: float,
                      native: dict[str, Any], total_seconds: float) -> ReadMetrics:
        return ReadMetrics(
            ids_preparation_seconds=ids_seconds,
            planning_seconds=float(native["planning_seconds"]),
            slot_acquire_seconds=acquire_seconds + float(native["slot_wait_seconds"]),
            native_call_seconds=native_seconds,
            submission_seconds=float(native["submission_seconds"]),
            io_wait_seconds=float(native["io_wait_seconds"]),
            metadata_copy_seconds=float(native["metadata_seconds"]),
            gather_launch_seconds=float(native["gather_launch_seconds"]),
            gather_gpu_seconds=float(native["gather_gpu_seconds"]),
            end_to_end_seconds=total_seconds,
            requested_rows=int(native["requested_count"]),
            requested_bytes=int(native["requested_bytes"]),
            read_bytes=int(native["bytes_read"]),
            io_count=int(native["io_count"]),
            batch_chunks=int(native["batch_chunks"]),
            mode=str(native["mode"]),
            slot=int(native["slot"]),
        )

    def _read_prepared(self, ids: Any, ids_count: int, *, mode: str, stream: torch.cuda.Stream,
                       measure_gpu: bool, ids_seconds: float = 0.0,
                       output_offset_bytes: int = 0) -> ReadLease:
        if mode == "batch" and not self.enable_diagnostic_batch:
            raise RuntimeError("cuFile Batch is diagnostic-only on this platform; use persistent sync")
        if mode == "async" and not self.enable_diagnostic_async:
            raise RuntimeError("cuFile Stream Async is diagnostic-only; use sync_mt")
        if mode == "async" and not measure_gpu:
            raise ValueError("diagnostic async requires measure_gpu=True for completion validation")
        total_start = time.perf_counter()
        requested_bytes = ids_count * self.row_bytes
        if output_offset_bytes < 0:
            raise ValueError("output byte offset must be non-negative")
        if output_offset_bytes > self.output_bytes or requested_bytes > (
            self.output_bytes - output_offset_bytes
        ):
            raise ChunkedReadRequired("request exceeds a fixed slot; consume it with iter_read()")
        if not ids_count:
            tensor = self._output_slots[0][:0].view(torch.float8_e4m3fn).reshape(0, self.row_bytes)
            metrics = ReadMetrics(ids_seconds, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, time.perf_counter() - total_start, 0, 0, 0, 0, 0,
                                  mode, -1)
            return ReadLease(self, None, tensor, metrics)
        acquire_start = time.perf_counter()
        slot = self._free_slots.get()
        acquire_seconds = time.perf_counter() - acquire_start
        with self._slot_lock:
            if self._closed:
                self._free_slots.put_nowait(slot)
                raise RuntimeError("persistent reader is closed")
            self._active_slots.add(slot)
        native_start = time.perf_counter()
        try:
            native = self._native.read_ids_at(
                slot,
                ids,
                self.row_count,
                self.base_offset,
                self.block_rows,
                self.block_stride,
                self.row_bytes,
                self.alignment,
                self.max_gap_pages,
                self.max_range_bytes,
                requested_bytes,
                output_offset_bytes,
                stream.cuda_stream,
                mode,
                measure_gpu,
            )
        except Exception as exc:
            with self._slot_lock:
                self._active_slots.discard(slot)
            self._free_slots.put_nowait(slot)
            if "fixed staging slot" in str(exc):
                raise ChunkedReadRequired("request exceeds a fixed slot; consume it with iter_read()") from exc
            raise
        native_seconds = time.perf_counter() - native_start
        tensor = self._output_slots[slot][
            output_offset_bytes : output_offset_bytes + requested_bytes
        ]
        tensor = tensor.view(torch.float8_e4m3fn).reshape(ids_count, self.row_bytes)
        metrics = self._make_metrics(
            ids_seconds=ids_seconds,
            acquire_seconds=acquire_seconds,
            native_seconds=native_seconds,
            native=native,
            total_seconds=time.perf_counter() - total_start + ids_seconds,
        )
        return ReadLease(self, slot, tensor, metrics)

    def read(self, row_ids: Sequence[int] | Iterable[int] | torch.Tensor, *, mode: str = "sync",
             stream: torch.cuda.Stream | None = None, measure_gpu: bool = False) -> ReadLease:
        if self._closed:
            raise RuntimeError("persistent reader is closed")
        ids_start = time.perf_counter()
        ids, ids_count = self._prepare_ids(row_ids)
        ids_seconds = time.perf_counter() - ids_start
        use_stream = stream or torch.cuda.current_stream(self.device)
        return self._read_prepared(ids, ids_count, mode=mode, stream=use_stream,
                                   measure_gpu=measure_gpu, ids_seconds=ids_seconds)

    def read_at(
        self,
        row_ids: Sequence[int] | Iterable[int] | torch.Tensor,
        *,
        output_offset_bytes: int,
        mode: str = "sync",
        stream: torch.cuda.Stream | None = None,
        measure_gpu: bool = False,
    ) -> ReadLease:
        if self._closed:
            raise RuntimeError("persistent reader is closed")
        ids_start = time.perf_counter()
        ids, ids_count = self._prepare_ids(row_ids)
        ids_seconds = time.perf_counter() - ids_start
        use_stream = stream or torch.cuda.current_stream(self.device)
        return self._read_prepared(
            ids,
            ids_count,
            mode=mode,
            stream=use_stream,
            measure_gpu=measure_gpu,
            ids_seconds=ids_seconds,
            output_offset_bytes=output_offset_bytes,
        )

    def iter_read(self, row_ids: Sequence[int] | Iterable[int] | torch.Tensor, *, mode: str = "sync",
                  stream: torch.cuda.Stream | None = None, measure_gpu: bool = False) -> Iterator[ReadLease]:
        yield from self.iter_read_at(
            row_ids,
            output_offset_bytes=0,
            mode=mode,
            stream=stream,
            measure_gpu=measure_gpu,
        )

    def iter_read_at(
        self,
        row_ids: Sequence[int] | Iterable[int] | torch.Tensor,
        *,
        output_offset_bytes: int,
        mode: str = "sync",
        stream: torch.cuda.Stream | None = None,
        measure_gpu: bool = False,
    ) -> Iterator[ReadLease]:
        if self._closed:
            raise RuntimeError("persistent reader is closed")
        ids_start = time.perf_counter()
        ids, ids_count = self._prepare_ids(row_ids)
        ids_seconds = time.perf_counter() - ids_start
        use_stream = stream or torch.cuda.current_stream(self.device)
        # One row touches at most two aligned pages, so every produced chunk fits staging.
        chunk_rows = min(self.output_bytes // self.row_bytes,
                         self.staging_bytes // (2 * self.alignment))
        if chunk_rows <= 0:
            raise ValueError("fixed slots cannot hold one worst-case row")
        if not ids_count:
            yield self._read_prepared(ids, 0, mode=mode, stream=use_stream,
                                      measure_gpu=measure_gpu, ids_seconds=ids_seconds,
                                      output_offset_bytes=output_offset_bytes)
            return
        for start in range(0, ids_count, chunk_rows):
            chunk = ids[start:start + chunk_rows]
            yield self._read_prepared(
                chunk, len(chunk),
                mode=mode,
                stream=use_stream,
                measure_gpu=measure_gpu,
                ids_seconds=ids_seconds if start == 0 else 0.0,
                output_offset_bytes=output_offset_bytes + start * self.row_bytes,
            )

    def rebind_output_slots(
        self, external_output_slots: Sequence[torch.Tensor]
    ) -> None:
        """Replace fixed outputs while all slots are inactive."""

        if self._closed:
            raise RuntimeError("persistent reader is closed")
        if len(external_output_slots) != len(self._output_slots):
            raise ValueError(
                "replacement output slot count must match reader slot count: "
                f"{len(external_output_slots)} != {len(self._output_slots)}"
            )
        with self._slot_lock:
            if self._active_slots:
                raise RuntimeError(
                    f"cannot rebind active reader slots: {sorted(self._active_slots)}"
                )
            replacements: list[torch.Tensor] = []
            capacity: int | None = None
            for slot, external in enumerate(external_output_slots):
                if (
                    not isinstance(external, torch.Tensor)
                    or external.device != self.device
                    or not external.is_contiguous()
                ):
                    raise ValueError(
                        f"replacement output slot {slot} must be contiguous on {self.device}"
                    )
                output = external.view(torch.uint8).reshape(-1)
                if not output.numel() or output.data_ptr() % 4096:
                    raise ValueError(
                        f"replacement output slot {slot} must be nonempty and 4096-byte aligned"
                    )
                if capacity is None:
                    capacity = int(output.numel())
                elif output.numel() != capacity:
                    raise ValueError("replacement output slots must have equal byte capacity")
                replacements.append(output)
            assert capacity is not None
            self._native.rebind_outputs(
                [tensor.data_ptr() for tensor in replacements], capacity
            )
            self.output_bytes = capacity
            self._output_slots = replacements
            self._output_owners.clear()

    def release(self, slot: int, stream: torch.cuda.Stream | None = None) -> None:
        use_stream = stream or torch.cuda.current_stream(self.device)
        with self._slot_lock:
            if slot not in self._active_slots:
                raise RuntimeError(f"slot {slot} is not owned by a consumer")
            self._native.release(slot, use_stream.cuda_stream)
            self._active_slots.remove(slot)
        self._free_slots.put_nowait(slot)

    def stats(self) -> dict[str, Any]:
        result = dict(self._native.stats())
        with self._slot_lock:
            result["active_slots"] = len(self._active_slots)
        result.update(
            staging_bytes_per_slot=self.staging_bytes,
            output_bytes_per_slot=self.output_bytes,
            cache_capacity_rows=self.cache_capacity_rows,
            worker_count=self.worker_count,
            pin_workers=self.pin_workers,
            poll_mode=self.poll_mode,
            max_gap_pages=self.max_gap_pages,
            max_range_bytes=self.max_range_bytes,
            diagnostic_batch_enabled=self.enable_diagnostic_batch,
            diagnostic_async_enabled=self.enable_diagnostic_async,
            generation_id=self.metadata["generation_id"],
            data_path=str(self.data_path),
        )
        return result

    def close(self) -> None:
        with self._slot_lock:
            if self._closed:
                return
            if self._active_slots:
                raise RuntimeError(f"cannot close with active slots: {sorted(self._active_slots)}")
            self._closed = True
        self._native.close()
        self.global_scale = None
        self._staging_slots.clear()
        self._staging_owners.clear()
        self._output_slots.clear()
        self._output_owners.clear()

    def __del__(self) -> None:
        native = getattr(self, "_native", None)
        if native is None:
            return
        try:
            native.close()
        except Exception:
            pass

    def __enter__(self) -> PersistentGdsPleReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
