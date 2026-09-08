"""Bounded MRV2 runner bridge for strict GDS-backed Qwen PLE.

The production path in this module handles token metadata only on CPU.  PLE
payload bytes move from NVMe to fixed CUDA buffers through the accepted
PersistentGdsPleReader sync_mt path.  Installation is opt-in so the vendor
runtime remains unchanged unless VLLM_PLE_GDS_RUNNER=1 is set.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm_ple_gds import _lookup_into, plan_ngram_row_ids

logger = init_logger(__name__)

_VENDOR_WARMUP_REQUEST_ID = re.compile(r"_warmup_[0-9]+_")


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _mtp_enabled(handler: Any) -> bool:
    """Return whether the PP handler owns an active speculative path.

    ``num_speculative_steps == 0`` is the runtime representation of
    ``speculative_config=None``.  Keep this gate on the handler so every MTP
    bridge operation is a true no-op for ordinary decode.
    """

    return bool(getattr(handler, "_ple_mtp_enabled", False))


def _is_vendor_warmup_req_id(req_id: str) -> bool:
    return _VENDOR_WARMUP_REQUEST_ID.fullmatch(str(req_id)) is not None


def _are_vendor_warmup_req_ids(req_ids: Sequence[str]) -> bool:
    values = tuple(str(req_id) for req_id in req_ids)
    return bool(values) and all(_is_vendor_warmup_req_id(req_id) for req_id in values)


def _is_vendor_warmup_scheduler_output(scheduler_output: Any) -> bool:
    """Recognize only the scheduler-realistic requests built by warmup_kernels."""

    req_ids: list[str] = []
    req_ids.extend(
        str(item.req_id)
        for item in getattr(scheduler_output, "scheduled_new_reqs", ())
    )
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids.extend(str(req_id) for req_id in getattr(cached, "req_ids", ()))
    for name in ("num_scheduled_tokens", "scheduled_spec_decode_tokens"):
        values = getattr(scheduler_output, name, None)
        if values:
            req_ids.extend(str(req_id) for req_id in values)
    req_ids.extend(
        str(req_id) for req_id in getattr(scheduler_output, "finished_req_ids", ())
    )
    return _are_vendor_warmup_req_ids(req_ids)


def _distribution_seconds(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return (ordered[lower] * (1 - weight) + ordered[upper] * weight) * 1000

    return {
        "samples": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


# Reader phase fields are deliberately kept at the IPC boundary.  The native
# reader already measures these values; copying the small dictionaries here
# lets a diagnostic run explain the full GDS-read interval without changing
# the production reader's scheduling or completion semantics.
_READER_PHASE_TIME_FIELDS = (
    "ids_preparation_seconds",
    "planning_seconds",
    "slot_acquire_seconds",
    "native_call_seconds",
    "submission_seconds",
    "io_wait_seconds",
    "metadata_copy_seconds",
    "gather_launch_seconds",
    "gather_gpu_seconds",
    "end_to_end_seconds",
)
_READER_PHASE_INT_FIELDS = (
    "requested_rows",
    "requested_bytes",
    "read_bytes",
    "io_count",
    "batch_chunks",
)


def _aggregate_lease_metrics(lease_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one IPC command's leases while retaining no tensor payload."""

    aggregate: dict[str, Any] = {"lease_count": len(lease_metrics)}
    for name in _READER_PHASE_TIME_FIELDS:
        aggregate[name] = sum(float(item.get(name, 0.0)) for item in lease_metrics)
    for name in _READER_PHASE_INT_FIELDS:
        aggregate[name] = sum(int(item.get(name, 0)) for item in lease_metrics)
    if lease_metrics:
        # All chunks of one command use the same mode; keep it useful in the
        # report without assuming a particular reader implementation.
        aggregate["mode"] = str(lease_metrics[0].get("mode", ""))
        aggregate["slots"] = [int(item.get("slot", -1)) for item in lease_metrics]
    else:
        aggregate["mode"] = ""
        aggregate["slots"] = []
    return aggregate


_IPC_PHASE_FIELDS = (
    "dispatch_to_child_seconds",
    "child_stream_completion_seconds",
    "child_complete_to_parent_seconds",
    "gpu_semaphore_wait_seconds",
    "post_wait_gather_signal_gpu_seconds",
)


def _phase_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return p50/p95/p99 for child/reader phase samples."""

    result: dict[str, Any] = {}
    for name in _IPC_PHASE_FIELDS:
        result[name] = _distribution_seconds(
            [float(sample[name]) for sample in samples if sample.get(name) is not None]
        )
    reader_samples = [
        sample.get("reader_phase", sample.get("reader", {}))
        for sample in samples
        if isinstance(sample.get("reader_phase", sample.get("reader", {})), Mapping)
    ]
    for name in _READER_PHASE_TIME_FIELDS:
        result[f"reader_{name}"] = _distribution_seconds(
            [float(sample.get(name, 0.0)) for sample in reader_samples]
        )
    # ``submission_seconds`` is the native reader's host-side cuFile
    # submission interval. Keep a plainly named alias in the diagnostic
    # report so it is not confused with parent-to-child IPC dispatch.
    result["reader_host_submission_seconds"] = dict(
        result["reader_submission_seconds"]
    )
    for name in ("io_count", "read_bytes", "lease_count"):
        values = [int(sample.get(name, 0)) for sample in reader_samples]
        result[f"reader_{name}"] = {
            "samples": len(values),
            "total": sum(values),
            "p50": sorted(values)[(len(values) - 1) // 2] if values else 0,
            "p95": (
                sorted(values)[min(len(values) - 1, int((len(values) - 1) * 0.95))]
                if values
                else 0
            ),
        }
    return result


def _merge_ipc_phase_message(
    previous: Mapping[str, Any] | None,
    message: Mapping[str, Any],
    *,
    dispatch_at: float,
    callback_at: float,
) -> dict[str, Any]:
    """Build a small parent-side diagnostic record from one child callback."""

    result = dict(previous or {})
    started_at = message.get("started_at")
    if started_at is not None:
        result["child_started_at"] = float(started_at)
        result["dispatch_to_child_seconds"] = max(
            0.0, float(started_at) - float(dispatch_at)
        )
    if message.get("lease_metrics") is not None:
        result["lease_metrics"] = [
            dict(item)
            for item in message.get("lease_metrics", ())
            if isinstance(item, Mapping)
        ]
    if message.get("reader_phase") is not None:
        reader_phase = message.get("reader_phase")
        if isinstance(reader_phase, Mapping):
            result["reader"] = dict(reader_phase)
    if message.get("phase_timing") is not None:
        phase = message.get("phase_timing")
        if isinstance(phase, Mapping):
            result.update(
                {
                    name: (
                        None
                        if phase.get(name) is None
                        else float(phase.get(name))
                    )
                    for name in _IPC_PHASE_FIELDS
                    if name in phase
                }
            )
    if message.get("type") == "completed":
        completed_at = message.get("completed_at")
        if completed_at is not None:
            completed_value = float(completed_at)
            result["child_completed_at"] = completed_value
            result["child_complete_to_parent_seconds"] = max(
                0.0, float(callback_at) - completed_value
            )
            if started_at is not None:
                result["child_stream_completion_seconds"] = max(
                    0.0, completed_value - float(started_at)
                )
    return result


class TicketError(RuntimeError):
    """Fail-closed runner ticket error."""


class MailboxFull(TicketError):
    pass


class LateSampleMetadata(TicketError):
    pass


@dataclasses.dataclass
class ShadowRequest:
    req_id: str
    generation: int
    tokens: list[int]
    num_computed: int


@dataclasses.dataclass(frozen=True)
class BatchIdentity:
    epoch: int
    step_id: int
    req_ids: tuple[str, ...]
    generations: tuple[int, ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    valid_tokens: int
    padded_tokens: int

    @property
    def digest(self) -> str:
        payload = repr(dataclasses.astuple(self)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class CpuBatchPlan:
    identity: BatchIdentity
    input_ids: torch.Tensor
    query_start_loc: torch.Tensor
    ngram_context: torch.Tensor


class CpuTokenShadow:
    """Bounded-by-request CPU token history derived from scheduler metadata."""

    def __init__(self, *, eos_token_id: int, context_len: int) -> None:
        if context_len < 1:
            raise ValueError("context_len must be positive")
        self.eos_token_id = int(eos_token_id)
        self.context_len = int(context_len)
        self._requests: dict[str, ShadowRequest] = {}
        self._next_generation = 1
        self.epoch = 1
        self._lock = threading.RLock()

    def add_or_replace(
        self, req_id: str, tokens: Sequence[int], num_computed: int
    ) -> ShadowRequest:
        with self._lock:
            values = [int(token) for token in tokens]
            if num_computed < 0 or num_computed > len(values):
                raise TicketError(
                    f"invalid computed position for {req_id}: "
                    f"{num_computed}/{len(values)}"
                )
            request = ShadowRequest(
                req_id=req_id,
                generation=self._next_generation,
                tokens=values,
                num_computed=int(num_computed),
            )
            self._next_generation += 1
            self._requests[req_id] = request
            return request

    def remove(self, req_id: str) -> bool:
        with self._lock:
            removed = self._requests.pop(req_id, None) is not None
            if removed:
                self.epoch += 1
            return removed

    def update_computed(self, req_id: str, num_computed: int) -> None:
        with self._lock:
            request = self._require(req_id)
            if num_computed < 0 or num_computed > len(request.tokens):
                raise TicketError(
                    f"computed position outside shadow for {req_id}: "
                    f"{num_computed}/{len(request.tokens)}"
                )
            request.num_computed = int(num_computed)

    def append_sampled(
        self,
        req_ids: Sequence[str],
        generations: Sequence[int],
        sampled_tokens: np.ndarray,
        num_sampled: np.ndarray,
        include_mask: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            if sampled_tokens.ndim != 2 or num_sampled.ndim != 1:
                raise TicketError("sample metadata shape mismatch")
            if len(req_ids) != len(generations) or len(req_ids) != len(num_sampled):
                raise TicketError("sample metadata batch mismatch")
            if include_mask is None:
                include_mask = np.ones(len(req_ids), dtype=np.bool_)
            for row, (req_id, generation) in enumerate(zip(req_ids, generations)):
                if not bool(include_mask[row]):
                    continue
                request = self._require(req_id)
                if request.generation != int(generation):
                    raise TicketError(
                        f"stale sampled metadata for {req_id}: "
                        f"{generation} != {request.generation}"
                    )
                count = int(num_sampled[row])
                if count < 0 or count > sampled_tokens.shape[1]:
                    raise TicketError(f"invalid sampled count for {req_id}: {count}")
                request.tokens.extend(
                    int(value) for value in sampled_tokens[row, :count]
                )

    def plan(
        self,
        *,
        step_id: int,
        req_ids: Sequence[str],
        starts: Sequence[int],
        lengths: Sequence[int],
        drafts: Mapping[str, Sequence[int]],
        padded_tokens: int,
    ) -> CpuBatchPlan:
        with self._lock:
            if len(req_ids) != len(starts) or len(req_ids) != len(lengths):
                raise TicketError("batch metadata lengths differ")
            packed: list[int] = []
            contexts: list[list[int]] = []
            query_start = [0]
            generations: list[int] = []
            for req_id, raw_start, raw_length in zip(req_ids, starts, lengths):
                request = self._require(req_id)
                start = int(raw_start)
                length = int(raw_length)
                if start < 0 or length < 0:
                    raise TicketError(f"negative slice for {req_id}: {start}+{length}")
                draft = [int(value) for value in drafts.get(req_id, ())]
                virtual = request.tokens + draft
                end = start + length
                if end > len(virtual):
                    raise TicketError(
                        f"shadow underflow for {req_id}: need [{start}:{end}], "
                        f"have {len(request.tokens)} accepted + {len(draft)} draft"
                    )
                packed.extend(virtual[start:end])
                prefix = request.tokens[max(0, start - self.context_len) : start]
                context = (
                    [self.eos_token_id] * (self.context_len - len(prefix)) + prefix
                )
                contexts.append(context)
                query_start.append(len(packed))
                generations.append(request.generation)
            valid_tokens = len(packed)
            if padded_tokens < valid_tokens:
                raise TicketError(
                    f"padded token count {padded_tokens} < valid {valid_tokens}"
                )
            identity = BatchIdentity(
                epoch=self.epoch,
                step_id=int(step_id),
                req_ids=tuple(req_ids),
                generations=tuple(generations),
                starts=tuple(int(value) for value in starts),
                lengths=tuple(int(value) for value in lengths),
                valid_tokens=valid_tokens,
                padded_tokens=int(padded_tokens),
            )
            return CpuBatchPlan(
                identity=identity,
                input_ids=torch.tensor(packed, dtype=torch.int64),
                query_start_loc=torch.tensor(query_start, dtype=torch.int64),
                ngram_context=torch.tensor(contexts, dtype=torch.int64),
            )

    def generation(self, req_id: str) -> int:
        with self._lock:
            return self._require(req_id).generation

    def token_count(self, req_id: str, generation: int) -> int:
        with self._lock:
            request = self._require(req_id)
            if request.generation != int(generation):
                raise TicketError(
                    f"stale token-count query for {req_id}: "
                    f"{generation} != {request.generation}"
                )
            return len(request.tokens)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                req_id: {
                    "generation": request.generation,
                    "num_computed": request.num_computed,
                    "tokens": list(request.tokens),
                }
                for req_id, request in self._requests.items()
            }

    def _require(self, req_id: str) -> ShadowRequest:
        try:
            return self._requests[req_id]
        except KeyError as exc:
            raise TicketError(f"unknown request in CPU shadow: {req_id}") from exc


def _shadow_add_requests(shadow: CpuTokenShadow, scheduler_output: Any) -> None:
    for item in scheduler_output.scheduled_new_reqs:
        if _is_vendor_warmup_req_id(item.req_id):
            continue
        shadow.add_or_replace(
            item.req_id, item.prefill_token_ids, item.num_computed_tokens
        )


def _shadow_update_requests(shadow: CpuTokenShadow, scheduler_output: Any) -> None:
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
        if _is_vendor_warmup_req_id(req_id):
            continue
        shadow.update_computed(req_id, int(computed))


class SlotState(str, Enum):
    FREE = "free"
    QUEUED = "queued"
    PRODUCING = "producing"
    READY = "ready"
    CONSUMING = "consuming"
    RELEASING = "releasing"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class PrefetchIdentity:
    sequence: int
    req_id: str
    generation: int
    start: int


@dataclasses.dataclass
class TicketSlot:
    index: int
    output: torch.Tensor
    producer_done: torch.cuda.Event
    consumer_done: torch.cuda.Event
    done_flag: torch.Tensor
    submission_ready: threading.Event
    state: SlotState = SlotState.FREE
    ticket: CpuBatchPlan | None = None
    prefetch: PrefetchIdentity | None = None
    discard: bool = False
    row_ids: torch.Tensor | None = None
    error: BaseException | None = None
    submitted_at: float = 0.0
    # Set immediately before the parent sends an IPC command.  It is paired
    # with the child process' monotonic ``started_at`` for diagnostic latency.
    dispatch_at: float = 0.0
    producer_started_at: float = 0.0
    producer_finished_at: float = 0.0
    consumer_acquired_at: float = 0.0
    done_value: int = 0
    producer_completed: bool = False
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


LookupFn = Callable[[Any, torch.Tensor, torch.Tensor], dict[str, Any]]


def _stream_wait_value32(stream: torch.cuda.Stream, flag: torch.Tensor, value: int) -> None:
    """Enqueue a cross-CUDA-context wait without synchronizing the model thread."""

    from cuda.bindings import driver as cuda_driver
    from cuda.bindings.driver import CUstreamWaitValue_flags

    # Python worker threads do not inherit the caller's current CUDA context.
    # Selecting the device here makes the primary context current before the
    # direct Driver API call while remaining a no-op on the model thread.
    with torch.cuda.device(flag.device):
        torch.cuda.set_device(flag.device)
        result = cuda_driver.cuStreamWaitValue32(
            cuda_driver.CUstream(stream.cuda_stream),
            cuda_driver.CUdeviceptr(flag.data_ptr()),
            int(value),
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        )
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise TicketError(f"cuStreamWaitValue32 failed: {error}")


def _stream_write_value32(stream: torch.cuda.Stream, flag: torch.Tensor, value: int) -> None:
    from cuda.bindings import driver as cuda_driver

    with torch.cuda.device(flag.device):
        torch.cuda.set_device(flag.device)
        result = cuda_driver.cuStreamWriteValue32(
            cuda_driver.CUstream(stream.cuda_stream),
            cuda_driver.CUdeviceptr(flag.data_ptr()),
            int(value),
            0,
        )
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"cuStreamWriteValue32 failed: {error}")


def _event_elapsed_seconds(start: torch.cuda.Event, end: torch.cuda.Event) -> float | None:
    """Read an already-completed CUDA event pair without synchronizing."""

    try:
        return float(start.elapsed_time(end)) / 1000.0
    except BaseException:
        # Event timing is diagnostic-only.  A driver that cannot timestamp a
        # wait-value command must not change the normal producer result path.
        return None


def _ipc_gds_producer_main(
    output_slots: list[torch.Tensor],
    done_flags: list[torch.Tensor],
    command_conn: Any,
    result_conn: Any,
    artifact: str,
    gpu_uuid: str,
    gpu_bdf: str,
    worker_count: int,
    direct_output: bool,
) -> None:
    """Process entry point for the async-only strict GDS producer."""

    ipc_log_path = os.getenv("VLLM_PLE_GDS_IPC_CUFILE_LOGFILE_PATH", "").strip()
    if ipc_log_path:
        os.environ["CUFILE_LOGFILE_PATH"] = ipc_log_path

    from ple_gds.persistent_reader import ChunkedReadRequired, PersistentGdsPleReader

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream(device=0)
    phase_timing_enabled = _enabled("VLLM_PLE_GDS_PHASE_TIMING")
    # Commands are serialized by command_conn and the stream is synchronized
    # before the next command, so one reusable event triplet is sufficient.
    # Events are recorded only when explicitly requested by the diagnostic env.
    phase_events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event] | None = None
    if phase_timing_enabled:
        phase_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
    output_capacity = int(output_slots[0].numel() * output_slots[0].element_size())
    if any(
        int(slot.numel() * slot.element_size()) != output_capacity
        for slot in output_slots
    ):
        raise RuntimeError("IPC GDS output slots must have equal byte capacity")
    reader_args = {
        "expected_gpu_uuid": gpu_uuid,
        "expected_gpu_bdf": gpu_bdf,
        "slot_count": len(output_slots),
        "staging_bytes": 32 * 1024 * 1024,
        "worker_count": worker_count,
        "pin_workers": True,
        "poll_mode": True,
        "cache_capacity_rows": 0,
    }
    reader = PersistentGdsPleReader(
        artifact,
        **reader_args,
        output_bytes=output_capacity if direct_output else 32 * 1024 * 1024,
        external_output_slots=output_slots if direct_output else None,
    )
    # A fragmented Prefill can exceed the fixed page-staging plan even when
    # its assembled result fits the final output. Keep one internal scratch
    # reader for that exceptional path; Decode remains a direct cuFile write
    # into the IPC output with no device-to-device payload copy.
    chunk_reader = (
        PersistentGdsPleReader(
            artifact,
            **reader_args,
            output_bytes=32 * 1024 * 1024,
        )
        if direct_output
        else reader
    )

    def reader_stats() -> dict[str, Any]:
        direct_stats = reader.stats()
        if chunk_reader is reader:
            return direct_stats
        fallback_stats = chunk_reader.stats()
        combined = dict(direct_stats)
        for name in ("calls", "releases", "bytes_read", "active_slots"):
            combined[name] = int(direct_stats[name]) + int(fallback_stats[name])
        combined["direct_output_reader"] = direct_stats
        combined["chunk_fallback_reader"] = fallback_stats
        return combined

    result_conn.send({"type": "ready", "reader": reader_stats()})
    try:
        while True:
            command = command_conn.recv()
            if command is None:
                break
            if len(command) == 4:
                slot_index, row_ids, value, valid_tokens = command
                wait_reset = False
                request_id = None
            elif len(command) == 6:
                (
                    slot_index,
                    row_ids,
                    value,
                    valid_tokens,
                    wait_reset,
                    request_id,
                ) = command
            else:
                raise RuntimeError(f"invalid IPC GDS command shape: {len(command)}")
            started = time.perf_counter()
            assembled_bytes = 0
            chunk_count = 0
            lease_metrics: list[dict[str, Any]] = []
            try:
                with torch.cuda.stream(stream):
                    if phase_events is not None:
                        wait_start_event, wait_end_event, signal_end_event = phase_events
                        wait_start_event.record(stream)
                    if wait_reset:
                        _stream_wait_value32(stream, done_flags[slot_index], 0)
                    if phase_events is not None:
                        wait_end_event.record(stream)
                    output_bytes = output_slots[slot_index].view(torch.uint8).reshape(-1)
                    expected_bytes = int(valid_tokens) * int(output_slots[slot_index].shape[1])
                    if expected_bytes > output_bytes.numel():
                        raise RuntimeError(
                            "IPC PLE output capacity exceeded: "
                            f"{expected_bytes} > {output_bytes.numel()}"
                        )

                    def append_lease(lease: Any) -> None:
                        nonlocal assembled_bytes, chunk_count
                        payload = lease.tensor.view(torch.uint8).reshape(-1)
                        chunk_bytes = int(payload.numel())
                        chunk_end = assembled_bytes + chunk_bytes
                        if chunk_end > expected_bytes or chunk_end > output_bytes.numel():
                            raise RuntimeError(
                                "IPC PLE chunk exceeds output boundary: "
                                f"offset={assembled_bytes} chunk={chunk_bytes} "
                                f"expected={expected_bytes} capacity={output_bytes.numel()}"
                            )
                        destination = output_bytes[assembled_bytes:chunk_end]
                        if destination.data_ptr() != payload.data_ptr():
                            destination.copy_(payload, non_blocking=True)
                        assembled_bytes = chunk_end
                        chunk_count += 1
                        if phase_timing_enabled:
                            lease_metrics.append(dict(lease.metrics.as_dict()))

                    try:
                        lease = reader.read(row_ids, mode="sync_mt", stream=stream)
                    except ChunkedReadRequired:
                        for lease in chunk_reader.iter_read(
                            row_ids, mode="sync_mt", stream=stream
                        ):
                            try:
                                append_lease(lease)
                            finally:
                                lease.release(stream)
                    else:
                        try:
                            append_lease(lease)
                        finally:
                            lease.release(stream)

                    if assembled_bytes != expected_bytes:
                        raise RuntimeError(
                            "IPC PLE assembled payload size mismatch: "
                            f"{assembled_bytes} != {expected_bytes}"
                        )
                    _stream_write_value32(stream, done_flags[slot_index], value)
                    if phase_events is not None:
                        signal_end_event.record(stream)
                current_reader_stats = reader_stats()
                submitted_message = {
                    "type": "submitted",
                    "slot": int(slot_index),
                    "value": int(value),
                    "seconds": time.perf_counter() - started,
                    "started_at": started,
                    "chunk_count": chunk_count,
                    "assembled_bytes": assembled_bytes,
                    "request_id": request_id,
                    "direct_output": bool(direct_output),
                    "reader": current_reader_stats,
                }
                if phase_timing_enabled:
                    submitted_message["lease_metrics"] = lease_metrics
                    submitted_message["reader_phase"] = _aggregate_lease_metrics(
                        lease_metrics
                    )
                result_conn.send(submitted_message)
                # Keep the shared output slot alive until the child stream has
                # completed; this is required before a discarded slot can be reused.
                stream.synchronize()
                completed_at = time.perf_counter()
                completed_message = {
                    "type": "completed",
                    "slot": int(slot_index),
                    "value": int(value),
                    "seconds": completed_at - started,
                    "started_at": started,
                    "completed_at": completed_at,
                    "chunk_count": chunk_count,
                    "assembled_bytes": assembled_bytes,
                    "request_id": request_id,
                    "direct_output": bool(direct_output),
                    "reader": reader_stats(),
                }
                if phase_timing_enabled:
                    completed_message["lease_metrics"] = lease_metrics
                    completed_message["reader_phase"] = _aggregate_lease_metrics(
                        lease_metrics
                    )
                    completed_message["phase_timing"] = {
                        "gpu_semaphore_wait_seconds": _event_elapsed_seconds(
                            wait_start_event, wait_end_event
                        ),
                        "post_wait_gather_signal_gpu_seconds": _event_elapsed_seconds(
                            wait_end_event, signal_end_event
                        ),
                    }
                result_conn.send(completed_message)
            except BaseException as exc:
                result_conn.send(
                    {
                        "type": "error",
                        "slot": int(slot_index),
                        "value": int(value),
                        "request_id": request_id,
                        "started_at": started,
                        "error": repr(exc),
                    }
                )
    finally:
        if chunk_reader is not reader:
            chunk_reader.close()
        reader.close()
        output_slots.clear()
        done_flags.clear()
        gc.collect()


class IpcGdsProducer:
    """Dedicated CUDA context for async GDS submission and semaphore signaling."""

    def __init__(
        self,
        *,
        output_slots: list[torch.Tensor],
        done_flags: list[torch.Tensor],
        artifact: str,
        gpu_uuid: str,
        gpu_bdf: str,
        worker_count: int,
        callback: Callable[[dict[str, Any]], None],
        direct_output: bool = False,
    ) -> None:
        self._callback = callback
        self._send_lock = threading.Lock()
        self._closed = False
        context = mp.get_context("spawn")
        self._parent_command, child_command = context.Pipe(duplex=True)
        parent_result, self._child_result = context.Pipe(duplex=False)
        self._result_conn = parent_result
        self._process = context.Process(
            target=_ipc_gds_producer_main,
            args=(
                output_slots,
                done_flags,
                child_command,
                self._child_result,
                artifact,
                gpu_uuid,
                gpu_bdf,
                int(worker_count),
                bool(direct_output),
            ),
            daemon=False,
        )
        current_process = mp.current_process()
        parent_daemon = bool(current_process._config.get("daemon", False))
        if parent_daemon:
            # vLLM marks rank workers as multiprocessing daemons. Python uses
            # that bookkeeping flag to forbid managed children even though this
            # producer is explicitly joined by close() and the container cgroup
            # provides fail-stop cleanup. Relax only around Process.start().
            current_process._config["daemon"] = False
        try:
            self._process.start()
        finally:
            if parent_daemon:
                current_process._config["daemon"] = True
        if not self._result_conn.poll(120):
            self._process.terminate()
            self._process.join(timeout=10)
            raise TicketError("IPC GDS producer startup timed out")
        ready = self._result_conn.recv()
        if ready.get("type") != "ready":
            raise TicketError(f"IPC GDS producer failed startup: {ready}")
        self.ready = ready
        self._result_thread = threading.Thread(
            target=self._result_loop,
            name="ple-gds-ipc-result",
            daemon=True,
        )
        self._result_thread.start()

    def dispatch(
        self,
        slot_index: int,
        row_ids: Sequence[int],
        value: int,
        valid_tokens: int,
        *,
        wait_reset: bool = False,
        request_id: int | None = None,
    ) -> None:
        with self._send_lock:
            if self._closed:
                raise TicketError("IPC GDS producer is closed")
            self._parent_command.send(
                (
                    int(slot_index),
                    [int(row_id) for row_id in row_ids],
                    int(value),
                    int(valid_tokens),
                    bool(wait_reset),
                    request_id,
                )
            )

    def close(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            self._parent_command.send(None)
        self._process.join(timeout=60)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
            raise TicketError("IPC GDS producer failed to stop")
        self._result_thread.join(timeout=10)
        self._parent_command.close()
        self._result_conn.close()

    def _result_loop(self) -> None:
        while True:
            try:
                message = self._result_conn.recv()
            except (EOFError, OSError):
                return
            self._callback(message)


@dataclasses.dataclass(frozen=True)
class InputDrivenRequest:
    sequence: int
    num_reqs: int
    num_tokens: int
    enqueued_at: float


def _aligned_cuda_output(
    rows: int, row_bytes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    capacity = int(rows) * int(row_bytes)
    owner = torch.empty(capacity + 4095, dtype=torch.uint8, device=device)
    offset = (-owner.data_ptr()) % 4096
    raw = owner[offset : offset + capacity]
    if raw.data_ptr() % 4096 or raw.numel() != capacity or not raw.is_contiguous():
        raise TicketError("failed to allocate an aligned input-driven PLE output")
    return owner, raw.view(torch.float8_e4m3fn).reshape(rows, row_bytes)


class InputDrivenGdsPleConnector:
    """Native-style metadata control plane with a strict GDS payload worker.

    Stable MRV2 GPU inputs are copied to fixed pinned buffers on a dedicated
    D2H stream. Only the background thread waits for D2H and plans row IDs.
    The GDS child writes PLE bytes into the final IPC CUDA output, while the
    PLE layer waits on a GPU semaphore instead of blocking the model thread.
    """

    def __init__(
        self,
        *,
        layer: Any,
        max_num_batched_tokens: int,
        max_num_reqs: int,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor,
        use_ipc_process: bool = True,
    ) -> None:
        reader = getattr(layer, "_ple_gds_reader", None)
        self.device = getattr(reader, "device", None)
        if self.device is None:
            raise TicketError("input-driven GDS PLE requires the PP0 strict reader")
        self.layer = layer
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.embedding_dim = int(layer.embedding_dim)
        self.use_ipc_process = bool(use_ipc_process)
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._validate_sources()

        self._input_ids_buf = torch.empty(
            self.max_num_batched_tokens, dtype=torch.int32, device="cpu"
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        ).share_memory_()
        self._ngram_context_buf = torch.empty(
            self.max_num_reqs,
            int(layer.ngram_size) - 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._pinned_input_buffers: list[torch.Tensor] = []
        self._closed = False
        self._error: BaseException | None = None
        self._condition = threading.Condition()
        self._active_sequence: int | None = None
        self._active_tokens = 0
        self._next_sequence = 1
        self._metadata_enqueued_sequence = 0
        self._metadata_handoff_timeout = 60.0
        self._records: dict[int, dict[str, Any]] = {}
        self._ipc_reader_stats: dict[str, Any] | None = None
        self._phase_timing_enabled = _enabled("VLLM_PLE_GDS_PHASE_TIMING")
        self._stats: dict[str, Any] = {
            "launched": 0,
            "dummy": 0,
            "completed": 0,
            "released": 0,
            "input_ready_events": 0,
            "metadata_handoffs": 0,
            "metadata_handoff_host_seconds": 0.0,
            "close_count": 0,
            "errors": 0,
            "input_ready_seconds": deque(maxlen=4096),
            "row_plan_seconds": deque(maxlen=4096),
            "gds_read_seconds": deque(maxlen=4096),
            "ple_consumer_wait_seconds": deque(maxlen=4096),
            "consumer_host_wait_seconds": deque(maxlen=4096),
            "ipc_phase_samples": deque(maxlen=4096),
            "samples": deque(maxlen=4096),
        }
        self._request_queue: queue.Queue[InputDrivenRequest | None] = queue.Queue(
            maxsize=1
        )
        self._request_thread: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None
        self._metrics_stop = threading.Event()
        metrics_path = os.getenv("VLLM_PLE_GDS_RUNNER_METRICS_PATH", "").strip()
        self._metrics_path = Path(metrics_path) if metrics_path else None
        self._metrics_interval = float(
            os.getenv("VLLM_PLE_GDS_RUNNER_METRICS_INTERVAL", "5")
        )
        if self._metrics_interval <= 0:
            raise ValueError("runner metrics interval must be positive")

        from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore

        with torch.cuda.device(self.device):
            self._output_owner, self.output = _aligned_cuda_output(
                self.max_num_batched_tokens, self.embedding_dim, self.device
            )
            self._sem = CpuGpuSemaphore(self.device)
            self._d2h_stream = torch.cuda.Stream(device=self.device)
            self._reader_stream = torch.cuda.Stream(device=self.device)
            self._input_ready_event = torch.cuda.Event()
            self._d2h_done_event = torch.cuda.Event()
            self._error_stream = torch.cuda.Stream(device=self.device)
            self._pin_input_buffers()

        self._reader = None
        self._ipc_producer = None
        if self.use_ipc_process:
            self._ipc_producer = IpcGdsProducer(
                output_slots=[self.output],
                done_flags=[self._sem.flag_tensor],
                artifact=os.environ["VLLM_PLE_GDS_ARTIFACT"],
                gpu_uuid=os.environ["VLLM_PLE_GDS_GPU_UUID"],
                gpu_bdf=os.environ["VLLM_PLE_GDS_GPU_BDF"],
                worker_count=int(os.getenv("VLLM_PLE_GDS_WORKERS", "32")),
                callback=self._handle_ipc_result,
                direct_output=True,
            )
            self._ipc_reader_stats = dict(self._ipc_producer.ready["reader"])
        else:
            if int(getattr(reader, "cache_capacity_rows", -1)) != 0:
                raise TicketError("same-process input-driven reader requires cache_rows=0")
            slot_count = int(reader.stats()["slot_count"])
            reader.rebind_output_slots([self.output] * slot_count)
            self._reader = reader
            self._ipc_reader_stats = dict(reader.stats())
        self._request_thread = threading.Thread(
            target=self._request_loop,
            name="ple-gds-input-driven",
            daemon=True,
        )
        self._request_thread.start()
        if self._metrics_path is not None:
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop,
                name="ple-gds-input-driven-metrics",
                daemon=True,
            )
            self._metrics_thread.start()
        layer._ple_gds_input_connector = self
        transport = "direct IPC output" if self.use_ipc_process else "same-process direct output"
        logger.warning(
            "PLE GDS input-driven control path enabled: pinned D2H metadata, "
            "background row planning, %s, GPU semaphore wait",
            transport,
        )

    def _validate_sources(self) -> None:
        context_len = int(self.layer.ngram_size) - 1
        expected = (
            (
                "input_ids",
                self._input_ids_source,
                1,
                (self.max_num_batched_tokens,),
            ),
            (
                "query_start_loc",
                self._query_start_loc_source,
                1,
                (self.max_num_reqs + 1,),
            ),
            (
                "ngram_context",
                self._ngram_context_source,
                2,
                (self.max_num_reqs, context_len),
            ),
        )
        for name, source, ndim, minimum in expected:
            if (
                not isinstance(source, torch.Tensor)
                or source.device != self.device
                or source.dtype != torch.int32
                or source.ndim != ndim
                or any(actual < required for actual, required in zip(source.shape, minimum))
                or source.shape[1:] != minimum[1:]
            ):
                raise TicketError(
                    f"input-driven PLE {name} source is not a stable compatible buffer"
                )

    @staticmethod
    def _cuda_check(result: Any, operation: str) -> None:
        error = result[0] if isinstance(result, tuple) else result
        if error.value != 0:
            raise TicketError(f"{operation} failed: {error}")

    def _pin_input_buffers(self) -> None:
        from cuda.bindings import driver as cuda_driver

        for buffer in (
            self._input_ids_buf,
            self._query_start_loc_buf,
            self._ngram_context_buf,
        ):
            self._cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    cuda_driver.CU_MEMHOSTREGISTER_PORTABLE,
                ),
                "cuMemHostRegister(input-driven PLE metadata)",
            )
            self._pinned_input_buffers.append(buffer)
            if not buffer.is_pinned():
                raise TicketError("CUDA did not page-lock input-driven PLE metadata")

    def _unpin_input_buffers(self) -> None:
        from cuda.bindings import driver as cuda_driver

        for buffer in reversed(self._pinned_input_buffers):
            try:
                self._cuda_check(
                    cuda_driver.cuMemHostUnregister(buffer.data_ptr()),
                    "cuMemHostUnregister(input-driven PLE metadata)",
                )
            except BaseException:
                logger.exception("failed to unregister input-driven PLE metadata")
        self._pinned_input_buffers.clear()

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise TicketError("input-driven GDS PLE background path failed") from error

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
    ) -> None:
        """Launch after stock GPUModelRunner has finalized stable GPU inputs."""
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        self.raise_if_failed()
        if num_reqs < 1 or num_reqs > self.max_num_reqs:
            raise TicketError(f"input-driven PLE request count is invalid: {num_reqs}")
        if num_tokens < 1 or num_tokens > self.max_num_batched_tokens:
            raise TicketError(f"input-driven PLE token count is invalid: {num_tokens}")
        with self._condition:
            if self._closed:
                raise TicketError("input-driven GDS PLE connector is closed")
            if self._active_sequence is not None:
                raise TicketError("previous input-driven PLE output was not released")
            sequence = self._next_sequence
            self._next_sequence += 1
            now = time.perf_counter()
            self._active_sequence = sequence
            self._active_tokens = int(num_tokens)
            self._records[sequence] = {
                "sequence": sequence,
                "num_reqs": int(num_reqs),
                "num_tokens": int(num_tokens),
                "enqueued_at": now,
                "consumer_enqueued_at": now,
                "dummy": False,
                "released": False,
                "completed": False,
            }
            self._stats["launched"] += 1
        use_stream = torch.cuda.current_stream(self.device)
        self._input_ready_event.record(use_stream)
        with self._condition:
            self._stats["input_ready_events"] += 1
        request = InputDrivenRequest(sequence, int(num_reqs), int(num_tokens), now)
        try:
            self._request_queue.put_nowait(request)
        except queue.Full as exc:
            self._fail(sequence, MailboxFull("input-driven metadata queue is full"))
            raise MailboxFull("input-driven metadata queue is full") from exc
        # The worker must bind this event generation and enqueue all metadata
        # copies before the model thread can reuse the shared event or inputs.
        self._wait_metadata_enqueued(sequence)
        use_stream.wait_event(self._d2h_done_event)

    def _wait_metadata_enqueued(self, sequence: int) -> None:
        started = time.perf_counter()
        deadline = started + self._metadata_handoff_timeout
        failure = None
        with self._condition:
            while True:
                if self._error is not None:
                    raise TicketError("input-driven GDS PLE background path failed") from self._error
                if self._closed:
                    raise TicketError("input-driven GDS PLE connector is closed")
                if self._metadata_enqueued_sequence == sequence:
                    self._stats["metadata_handoffs"] += 1
                    self._stats["metadata_handoff_host_seconds"] += time.perf_counter() - started
                    return
                if self._metadata_enqueued_sequence > sequence:
                    failure = TicketError("input-driven PLE metadata handoff identity mismatch")
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    failure = TicketError("input-driven PLE metadata handoff timed out")
                    break
                self._condition.wait(remaining)
        self._fail(None, failure)
        raise failure

    def signal_dummy_outputs(self, max_num_tokens: int) -> None:
        """Signal once for a profile run or an entire multi-size graph capture."""
        self.raise_if_failed()
        if max_num_tokens < 1 or max_num_tokens > self.max_num_batched_tokens:
            raise TicketError(
                f"input-driven PLE dummy token count is invalid: {max_num_tokens}"
            )
        with self._condition:
            if self._closed:
                raise TicketError("input-driven GDS PLE connector is closed")
            if self._active_sequence is not None:
                raise TicketError("previous input-driven PLE output was not released")
            sequence = self._next_sequence
            self._next_sequence += 1
            now = time.perf_counter()
            self._active_sequence = sequence
            self._active_tokens = int(max_num_tokens)
            self._records[sequence] = {
                "sequence": sequence,
                "num_reqs": 0,
                "num_tokens": int(max_num_tokens),
                "enqueued_at": now,
                "consumer_enqueued_at": now,
                "dummy": True,
                "released": False,
                "completed": True,
            }
            self._stats["launched"] += 1
            self._stats["dummy"] += 1
        stream = torch.cuda.current_stream(self.device)
        self.output[:max_num_tokens].zero_()
        self._sem.signal(stream)

    def consume_output(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = input_ids.reshape(-1).numel()
        if torch.compiler.is_compiling():
            torch.ops.vllm.ple_offload_wait(
                self._sem.flag_tensor,
                self.output,
                hidden_states,
            )
            return self.output[:num_tokens, : self.embedding_dim]

        with self._condition:
            active_sequence = self._active_sequence
            active_tokens = self._active_tokens
            record = self._records.get(active_sequence) if active_sequence else None
            dummy = bool(record and record.get("dummy"))
        if active_sequence is None or (
            num_tokens > active_tokens if dummy else active_tokens != num_tokens
        ):
            raise TicketError(
                "input-driven PLE output identity mismatch: "
                f"active={active_tokens}, model={num_tokens}"
            )
        started = time.perf_counter()
        torch.ops.vllm.ple_offload_wait(
            self._sem.flag_tensor,
            self.output,
            hidden_states,
        )
        elapsed = time.perf_counter() - started
        with self._condition:
            sequence = self._active_sequence
            if (
                sequence is not None
                and sequence in self._records
                and not self._records[sequence].get("dummy")
            ):
                self._records[sequence]["consumer_host_enqueue_seconds"] = elapsed
        return self.output[:num_tokens, : self.embedding_dim]

    def release_outputs(self) -> None:
        """Reset once after stock GPUModelRunner has consumed the output."""
        with self._condition:
            sequence = self._active_sequence
            if sequence is None:
                raise TicketError("no input-driven PLE output to release")
        self._sem.reset(torch.cuda.current_stream(self.device))
        with self._condition:
            record = self._records[sequence]
            record["released"] = True
            record["released_at"] = time.perf_counter()
            if not record.get("dummy"):
                host_wait = float(record.get("consumer_host_enqueue_seconds", 0.0))
                record["consumer_host_wait_seconds"] = host_wait
                self._stats["consumer_host_wait_seconds"].append(host_wait)
            self._stats["released"] += 1
            self._active_sequence = None
            self._active_tokens = 0
            self._finalize_record_locked(sequence)
            error = self._error
        if error is not None:
            raise TicketError("input-driven GDS PLE background path failed") from error

    def _request_loop(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            request = self._request_queue.get()
            if request is None:
                return
            try:
                wait_started = time.perf_counter()
                with torch.cuda.stream(self._d2h_stream):
                    self._d2h_stream.wait_event(self._input_ready_event)
                    self._input_ids_buf[: request.num_tokens].copy_(
                        self._input_ids_source[: request.num_tokens], non_blocking=True
                    )
                    self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                        self._query_start_loc_source[: request.num_reqs + 1],
                        non_blocking=True,
                    )
                    self._ngram_context_buf[: request.num_reqs].copy_(
                        self._ngram_context_source[: request.num_reqs],
                        non_blocking=True,
                    )
                    self._d2h_done_event.record(self._d2h_stream)
                with self._condition:
                    self._metadata_enqueued_sequence = request.sequence
                    self._condition.notify_all()
                self._d2h_done_event.synchronize()
                input_ready = time.perf_counter() - wait_started
                plan_started = time.perf_counter()
                row_ids = plan_ngram_row_ids(
                    self.layer,
                    self._input_ids_buf[: request.num_tokens],
                    self._query_start_loc_buf[: request.num_reqs + 1],
                    self._ngram_context_buf[: request.num_reqs],
                )
                row_plan = time.perf_counter() - plan_started
                expected_bytes = request.num_tokens * self.embedding_dim
                actual_bytes = row_ids.numel() * int(self.layer.head_dim)
                if actual_bytes != expected_bytes:
                    raise TicketError(
                        "input-driven PLE row plan byte mismatch: "
                        f"{actual_bytes} != {expected_bytes}"
                    )
                with self._condition:
                    record = self._records.get(request.sequence)
                    if record is None:
                        raise TicketError("input-driven PLE request identity disappeared")
                    record["input_ready_seconds"] = input_ready
                    record["row_plan_seconds"] = row_plan
                    record["row_count"] = int(row_ids.numel())
                    if self._phase_timing_enabled:
                        record["dispatch_at"] = time.perf_counter()
                if self._ipc_producer is not None:
                    self._ipc_producer.dispatch(
                        0,
                        row_ids.tolist(),
                        1,
                        request.num_tokens,
                        wait_reset=True,
                        request_id=request.sequence,
                    )
                else:
                    self._produce_in_process(request, row_ids)
            except BaseException as exc:
                self._fail(request.sequence, exc)

    def _produce_in_process(
        self, request: InputDrivenRequest, row_ids: torch.Tensor
    ) -> None:
        """Run strict GDS on the background thread's dedicated CUDA stream."""

        from ple_gds.persistent_reader import ChunkedReadRequired

        reader = self._reader
        if reader is None:
            raise TicketError("same-process input-driven reader is not initialized")
        started = time.perf_counter()
        expected_bytes = int(request.num_tokens) * self.embedding_dim
        assembled_bytes = 0
        chunk_count = 0
        lease_metrics: list[dict[str, Any]] = []
        phase_events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event] | None = None
        if self._phase_timing_enabled:
            phase_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        with torch.cuda.stream(self._reader_stream):
            if phase_events is not None:
                wait_start_event, wait_end_event, signal_end_event = phase_events
                wait_start_event.record(self._reader_stream)
            _stream_wait_value32(self._reader_stream, self._sem.flag_tensor, 0)
            if phase_events is not None:
                wait_end_event.record(self._reader_stream)

            def account_lease(lease: Any) -> None:
                nonlocal assembled_bytes, chunk_count
                chunk_bytes = int(lease.tensor.numel() * lease.tensor.element_size())
                chunk_end = assembled_bytes + chunk_bytes
                if chunk_end > expected_bytes:
                    raise TicketError(
                        "same-process PLE chunk exceeds output boundary: "
                        f"offset={assembled_bytes} chunk={chunk_bytes} "
                        f"expected={expected_bytes}"
                    )
                assembled_bytes = chunk_end
                chunk_count += 1
                if self._phase_timing_enabled:
                    lease_metrics.append(dict(lease.metrics.as_dict()))

            row_id_list = row_ids.tolist()
            try:
                lease = reader.read_at(
                    row_id_list,
                    output_offset_bytes=0,
                    mode="sync_mt",
                    stream=self._reader_stream,
                )
            except ChunkedReadRequired:
                for lease in reader.iter_read_at(
                    row_id_list,
                    output_offset_bytes=0,
                    mode="sync_mt",
                    stream=self._reader_stream,
                ):
                    try:
                        account_lease(lease)
                    finally:
                        lease.release(self._reader_stream)
            else:
                try:
                    account_lease(lease)
                finally:
                    lease.release(self._reader_stream)
            if assembled_bytes != expected_bytes:
                raise TicketError(
                    "same-process PLE assembled payload size mismatch: "
                    f"{assembled_bytes} != {expected_bytes}"
                )
            _stream_write_value32(self._reader_stream, self._sem.flag_tensor, 1)
            if phase_events is not None:
                signal_end_event.record(self._reader_stream)

        submitted_at = time.perf_counter()
        reader_stats = reader.stats()
        submitted = {
            "type": "submitted",
            "slot": 0,
            "value": 1,
            "seconds": submitted_at - started,
            "started_at": started,
            "chunk_count": chunk_count,
            "assembled_bytes": assembled_bytes,
            "request_id": request.sequence,
            "direct_output": True,
            "reader": reader_stats,
        }
        if self._phase_timing_enabled:
            submitted["lease_metrics"] = lease_metrics
            submitted["reader_phase"] = _aggregate_lease_metrics(lease_metrics)
        self._handle_ipc_result(submitted)

        # This is the existing final producer completion wait. It executes only
        # on the background control thread and protects reader/output lifetime.
        self._reader_stream.synchronize()
        completed_at = time.perf_counter()
        completed = {
            "type": "completed",
            "slot": 0,
            "value": 1,
            "seconds": completed_at - started,
            "started_at": started,
            "completed_at": completed_at,
            "chunk_count": chunk_count,
            "assembled_bytes": assembled_bytes,
            "request_id": request.sequence,
            "direct_output": True,
            "reader": reader.stats(),
        }
        if self._phase_timing_enabled:
            completed["lease_metrics"] = lease_metrics
            completed["reader_phase"] = _aggregate_lease_metrics(lease_metrics)
            completed["phase_timing"] = {
                "gpu_semaphore_wait_seconds": _event_elapsed_seconds(
                    wait_start_event, wait_end_event
                ),
                "post_wait_gather_signal_gpu_seconds": _event_elapsed_seconds(
                    wait_end_event, signal_end_event
                ),
            }
        self._handle_ipc_result(completed)

    def _handle_ipc_result(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "ready":
            return
        callback_at = time.perf_counter() if self._phase_timing_enabled else 0.0
        request_id = message.get("request_id")
        if not isinstance(request_id, int):
            self._fail(None, TicketError(f"IPC result lacks request identity: {message}"))
            return
        with self._condition:
            record = self._records.get(request_id)
            if record is None or record.get("dummy"):
                error = TicketError(f"stale input-driven IPC result: {message}")
            elif int(message.get("slot", -1)) != 0 or int(message.get("value", -1)) != 1:
                error = TicketError(f"invalid input-driven IPC result: {message}")
            elif not bool(message.get("direct_output", False)):
                error = TicketError("input-driven IPC producer did not use direct output")
            else:
                error = None
                if kind == "submitted":
                    if self._phase_timing_enabled:
                        record["ipc_phase"] = _merge_ipc_phase_message(
                            record.get("ipc_phase"),
                            message,
                            dispatch_at=float(record.get("dispatch_at", callback_at)),
                            callback_at=callback_at,
                        )
                    record["gds_submitted_at"] = time.perf_counter()
                    record["gds_submission_seconds"] = float(
                        message.get("seconds", 0.0)
                    )
                    return
                if kind == "completed":
                    if record.get("completed"):
                        error = TicketError(
                            f"duplicate input-driven IPC completion {request_id}"
                        )
                    else:
                        completed_at = float(
                            message.get("completed_at", time.perf_counter())
                        )
                        if self._phase_timing_enabled:
                            record["ipc_phase"] = _merge_ipc_phase_message(
                                record.get("ipc_phase"),
                                message,
                                dispatch_at=float(
                                    record.get("dispatch_at", callback_at)
                                ),
                                callback_at=callback_at,
                            )
                        record["completed"] = True
                        record["completed_at"] = completed_at
                        record["gds_read_seconds"] = float(
                            message.get("seconds", 0.0)
                        )
                        record["chunk_count"] = int(message.get("chunk_count", 0))
                        record["assembled_bytes"] = int(
                            message.get("assembled_bytes", 0)
                        )
                        record["ple_consumer_wait_seconds"] = max(
                            0.0,
                            completed_at - float(record["consumer_enqueued_at"]),
                        )
                        reader_stats = message.get("reader")
                        if isinstance(reader_stats, Mapping):
                            self._ipc_reader_stats = dict(reader_stats)
                        self._stats["completed"] += 1
                        self._finalize_record_locked(request_id)
                        return
                elif kind == "error":
                    error = TicketError(str(message.get("error", "IPC GDS failed")))
                else:
                    error = TicketError(f"unknown input-driven IPC result: {message}")
        assert error is not None
        self._fail(request_id, error)

    def _finalize_record_locked(self, sequence: int) -> None:
        record = self._records.get(sequence)
        if record is None or record.get("dummy"):
            if record is not None and record.get("released"):
                self._records.pop(sequence, None)
            return
        if not record.get("released") or not record.get("completed"):
            return
        sample = dict(record)
        for name in (
            "input_ready_seconds",
            "row_plan_seconds",
            "gds_read_seconds",
            "ple_consumer_wait_seconds",
        ):
            value = float(sample.get(name, 0.0))
            self._stats[name].append(value)
        self._stats["samples"].append(sample)
        if self._phase_timing_enabled and isinstance(sample.get("ipc_phase"), Mapping):
            phase = dict(sample["ipc_phase"])
            phase["request_id"] = sequence
            self._stats["ipc_phase_samples"].append(phase)
        self._records.pop(sequence, None)

    def _fail(self, sequence: int | None, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._stats["errors"] += 1
            self._condition.notify_all()
            active = self._active_sequence
            active_tokens = self._active_tokens
        # Unblock a GPU wait with zero data, then raise fail-closed at release
        # or the next entry point. This is an error wakeup, never a success.
        if active is not None and (sequence is None or sequence == active):
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(
                    self._error_stream
                ):
                    self.output[:active_tokens].zero_()
                    self._sem.signal(self._error_stream)
            except BaseException:
                logger.exception("failed to wake input-driven PLE after an error")
        logger.error("input-driven GDS PLE failed: %r", error)

    def stats(self) -> dict[str, Any]:
        with self._condition:
            result = dict(self._stats)
            for name in (
                "input_ready_seconds",
                "row_plan_seconds",
                "gds_read_seconds",
                "ple_consumer_wait_seconds",
                "consumer_host_wait_seconds",
                "ipc_phase_samples",
                "samples",
            ):
                result[name] = list(result[name])
            result.update(
                input_driven=True,
                ipc_process=self.use_ipc_process,
                same_process_reader=not self.use_ipc_process,
                direct_ipc_output=self.use_ipc_process,
                direct_output=True,
                cache_rows=0,
                cpu_ple_payload=False,
                active_sequence=self._active_sequence,
                active_kind=(
                    None
                    if self._active_sequence is None
                    else (
                        "dummy"
                        if self._records[self._active_sequence].get("dummy")
                        else "real"
                    )
                ),
                pending_metadata=self._request_queue.qsize(),
                failed=self._error is not None,
                closed=self._closed,
                ipc_reader=(
                    dict(self._ipc_reader_stats)
                    if self._ipc_reader_stats is not None
                    else None
                ),
            )
            return result

    def _metrics_loop(self) -> None:
        try:
            self._export_metrics()
            while not self._metrics_stop.wait(self._metrics_interval):
                self._export_metrics()
        except BaseException:
            logger.exception("input-driven GDS PLE metrics exporter failed")

    def _export_metrics(self) -> None:
        if self._metrics_path is None:
            return
        stats = self.stats()
        payload = {
            "schema": "gds-ple-input-driven-metrics-v1",
            "timestamp": time.time(),
            "summary": {
                "input_ready": _distribution_seconds(stats["input_ready_seconds"]),
                "row_plan": _distribution_seconds(stats["row_plan_seconds"]),
                "gds_read": _distribution_seconds(stats["gds_read_seconds"]),
                "ple_consumer_wait": _distribution_seconds(
                    stats["ple_consumer_wait_seconds"]
                ),
                "consumer_host_wait": _distribution_seconds(
                    stats["consumer_host_wait_seconds"]
                ),
                "ipc_phase": _phase_summary(stats["ipc_phase_samples"]),
            },
            "phase_timing_enabled": self._phase_timing_enabled,
            "connector": stats,
        }
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_name(
            f".{self._metrics_path.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self._metrics_path)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            self._stats["close_count"] += 1
            active = self._active_sequence is not None
        if active:
            self._sem.reset(torch.cuda.current_stream(self.device))
        try:
            self._request_queue.put(None, timeout=5)
        except queue.Full as exc:
            raise TicketError("input-driven PLE request thread did not drain") from exc
        if self._request_thread is not None:
            self._request_thread.join(timeout=30)
            if self._request_thread.is_alive():
                raise TicketError("input-driven PLE request thread failed to stop")
        if self._ipc_producer is not None:
            self._ipc_producer.close()
        self._metrics_stop.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=10)
        self._export_metrics()
        with torch.cuda.device(self.device):
            self._unpin_input_buffers()
        self.layer._ple_gds_input_connector = None


class GdsPleRunnerConnector:
    """Three-slot row-ID mailbox and PP0-only strict GDS producer."""

    def __init__(
        self,
        *,
        layer: Any,
        max_num_batched_tokens: int,
        shadow: CpuTokenShadow,
        queue_depth: int = 3,
        owner: bool = True,
        lookup_fn: LookupFn | None = None,
        use_ipc_process: bool = False,
    ) -> None:
        if queue_depth != 3:
            raise ValueError("production GDS PLE mailbox depth is fixed at 3")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self.layer = layer
        self.shadow = shadow
        self.owner = bool(owner)
        self.device = getattr(getattr(layer, "_ple_gds_reader", None), "device", None)
        if self.owner and self.device is None:
            raise TicketError("PP0 GDS runner requires an initialized strict reader")
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.embedding_dim = int(layer.embedding_dim)
        self.queue_depth = queue_depth
        self._lookup_fn = lookup_fn or _lookup_into
        self.use_ipc_process = bool(use_ipc_process and self.owner)
        self._condition = threading.Condition()
        self._pending: queue.Queue[int | None] = queue.Queue(maxsize=queue_depth)
        self._closed = False
        self._next_step = 1
        self._last_submitted_step = 0
        self._last_consumed_step = 0
        self._next_prefetch_sequence = 1
        self._next_done_value = 1
        self._active_slot: int | None = None
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "errors": 0,
            "late_tickets": 0,
            "prefetch_reserved": 0,
            "prefetch_completed": 0,
            "prefetch_consumed": 0,
            "prefetch_cancelled": 0,
            "prefetch_submission_wait_seconds": deque(maxlen=4096),
            "output_allocations_per_forward": 0,
            "producer_seconds": deque(maxlen=4096),
            "chunk_count": deque(maxlen=4096),
            "assembled_bytes": deque(maxlen=4096),
            "consumer_wait_seconds": deque(maxlen=4096),
            "visible_seconds": deque(maxlen=4096),
            "ticket_samples": deque(maxlen=4096),
            "ipc_phase_samples": deque(maxlen=4096),
        }
        self._phase_timing_enabled = _enabled("VLLM_PLE_GDS_PHASE_TIMING")
        self._thread: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None
        self._metrics_stop = threading.Event()
        metrics_path = os.getenv("VLLM_PLE_GDS_RUNNER_METRICS_PATH", "").strip()
        self._metrics_path = Path(metrics_path) if metrics_path else None
        self.control_worker: PpSampleControlWorker | None = None
        self._ipc_reader_stats: dict[str, Any] | None = None
        self._metrics_interval = float(
            os.getenv("VLLM_PLE_GDS_RUNNER_METRICS_INTERVAL", "5")
        )
        if self._metrics_interval <= 0:
            raise ValueError("runner metrics interval must be positive")
        self._slots: list[TicketSlot] = []
        self.consumer_buffer: torch.Tensor | None = None
        if self.owner:
            with torch.cuda.device(self.device):
                self.consumer_buffer = torch.empty(
                    (self.max_num_batched_tokens, self.embedding_dim),
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )
                for index in range(queue_depth):
                    self._slots.append(
                        TicketSlot(
                            index=index,
                            output=torch.empty_like(self.consumer_buffer),
                            producer_done=torch.cuda.Event(),
                            consumer_done=torch.cuda.Event(),
                            done_flag=torch.zeros(1, dtype=torch.int32, device=self.device),
                            submission_ready=threading.Event(),
                        )
                    )
            layer._ple_gds_runner_output = self.consumer_buffer
            layer._ple_gds_runner_active_tokens = 0
            self._ipc_producer: IpcGdsProducer | None = None
            if self.use_ipc_process:
                self._ipc_producer = IpcGdsProducer(
                    output_slots=[slot.output for slot in self._slots],
                    done_flags=[slot.done_flag for slot in self._slots],
                    artifact=os.environ["VLLM_PLE_GDS_ARTIFACT"],
                    gpu_uuid=os.environ["VLLM_PLE_GDS_GPU_UUID"],
                    gpu_bdf=os.environ["VLLM_PLE_GDS_GPU_BDF"],
                    worker_count=int(os.getenv("VLLM_PLE_GDS_WORKERS", "32")),
                    callback=self._handle_ipc_result,
                )
                self._ipc_reader_stats = dict(self._ipc_producer.ready["reader"])
            self._thread = threading.Thread(
                target=self._producer_loop,
                name="ple-gds-pp0-producer",
                daemon=True,
            )
            self._thread.start()
            if self._metrics_path is not None:
                self._metrics_thread = threading.Thread(
                    target=self._metrics_loop,
                    name="ple-gds-metrics-exporter",
                    daemon=True,
                )
                self._metrics_thread.start()
        else:
            self._ipc_producer = None

    def next_step_id(self) -> int:
        step = self._next_step
        self._next_step += 1
        return step

    def _handle_ipc_result(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "ready":
            return
        callback_at = time.perf_counter() if self._phase_timing_enabled else 0.0
        try:
            slot_index = int(message["slot"])
            if slot_index < 0 or slot_index >= len(self._slots):
                raise IndexError(slot_index)
            slot = self._slots[slot_index]
            value = int(message["value"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            with self._condition:
                self._stats["errors"] += 1
            self._record_ipc_error(TicketError(f"invalid IPC producer message: {message}"))
            return
        with self._condition:
            if slot.done_value != value:
                self._record_ipc_error(
                    TicketError(f"IPC producer value mismatch for slot {slot.index}: {value} != {slot.done_value}")
                )
                return
            if kind == "submitted":
                has_ticket = slot.ticket is not None
                has_prefetch = slot.prefetch is not None
                has_active_identity = has_ticket != has_prefetch
                submission_pending = not slot.submission_ready.is_set()
                if not has_active_identity or not submission_pending:
                    self._record_ipc_error(TicketError("IPC producer submitted a non-active slot"))
                    return
                if slot.state in {SlotState.QUEUED, SlotState.PRODUCING}:
                    slot.state = SlotState.READY
                elif slot.state == SlotState.CONSUMING and has_prefetch:
                    # A prefetch consumer may claim the matching identity before
                    # the child result thread reports its already-enqueued work.
                    # Preserve consumer ownership while publishing submission.
                    pass
                else:
                    self._record_ipc_error(TicketError("IPC producer submitted a non-active slot"))
                    return
                slot.submission_ready.set()
                submission_seconds = float(message.get("seconds", 0.0))
                slot.metrics["ipc_submission_seconds"] = submission_seconds
                slot.metrics["chunk_count"] = int(message.get("chunk_count", 0))
                slot.metrics["assembled_bytes"] = int(
                    message.get("assembled_bytes", 0)
                )
                # Match the in-process producer contract: READY always carries
                # a producer duration that the consumer can sample immediately.
                # COMPLETED may later replace it with the full stream duration.
                slot.metrics["producer_seconds"] = submission_seconds
                if self._phase_timing_enabled:
                    slot.metrics["ipc_phase"] = _merge_ipc_phase_message(
                        slot.metrics.get("ipc_phase"),
                        message,
                        dispatch_at=float(slot.dispatch_at),
                        callback_at=callback_at,
                    )
                self._condition.notify_all()
                return
            if kind == "completed":
                has_ticket = slot.ticket is not None
                has_prefetch = slot.prefetch is not None
                if (
                    has_ticket == has_prefetch
                    or not slot.submission_ready.is_set()
                    or slot.producer_completed
                    or slot.state
                    not in {SlotState.READY, SlotState.CONSUMING, SlotState.RELEASING}
                ):
                    self._record_ipc_error(TicketError("IPC producer completed a non-active slot"))
                    return
                slot.producer_completed = True
                slot.producer_finished_at = time.perf_counter()
                producer_seconds = float(message.get("seconds", 0.0))
                chunk_count = int(message.get("chunk_count", 0))
                assembled_bytes = int(message.get("assembled_bytes", 0))
                slot.metrics["producer_seconds"] = producer_seconds
                slot.metrics["chunk_count"] = chunk_count
                slot.metrics["assembled_bytes"] = assembled_bytes
                if self._phase_timing_enabled:
                    slot.metrics["ipc_phase"] = _merge_ipc_phase_message(
                        slot.metrics.get("ipc_phase"),
                        message,
                        dispatch_at=float(slot.dispatch_at),
                        callback_at=callback_at,
                    )
                ticket_sample = slot.metrics.get("ticket_sample")
                if ticket_sample is not None:
                    ticket_sample["producer_seconds"] = producer_seconds
                    ticket_sample["chunk_count"] = chunk_count
                    ticket_sample["assembled_bytes"] = assembled_bytes
                self._stats["completed"] += 1
                self._stats["producer_seconds"].append(producer_seconds)
                self._stats["chunk_count"].append(chunk_count)
                self._stats["assembled_bytes"].append(assembled_bytes)
                if self._phase_timing_enabled and isinstance(
                    slot.metrics.get("ipc_phase"), Mapping
                ):
                    phase = dict(slot.metrics["ipc_phase"])
                    phase["slot"] = slot.index
                    phase["done_value"] = slot.done_value
                    if slot.ticket is not None:
                        phase["step_id"] = slot.ticket.identity.step_id
                    if slot.prefetch is not None:
                        phase["prefetch_sequence"] = slot.prefetch.sequence
                    self._stats["ipc_phase_samples"].append(phase)
                reader_stats = message.get("reader")
                if isinstance(reader_stats, Mapping):
                    self._ipc_reader_stats = dict(reader_stats)
                if slot.prefetch is not None:
                    self._stats["prefetch_completed"] += 1
                if slot.discard and slot.state in {SlotState.READY, SlotState.RELEASING}:
                    self._reset_slot_locked(slot)
                self._condition.notify_all()
                return
            if kind == "error":
                slot.error = TicketError(str(message.get("error", "IPC producer failed")))
                slot.state = SlotState.ERROR
                slot.submission_ready.set()
                self._stats["errors"] += 1
                self._condition.notify_all()
                self._record_ipc_error(slot.error)
                return
        self._record_ipc_error(TicketError(f"unknown IPC producer message: {message}"))

    def _record_ipc_error(self, error: BaseException) -> None:
        worker = self.control_worker
        if worker is not None:
            worker._record_error(error)

    def submit(self, plan: CpuBatchPlan) -> BatchIdentity:
        if not self.owner:
            raise TicketError("non-owner PP rank cannot submit GDS tickets")
        if plan.identity.step_id <= self._last_submitted_step:
            raise TicketError(
                f"duplicate or out-of-order ticket {plan.identity.step_id}"
            )
        if plan.identity.padded_tokens > self.max_num_batched_tokens:
            raise TicketError("ticket exceeds fixed runner output capacity")
        row_ids = plan_ngram_row_ids(
            self.layer, plan.input_ids, plan.query_start_loc, plan.ngram_context
        )
        with self._condition:
            if self._closed:
                raise TicketError("GDS PLE runner is closed")
            self._reap_locked()
            slot = next((item for item in self._slots if item.state == SlotState.FREE), None)
            if slot is None:
                raise MailboxFull("all three GDS PLE output slots are active")
            slot.state = SlotState.QUEUED
            slot.ticket = plan
            slot.row_ids = row_ids
            slot.error = None
            slot.submitted_at = time.perf_counter()
            slot.producer_completed = False
            slot.done_value = self._next_done_value
            self._next_done_value += 1
            slot.metrics = {}
            self._last_submitted_step = plan.identity.step_id
            self._stats["submitted"] += 1
        try:
            self._pending.put_nowait(slot.index)
        except queue.Full as exc:
            with self._condition:
                self._reset_slot_locked(slot)
            raise MailboxFull("bounded GDS PLE producer queue is full") from exc
        return plan.identity

    def reserve_prefetch(
        self, *, req_id: str, generation: int, start: int
    ) -> tuple[PrefetchIdentity, int]:
        """Reserve a ring slot before sampled-token D2H completion."""

        with self._condition:
            if self._closed:
                raise TicketError("GDS PLE runner is closed")
            self._reap_locked()
            slot = next((item for item in self._slots if item.state == SlotState.FREE), None)
            if slot is None:
                raise MailboxFull("all three GDS PLE output slots are active")
            identity = PrefetchIdentity(
                sequence=self._next_prefetch_sequence,
                req_id=str(req_id),
                generation=int(generation),
                start=int(start),
            )
            self._next_prefetch_sequence += 1
            slot.state = SlotState.QUEUED
            slot.prefetch = identity
            slot.discard = False
            slot.error = None
            slot.submission_ready.clear()
            slot.submitted_at = time.perf_counter()
            slot.producer_completed = False
            slot.done_value = self._next_done_value
            self._next_done_value += 1
            slot.metrics = {}
            self._stats["prefetch_reserved"] += 1
            return identity, slot.index

    def cancel_prefetch(self, identity: PrefetchIdentity, slot_index: int) -> None:
        with self._condition:
            slot = self._slots[slot_index]
            if slot.prefetch != identity:
                raise TicketError("prefetch cancellation identity mismatch")
            if slot.state == SlotState.CONSUMING:
                raise TicketError("cannot cancel an already consumed prefetch")
            self._reset_slot_locked(slot)
            self._stats["prefetch_cancelled"] += 1
            self._condition.notify_all()

    def produce_prefetch(
        self,
        identity: PrefetchIdentity,
        slot_index: int,
        plan: CpuBatchPlan,
        stream: torch.cuda.Stream,
    ) -> None:
        slot = self._slots[slot_index]
        with self._condition:
            if slot.prefetch != identity or slot.state not in {
                SlotState.QUEUED,
                SlotState.CONSUMING,
            }:
                raise TicketError("prefetch producer observed corrupt slot state")
            if (
                plan.identity.req_ids != (identity.req_id,)
                or plan.identity.generations != (identity.generation,)
                or plan.identity.starts != (identity.start,)
                or plan.identity.lengths != (1,)
                or plan.identity.valid_tokens != 1
                or plan.identity.padded_tokens != 1
            ):
                raise TicketError("background prefetch plan identity mismatch")
            if slot.state == SlotState.QUEUED:
                slot.state = SlotState.PRODUCING
            slot.producer_started_at = time.perf_counter()
        try:
            row_ids = plan_ngram_row_ids(
                self.layer, plan.input_ids, plan.query_start_loc, plan.ngram_context
            )
            if self._ipc_producer is not None:
                with self._condition:
                    if (
                        slot.prefetch != identity
                        or slot.state not in {SlotState.PRODUCING, SlotState.CONSUMING}
                    ):
                        raise TicketError("prefetch slot changed before IPC dispatch")
                    if self._phase_timing_enabled:
                        slot.dispatch_at = time.perf_counter()
                self._ipc_producer.dispatch(
                    slot.index,
                    row_ids.tolist(),
                    slot.done_value,
                    plan.identity.valid_tokens,
                )
                return
            with torch.cuda.stream(stream):
                metrics = self._lookup_fn(self.layer, row_ids, slot.output[:1])
                slot.producer_done.record(stream)
            finished = time.perf_counter()
            with self._condition:
                slot.producer_finished_at = finished
                slot.metrics = dict(metrics)
                slot.metrics.setdefault("chunk_count", 1)
                slot.metrics.setdefault(
                    "assembled_bytes", plan.identity.valid_tokens * self.embedding_dim
                )
                slot.metrics["producer_seconds"] = (
                    finished - slot.producer_started_at
                )
                if slot.discard:
                    slot.consumer_done.record(stream)
                    slot.state = SlotState.RELEASING
                elif slot.state == SlotState.PRODUCING:
                    slot.state = SlotState.READY
                elif slot.state != SlotState.CONSUMING:
                    raise TicketError("prefetch slot changed state during production")
                self._stats["prefetch_completed"] += 1
                self._stats["producer_seconds"].append(
                    slot.metrics["producer_seconds"]
                )
                self._stats["chunk_count"].append(slot.metrics["chunk_count"])
                self._stats["assembled_bytes"].append(
                    slot.metrics["assembled_bytes"]
                )
                self._condition.notify_all()
            slot.submission_ready.set()
        except BaseException as exc:
            with torch.cuda.stream(stream):
                slot.output[:1].zero_()
                slot.producer_done.record(stream)
            with self._condition:
                slot.error = exc
                slot.state = SlotState.ERROR
                self._stats["errors"] += 1
                self._condition.notify_all()
            slot.submission_ready.set()
            raise

    def consume_prefetch(
        self,
        expected: PrefetchIdentity,
        stream: torch.cuda.Stream | None = None,
        timeout: float = 30.0,
    ) -> torch.Tensor:
        if self.consumer_buffer is None:
            raise TicketError("GDS PLE runner has no consumer buffer")
        with self._condition:
            slot = next(
                (item for item in self._slots if item.prefetch == expected), None
            )
            if slot is None:
                raise LateSampleMetadata(
                    f"no background prefetch for {expected.req_id} at {expected.start}"
                )
            if slot.state == SlotState.ERROR:
                assert slot.error is not None
                raise TicketError("background prefetch failed") from slot.error
            if slot.state not in {
                SlotState.QUEUED,
                SlotState.PRODUCING,
                SlotState.READY,
            }:
                raise TicketError(f"prefetch slot is not consumable: {slot.state}")
            slot.state = SlotState.CONSUMING
            slot.consumer_acquired_at = time.perf_counter()
            self._active_slot = slot.index
            self._stats["prefetch_consumed"] += 1
        wait_started = time.perf_counter()
        if not slot.submission_ready.wait(timeout):
            with self._condition:
                self._stats["late_tickets"] += 1
            raise LateSampleMetadata(
                f"timed out waiting for prefetch submission for {expected.req_id}"
            )
        submission_wait = time.perf_counter() - wait_started
        with self._condition:
            self._stats["prefetch_submission_wait_seconds"].append(submission_wait)
            if slot.error is not None:
                raise TicketError("background prefetch failed") from slot.error
        use_stream = stream or torch.cuda.current_stream(self.device)
        if self._ipc_producer is not None:
            _stream_wait_value32(use_stream, slot.done_flag, slot.done_value)
        else:
            use_stream.wait_event(slot.producer_done)
        self.consumer_buffer[:1].copy_(slot.output[:1], non_blocking=True)
        self.layer._ple_gds_runner_active_tokens = 1
        return self.consumer_buffer[:1]

    def find_prefetch(
        self, *, req_id: str, generation: int, start: int
    ) -> PrefetchIdentity:
        with self._condition:
            candidates = sorted(
                (
                    item.prefetch
                    for item in self._slots
                    if item.prefetch is not None
                    and not item.discard
                    and item.state
                    in {SlotState.QUEUED, SlotState.PRODUCING, SlotState.READY}
                ),
                key=lambda item: item.sequence,
            )
            if not candidates:
                raise LateSampleMetadata(
                    f"no queued prefetch for {req_id} generation {generation}"
                )
            expected = candidates[0]
            actual = (str(req_id), int(generation), int(start))
            planned = (expected.req_id, expected.generation, expected.start)
            if actual != planned:
                raise TicketError(
                    f"background prefetch diverged from scheduler: "
                    f"{planned} != {actual}"
                )
            return expected

    def discard_prefetches(self, req_id: str, generation: int) -> int:
        """Retire speculative tail reads for a request that has finished."""

        discarded = 0
        with self._condition:
            self._reap_locked()
            for slot in self._slots:
                identity = slot.prefetch
                if (
                    identity is None
                    or identity.req_id != str(req_id)
                    or identity.generation != int(generation)
                    or slot.state == SlotState.FREE
                    or slot.discard
                ):
                    continue
                slot.discard = True
                discarded += 1
                if slot.state in {SlotState.READY, SlotState.ERROR}:
                    if slot.producer_done.query():
                        self._reset_slot_locked(slot)
                # PRODUCING transitions to RELEASING after recording its
                # producer event. CONSUMING/RELEASING is owned by release().
            self._stats["prefetch_cancelled"] += discarded
            self._condition.notify_all()
        return discarded

    def consume(
        self,
        expected: BatchIdentity,
        stream: torch.cuda.Stream | None = None,
        timeout: float = 30.0,
    ) -> torch.Tensor:
        if not self.owner or self.consumer_buffer is None:
            raise TicketError("non-owner PP rank cannot consume GDS tickets")
        if expected.step_id <= self._last_consumed_step:
            raise TicketError(f"stale or duplicate consume for step {expected.step_id}")
        wait_started = time.perf_counter()
        deadline = wait_started + timeout
        with self._condition:
            while True:
                slot = next(
                    (
                        item
                        for item in self._slots
                        if item.ticket is not None
                        and item.ticket.identity.step_id == expected.step_id
                    ),
                    None,
                )
                if slot is None:
                    raise TicketError(f"no ticket for step {expected.step_id}")
                if slot.ticket is None or slot.ticket.identity != expected:
                    raise TicketError("ticket batch identity mismatch")
                if slot.state == SlotState.ERROR:
                    assert slot.error is not None
                    raise TicketError(
                        f"producer failed for step {expected.step_id}"
                    ) from slot.error
                if slot.state == SlotState.READY:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self._stats["late_tickets"] += 1
                    raise TicketError(f"timed out waiting for step {expected.step_id}")
                self._condition.wait(remaining)
            slot.state = SlotState.CONSUMING
            slot.consumer_acquired_at = time.perf_counter()
            self._active_slot = slot.index
            self._last_consumed_step = expected.step_id
        use_stream = stream or torch.cuda.current_stream(self.device)
        if self._ipc_producer is not None:
            _stream_wait_value32(use_stream, slot.done_flag, slot.done_value)
        else:
            use_stream.wait_event(slot.producer_done)
        visible_started = time.perf_counter()
        self.consumer_buffer[: expected.valid_tokens].copy_(
            slot.output[: expected.valid_tokens], non_blocking=True
        )
        if expected.padded_tokens > expected.valid_tokens:
            self.consumer_buffer[
                expected.valid_tokens : expected.padded_tokens
            ].zero_()
        self.layer._ple_gds_runner_active_tokens = expected.padded_tokens
        wait_seconds = slot.consumer_acquired_at - wait_started
        visible_seconds = wait_seconds + time.perf_counter() - visible_started
        with self._condition:
            self._stats["consumer_wait_seconds"].append(wait_seconds)
            self._stats["visible_seconds"].append(visible_seconds)
            ticket_sample = {
                "step_id": expected.step_id,
                "valid_tokens": expected.valid_tokens,
                "padded_tokens": expected.padded_tokens,
                "ticket_lead_seconds": wait_started - slot.submitted_at,
                "producer_seconds": float(slot.metrics["producer_seconds"]),
                "chunk_count": int(slot.metrics.get("chunk_count", 1)),
                "assembled_bytes": int(
                    slot.metrics.get(
                        "assembled_bytes",
                        expected.valid_tokens * self.embedding_dim,
                    )
                ),
                "consumer_wait_seconds": wait_seconds,
                "visible_seconds": visible_seconds,
            }
            slot.metrics["ticket_sample"] = ticket_sample
            self._stats["ticket_samples"].append(ticket_sample)
        return self.consumer_buffer[: expected.padded_tokens]

    def release(self, stream: torch.cuda.Stream | None = None) -> None:
        if not self.owner:
            return
        with self._condition:
            if self._active_slot is None:
                raise TicketError("no consumed GDS PLE slot to release")
            slot = self._slots[self._active_slot]
            if slot.state != SlotState.CONSUMING:
                raise TicketError(f"slot {slot.index} is not being consumed")
        use_stream = stream or torch.cuda.current_stream(self.device)
        slot.consumer_done.record(use_stream)
        with self._condition:
            slot.state = SlotState.RELEASING
            self._active_slot = None
        self.layer._ple_gds_runner_active_tokens = 0

    def prepare_dummy(
        self, num_tokens: int, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        if not self.owner or self.consumer_buffer is None:
            raise TicketError("non-owner PP rank cannot prepare PLE dummy output")
        if num_tokens < 0 or num_tokens > self.max_num_batched_tokens:
            raise TicketError("dummy token count exceeds fixed output")
        use_stream = stream or torch.cuda.current_stream(self.device)
        with torch.cuda.stream(use_stream):
            self.consumer_buffer[:num_tokens].zero_()
        self.layer._ple_gds_runner_active_tokens = num_tokens
        return self.consumer_buffer[:num_tokens]

    def cancel_pending(self) -> None:
        with self._condition:
            for slot in self._slots:
                if slot.state == SlotState.QUEUED:
                    slot.state = SlotState.ERROR
                    slot.error = TicketError("ticket cancelled")
                    self._stats["cancelled"] += 1
            self._condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._condition:
            self._reap_locked()
            result = dict(self._stats)
            for name in (
                "producer_seconds",
                "chunk_count",
                "assembled_bytes",
                "consumer_wait_seconds",
                "visible_seconds",
                "ticket_samples",
                "prefetch_submission_wait_seconds",
                "ipc_phase_samples",
            ):
                result[name] = list(result[name])
            result.update(
                mailbox_depth=self.queue_depth,
                pending_tickets=sum(
                    item.state in {SlotState.QUEUED, SlotState.PRODUCING}
                    for item in self._slots
                ),
                pending_prefetch=sum(
                    item.prefetch is not None
                    and not item.discard
                    and item.state != SlotState.FREE
                    for item in self._slots
                ),
                active_output_slots=sum(item.state != SlotState.FREE for item in self._slots),
                slot_states=[item.state.value for item in self._slots],
                owner=self.owner,
                closed=self._closed,
                ipc_process=self._ipc_producer is not None,
                ipc_reader=(
                    dict(self._ipc_reader_stats)
                    if self._ipc_reader_stats is not None
                    else None
                ),
            )
            return result

    def close(self) -> None:
        if not self.owner:
            self._closed = True
            return
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self._pending.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise TicketError("GDS PLE producer thread failed to stop")
        if self._ipc_producer is not None:
            self._ipc_producer.close()
        with self._condition:
            for slot in self._slots:
                if slot.state in {SlotState.CONSUMING, SlotState.RELEASING}:
                    if self._ipc_producer is None:
                        slot.consumer_done.synchronize()
                if slot.state in {
                    SlotState.READY,
                    SlotState.ERROR,
                    SlotState.CONSUMING,
                    SlotState.RELEASING,
                }:
                    self._reset_slot_locked(slot)
            active = [item.index for item in self._slots if item.state != SlotState.FREE]
            if active:
                raise TicketError(f"active output slots at shutdown: {active}")
        self._metrics_stop.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=10)
            if self._metrics_thread.is_alive():
                raise TicketError("GDS PLE metrics exporter failed to stop")
        self._export_metrics()
        self.layer._ple_gds_runner_output = None
        self.layer._ple_gds_runner_active_tokens = 0

    def _metrics_loop(self) -> None:
        try:
            self._export_metrics()
            while not self._metrics_stop.wait(self._metrics_interval):
                self._export_metrics()
        except BaseException:
            logger.exception("GDS PLE runner metrics exporter stopped after an error")

    def _export_metrics(self) -> None:
        if self._metrics_path is None:
            return
        stats = self.stats()
        samples = stats["ticket_samples"]
        decode = [sample for sample in samples if sample["valid_tokens"] <= 4]

        def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "ticket_lead": _distribution_seconds(
                    [item["ticket_lead_seconds"] for item in items]
                ),
                "producer": _distribution_seconds(
                    [item["producer_seconds"] for item in items]
                ),
                "consumer_wait": _distribution_seconds(
                    [item["consumer_wait_seconds"] for item in items]
                ),
                "visible": _distribution_seconds(
                    [item["visible_seconds"] for item in items]
                ),
            }

        payload = {
            "schema": "gds-ple-runner-metrics-v1",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "summary": {
                "all": summarize(samples),
                "decode_valid_tokens_le_4": summarize(decode),
                "prefetch": {
                    "submission_wait": _distribution_seconds(
                        stats["prefetch_submission_wait_seconds"]
                    )
                },
                "ipc_phase": _phase_summary(stats["ipc_phase_samples"]),
            },
            "phase_timing_enabled": self._phase_timing_enabled,
            "connector": stats,
            "sample_control": (
                self.control_worker.stats()
                if self.control_worker is not None
                else None
            ),
            "reader": self.layer._ple_gds_reader.stats(),
        }
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_name(
            f".{self._metrics_path.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self._metrics_path)

    def _producer_loop(self) -> None:
        assert self.device is not None
        torch.cuda.set_device(self.device)
        stream = torch.cuda.Stream(device=self.device)
        while True:
            index = self._pending.get()
            if index is None:
                return
            slot = self._slots[index]
            with self._condition:
                if slot.state == SlotState.ERROR:
                    self._condition.notify_all()
                    continue
                if slot.state != SlotState.QUEUED or slot.ticket is None or slot.row_ids is None:
                    slot.state = SlotState.ERROR
                    slot.error = TicketError("producer observed corrupt slot state")
                    self._stats["errors"] += 1
                    self._condition.notify_all()
                    continue
                slot.state = SlotState.PRODUCING
                slot.producer_started_at = time.perf_counter()
            try:
                if self._ipc_producer is not None:
                    with self._condition:
                        if self._phase_timing_enabled:
                            slot.dispatch_at = time.perf_counter()
                    self._ipc_producer.dispatch(
                        slot.index,
                        slot.row_ids.tolist(),
                        slot.done_value,
                        slot.ticket.identity.valid_tokens if slot.ticket is not None else 1,
                    )
                    continue
                with torch.cuda.stream(stream):
                    output = slot.output[: slot.ticket.identity.valid_tokens]
                    metrics = self._lookup_fn(self.layer, slot.row_ids, output)
                    slot.producer_done.record(stream)
                finished = time.perf_counter()
                with self._condition:
                    slot.producer_finished_at = finished
                    slot.metrics = dict(metrics)
                    slot.metrics.setdefault("chunk_count", 1)
                    slot.metrics.setdefault(
                        "assembled_bytes",
                        slot.ticket.identity.valid_tokens * self.embedding_dim,
                    )
                    slot.metrics["producer_seconds"] = finished - slot.producer_started_at
                    slot.state = SlotState.READY
                    self._stats["completed"] += 1
                    self._stats["producer_seconds"].append(slot.metrics["producer_seconds"])
                    self._stats["chunk_count"].append(slot.metrics["chunk_count"])
                    self._stats["assembled_bytes"].append(
                        slot.metrics["assembled_bytes"]
                    )
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    slot.error = exc
                    slot.state = SlotState.ERROR
                    self._stats["errors"] += 1
                    self._condition.notify_all()

    def _reap_locked(self) -> None:
        for slot in self._slots:
            if (
                slot.state == SlotState.RELEASING
                and slot.consumer_done.query()
                and (self._ipc_producer is None or slot.producer_completed)
            ):
                self._reset_slot_locked(slot)
            elif (
                slot.discard
                and slot.state in {SlotState.READY, SlotState.ERROR}
                and (slot.producer_completed if self._ipc_producer is not None
                     else slot.producer_done.query())
            ):
                self._reset_slot_locked(slot)

    @staticmethod
    def _reset_slot_locked(slot: TicketSlot) -> None:
        slot.state = SlotState.FREE
        slot.ticket = None
        slot.prefetch = None
        slot.discard = False
        slot.row_ids = None
        slot.error = None
        slot.metrics = {}
        slot.dispatch_at = 0.0
        slot.producer_completed = False
        slot.submission_ready.clear()


@dataclasses.dataclass(frozen=True)
class CpuSampleMetadata:
    req_ids: tuple[str, ...]
    generations: tuple[int, ...]
    sampled_tokens: np.ndarray
    num_sampled: np.ndarray
    num_rejected: np.ndarray
    draft_tokens: np.ndarray
    include_mask: np.ndarray


class PpPinnedSampleBridge:
    """Fixed pinned ring populated on PP receive stream without synchronize()."""

    def __init__(self, max_num_reqs: int, max_sample_len: int, depth: int) -> None:
        if depth < 2:
            raise ValueError("PP metadata ring must cover pipeline delay")
        self.depth = depth
        self._next = 0
        self.sampled = torch.empty(
            depth, max_num_reqs, max_sample_len, dtype=torch.int64, pin_memory=True
        )
        self.counts = torch.empty(
            depth, 2, max_num_reqs, dtype=torch.int32, pin_memory=True
        )
        self.drafts = torch.empty(
            depth,
            max_num_reqs,
            max(max_sample_len - 1, 0),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.events = [torch.cuda.Event() for _ in range(depth)]
        self.in_use = [False] * depth
        self._lock = threading.Lock()

    def stage(
        self,
        *,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        draft_tokens: torch.Tensor | None,
        stream: torch.cuda.Stream,
    ) -> int:
        with self._lock:
            slot = self._next
            self._next = (self._next + 1) % self.depth
            if self.in_use[slot]:
                raise LateSampleMetadata("PP sampled-token pinned ring overrun")
            rows = sampled_tokens.shape[0]
            self.sampled[slot, :rows].copy_(sampled_tokens, non_blocking=True)
            self.counts[slot, 0, :rows].copy_(num_sampled, non_blocking=True)
            self.counts[slot, 1, :rows].copy_(num_rejected, non_blocking=True)
            if self.drafts.shape[-1]:
                self.drafts[slot, :rows].fill_(-1)
                if draft_tokens is not None:
                    if draft_tokens.shape != self.drafts[slot, :rows].shape:
                        raise TicketError(
                            "PP draft metadata shape mismatch: "
                            f"{tuple(draft_tokens.shape)} != "
                            f"{tuple(self.drafts[slot, :rows].shape)}"
                        )
                    self.drafts[slot, :rows].copy_(draft_tokens, non_blocking=True)
            self.events[slot].record(stream)
            self.in_use[slot] = True
            return slot

    def poll(
        self, slot: int, rows: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.events[slot].query():
            raise LateSampleMetadata("PP sampled metadata missed its pipeline lead time")
        return self._copy_and_release(slot, rows)

    def wait(
        self, slot: int, rows: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wait for the D2H event on a control thread, never the model thread."""

        with self._lock:
            if not self.in_use[slot]:
                raise LateSampleMetadata("PP sampled metadata slot is not active")
        self.events[slot].synchronize()
        return self._copy_and_release(slot, rows)

    def _copy_and_release(
        self, slot: int, rows: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if not self.in_use[slot]:
                raise LateSampleMetadata("PP sampled metadata slot was already consumed")
            result = (
                self.sampled[slot, :rows].numpy().copy(),
                self.counts[slot, 0, :rows].numpy().copy(),
                self.counts[slot, 1, :rows].numpy().copy(),
                self.drafts[slot, :rows].numpy().copy(),
            )
            self.in_use[slot] = False
            return result


@dataclasses.dataclass(frozen=True)
class PpSampleControlJob:
    slot: int
    rows: int
    req_ids: tuple[str, ...]
    generations: tuple[int, ...]
    idx_mapping_np: np.ndarray
    need_sampled_mask: np.ndarray
    gen_at_receive_np: np.ndarray
    prefetch: PrefetchIdentity | None
    prefetch_slot: int | None
    submitted_at: float


class PpSampleControlWorker:
    """Background producer for MTP-off sampled-token CPU shadow metadata."""

    def __init__(
        self,
        *,
        bridge: PpPinnedSampleBridge,
        shadow: CpuTokenShadow,
        req_idx_gen_np: np.ndarray,
        connector: GdsPleRunnerConnector,
    ) -> None:
        self.bridge = bridge
        self.shadow = shadow
        self.req_idx_gen_np = req_idx_gen_np
        self.connector = connector
        self._pending: queue.Queue[PpSampleControlJob | None] = queue.Queue(
            maxsize=bridge.depth
        )
        self._lock = threading.Lock()
        self._closed = False
        self._error: BaseException | None = None
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "completed": 0,
            "errors": 0,
            "event_wait_seconds": deque(maxlen=4096),
            "copy_and_shadow_seconds": deque(maxlen=4096),
            "total_seconds": deque(maxlen=4096),
        }
        self._thread = threading.Thread(
            target=self._loop,
            name="ple-gds-pp0-sample-control",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        slot: int,
        req_ids: Sequence[str],
        generations: Sequence[int],
        idx_mapping_np: np.ndarray,
        need_sampled_mask: np.ndarray,
        gen_at_receive_np: np.ndarray,
    ) -> None:
        self.raise_if_failed()
        prefetch = None
        prefetch_slot = None
        if len(req_ids) == 1 and bool(need_sampled_mask[0]):
            generation = int(generations[0])
            start = self.shadow.token_count(str(req_ids[0]), generation)
            prefetch, prefetch_slot = self.connector.reserve_prefetch(
                req_id=str(req_ids[0]), generation=generation, start=start
            )
        job = PpSampleControlJob(
            slot=int(slot),
            rows=len(req_ids),
            req_ids=tuple(req_ids),
            generations=tuple(int(value) for value in generations),
            idx_mapping_np=np.asarray(idx_mapping_np, dtype=np.int64).copy(),
            need_sampled_mask=np.asarray(need_sampled_mask, dtype=np.bool_).copy(),
            gen_at_receive_np=np.asarray(gen_at_receive_np, dtype=np.int32).copy(),
            prefetch=prefetch,
            prefetch_slot=prefetch_slot,
            submitted_at=time.perf_counter(),
        )
        with self._lock:
            if self._closed:
                raise TicketError("PP sample control worker is closed")
            self._stats["submitted"] += 1
        try:
            self._pending.put_nowait(job)
        except queue.Full as exc:
            if prefetch is not None and prefetch_slot is not None:
                self.connector.cancel_prefetch(prefetch, prefetch_slot)
            self._record_error(LateSampleMetadata("PP sample control queue is full"))
            raise LateSampleMetadata("PP sample control queue is full") from exc

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise TicketError("PP sample control worker failed") from error

    def stats(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._stats)
            for name in (
                "event_wait_seconds",
                "copy_and_shadow_seconds",
                "total_seconds",
            ):
                result[name] = list(result[name])
            result.update(
                pending=self._pending.qsize(),
                closed=self._closed,
                failed=self._error is not None,
            )
            return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pending.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise TicketError("PP sample control worker failed to stop")
        self.raise_if_failed()

    def _record_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            self._stats["errors"] += 1

    def _loop(self) -> None:
        if self.connector.device is None:
            self._record_error(TicketError("sample control worker has no CUDA device"))
            return
        torch.cuda.set_device(self.connector.device)
        stream = torch.cuda.Stream(device=self.connector.device)
        while True:
            job = self._pending.get()
            if job is None:
                return
            try:
                wait_started = time.perf_counter()
                sampled, num_sampled, _num_rejected, _draft_tokens = self.bridge.wait(
                    job.slot, job.rows
                )
                wait_finished = time.perf_counter()
                current_generations = self.req_idx_gen_np[job.idx_mapping_np].copy()
                include = job.need_sampled_mask & (
                    current_generations == job.gen_at_receive_np
                )
                self.shadow.append_sampled(
                    job.req_ids,
                    job.generations,
                    sampled,
                    num_sampled,
                    include,
                )
                if job.prefetch is not None and job.prefetch_slot is not None:
                    if bool(include[0]):
                        plan = self.shadow.plan(
                            step_id=0,
                            req_ids=(job.prefetch.req_id,),
                            starts=(job.prefetch.start,),
                            lengths=(1,),
                            drafts={},
                            padded_tokens=1,
                        )
                        self.connector.produce_prefetch(
                            job.prefetch,
                            job.prefetch_slot,
                            plan,
                            stream,
                        )
                    else:
                        self.connector.cancel_prefetch(
                            job.prefetch, job.prefetch_slot
                        )
                finished = time.perf_counter()
                with self._lock:
                    self._stats["completed"] += 1
                    self._stats["event_wait_seconds"].append(
                        wait_finished - wait_started
                    )
                    self._stats["copy_and_shadow_seconds"].append(
                        finished - wait_finished
                    )
                    self._stats["total_seconds"].append(finished - job.submitted_at)
            except BaseException as exc:
                self._record_error(exc)


_HOOK_INSTALLED = False


def install_runner_hooks() -> None:
    """Install the opt-in MRV2 bridge hooks in the final vendor runtime.

    The hook intentionally refuses CPU PLE offload and only creates a reader on
    the first PP rank.  It is imported by the derived image but dormant until
    both VLLM_PLE_GDS and VLLM_PLE_GDS_RUNNER are enabled.
    """

    global _HOOK_INSTALLED
    if _HOOK_INSTALLED or not _enabled("VLLM_PLE_GDS_RUNNER"):
        return
    if not _enabled("VLLM_PLE_GDS"):
        raise RuntimeError("VLLM_PLE_GDS_RUNNER requires VLLM_PLE_GDS=1")
    if _enabled("VLLM_PLE_CPU_OFFLOAD"):
        raise RuntimeError("GDS PLE runner is mutually exclusive with CPU PLE offload")

    from vllm.distributed import get_pp_group
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.pp_utils import PPHandler
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    original_load_model = GPUModelRunner.load_model
    original_execute_model = GPUModelRunner.execute_model
    original_prepare_inputs = GPUModelRunner.prepare_inputs
    original_add_requests = GPUModelRunner.add_requests
    original_remove_request = GPUModelRunner._remove_request
    original_update_requests = GPUModelRunner.update_requests
    original_update_pp = GPUModelRunner.update_pp_decode_requests
    original_shutdown = GPUModelRunner.shutdown

    # Qwen3.8's NVFP4 target requires the Marlin backend, while its unquantized
    # MTP draft requires Triton. vLLM currently shares one runner backend in
    # VllmConfig, so scope the draft override to draft construction only.
    original_draft_load_model = DraftModelSpeculator.load_model

    def load_draft_model_with_backend(
        self: Any, target_model: Any
    ) -> Any:
        backend = os.getenv("VLLM_PLE_MTP_MOE_BACKEND", "").strip()
        if not backend:
            return original_draft_load_model(self, target_model)
        kernel_config = self.vllm_config.kernel_config
        previous = kernel_config.moe_backend
        kernel_config.moe_backend = backend
        # vLLM worker processes suppress INFO for non-vllm module loggers.
        # Keep this safety-critical, temporary backend override observable.
        logger.warning(
            "Temporarily using %s MoE backend for MTP draft construction "
            "(target backend=%s)",
            backend,
            previous,
        )
        try:
            return original_draft_load_model(self, target_model)
        finally:
            kernel_config.moe_backend = previous

    DraftModelSpeculator.load_model = load_draft_model_with_backend

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        # TP2 transport is opt-in; preserve the released PP2 implementation.
        if os.environ.get("Q38_TP_PLE") == "1":
            from vllm.distributed import get_tp_group
            from tp_ple_transport import TpPleTransport
            parallel = self.vllm_config.parallel_config
            assert parallel.pipeline_parallel_size == 1 and parallel.tensor_parallel_size == 2
            spec = self.vllm_config.speculative_config
            if spec is not None:
                assert spec.method == "mtp" and spec.num_speculative_tokens == 6
                logger.warning("Experimental TP2 input-driven GDS with native MTP6 enabled")
            assert _enabled("VLLM_PLE_GDS_INPUT_DRIVEN") and not _enabled("VLLM_PLE_GDS_IPC_PROCESS")
            layers = [m for m in self.model.modules() if getattr(m, "_ple_gds_patched", False)]
            assert len(layers) == 1
            layer = layers[0]
            owner_connector = None
            if get_tp_group().rank_in_group == 0:
                owner_connector = InputDrivenGdsPleConnector(
                    layer=layer, max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
                    max_num_reqs=self.scheduler_config.max_num_seqs,
                    input_ids_source=self.input_buffers.input_ids,
                    query_start_loc_source=self.model_state.ple_query_start_loc,
                    ngram_context_source=self.model_state.ngram_context, use_ipc_process=False)
            transport = TpPleTransport(layer, self.device, self.scheduler_config.max_num_batched_tokens, owner_connector, (self.input_buffers.input_ids, self.model_state.ple_query_start_loc, self.model_state.ngram_context))
            self._ple_gds_runner_connector = None
            self._ple_gds_input_connector = transport
            self.model_state._ple_gds_input_connector = transport
            assert self._ple_offload_connector is None
            self._ple_offload_connector = transport
            return result
        owner = get_pp_group().is_first_rank
        input_driven = _enabled("VLLM_PLE_GDS_INPUT_DRIVEN")
        layers = [
            module
            for module in self.model.modules()
            if getattr(module, "_ple_gds_patched", False)
        ]
        if owner and len(layers) != 1:
            raise RuntimeError(f"expected exactly one PP0 GDS PLE layer, found {len(layers)}")
        if not owner:
            self._ple_gds_runner_connector = None
            self._ple_gds_input_connector = None
            return result
        layer = layers[0]
        if input_driven:
            parallel = self.vllm_config.parallel_config
            if (
                int(parallel.pipeline_parallel_size) != 2
                or int(parallel.tensor_parallel_size) != 1
            ):
                raise RuntimeError("input-driven GDS PLE currently requires PP2/TP1")
            if getattr(self.vllm_config, "speculative_config", None) is not None:
                raise RuntimeError("input-driven GDS PLE currently requires MTP off")
            query_start_loc_source = getattr(
                self.model_state, "ple_query_start_loc", None
            )
            ngram_context_source = getattr(self.model_state, "ngram_context", None)
            if not isinstance(query_start_loc_source, torch.Tensor) or not isinstance(
                ngram_context_source, torch.Tensor
            ):
                raise RuntimeError("input-driven GDS PLE stable model inputs are missing")
            input_connector = InputDrivenGdsPleConnector(
                layer=layer,
                max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
                max_num_reqs=self.scheduler_config.max_num_seqs,
                input_ids_source=self.input_buffers.input_ids,
                query_start_loc_source=query_start_loc_source,
                ngram_context_source=ngram_context_source,
                use_ipc_process=_enabled("VLLM_PLE_GDS_IPC_PROCESS"),
            )
            if self._ple_offload_connector is not None:
                input_connector.close()
                raise RuntimeError("input-driven GDS PLE connector slot is already in use")
            self._ple_gds_runner_connector = None
            self._ple_gds_input_connector = input_connector
            self.model_state._ple_gds_input_connector = input_connector
            # Use the native connector lifecycle. GPUModelRunner calls this only
            # after model_state.prepare_inputs has finalized the stable inputs,
            # and also owns dummy/capture signaling, release, and shutdown.
            self._ple_offload_connector = input_connector
            if self.pp_handler is not None:
                self.pp_handler._ple_gds_input_driven = True
            return result
        shadow = CpuTokenShadow(
            eos_token_id=int(layer.eos_token_id), context_len=int(layer.ngram_size) - 1
        )
        connector = GdsPleRunnerConnector(
            layer=layer,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            shadow=shadow,
            owner=True,
            # The GDS producer process is an independent PLE transport choice.
            # Keep it decoupled from vLLM's scheduler mode so experiments can
            # compare IPC PLE against the synchronous reader with identical
            # ``--no-async-scheduling`` execution.
            use_ipc_process=_enabled("VLLM_PLE_GDS_IPC_PROCESS"),
        )
        self._ple_gds_runner_connector = connector
        self._ple_gds_input_connector = None
        self.model_state._ple_gds_runner_connector = connector
        if self.pp_handler is not None:
            self.pp_handler._ple_gds_shadow = shadow
            if not _mtp_enabled(self.pp_handler):
                control_worker = PpSampleControlWorker(
                    bridge=self.pp_handler._ple_cpu_bridge,
                    shadow=shadow,
                    req_idx_gen_np=self.pp_handler.req_idx_gen_np,
                    connector=connector,
                )
                self.pp_handler._ple_sample_control_worker = control_worker
                connector.control_worker = control_worker
        return result

    def add_requests(self: Any, scheduler_output: Any) -> None:
        result = original_add_requests(self, scheduler_output)
        connector = getattr(self, "_ple_gds_runner_connector", None)
        if connector is not None:
            _shadow_add_requests(connector.shadow, scheduler_output)
        return result

    def remove_request(self: Any, req_id: str) -> bool:
        connector = getattr(self, "_ple_gds_runner_connector", None)
        generation = None
        if connector is not None:
            try:
                generation = connector.shadow.generation(req_id)
            except TicketError:
                pass
        result = original_remove_request(self, req_id)
        if connector is not None:
            if generation is not None:
                connector.discard_prefetches(req_id, generation)
            connector.shadow.remove(req_id)
            if _mtp_enabled(getattr(self, "pp_handler", None)):
                getattr(self, "_ple_gds_actual_drafts", {}).pop(req_id, None)
        return result

    def update_requests(self: Any, scheduler_output: Any) -> None:
        result = original_update_requests(self, scheduler_output)
        connector = getattr(self, "_ple_gds_runner_connector", None)
        if connector is not None:
            _shadow_update_requests(connector.shadow, scheduler_output)
        return result

    def update_pp_decode_requests(self: Any) -> None:
        control_worker = getattr(
            getattr(self, "pp_handler", None), "_ple_sample_control_worker", None
        )
        if control_worker is not None:
            control_worker.raise_if_failed()
        result = original_update_pp(self)
        connector = getattr(self, "_ple_gds_runner_connector", None)
        pp_handler = getattr(self, "pp_handler", None)
        metadata = getattr(pp_handler, "_ple_last_cpu_metadata", None)
        if connector is not None and metadata is not None:
            connector.shadow.append_sampled(
                metadata.req_ids,
                metadata.generations,
                metadata.sampled_tokens,
                metadata.num_sampled,
                metadata.include_mask,
            )
            if _mtp_enabled(pp_handler):
                actual_drafts = getattr(self, "_ple_gds_actual_drafts", {})
                for row, (req_id, generation) in enumerate(
                    zip(metadata.req_ids, metadata.generations)
                ):
                    if bool(metadata.include_mask[row]):
                        actual_drafts[req_id] = (
                            int(generation),
                            tuple(int(value) for value in metadata.draft_tokens[row]),
                        )
                self._ple_gds_actual_drafts = actual_drafts
            self.pp_handler._ple_last_cpu_metadata = None
        return result

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        connector = getattr(self, "_ple_gds_runner_connector", None)
        if connector is not None:
            self.model_state._ple_gds_scheduler_output = scheduler_output
            self.model_state._ple_gds_dummy_run = bool(
                kwargs.get("dummy_run", False)
                or _is_vendor_warmup_scheduler_output(scheduler_output)
            )
        try:
            return original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            if connector is not None and connector._active_slot is not None:
                connector.release(self.main_stream)

    def prepare_inputs(
        self: Any, scheduler_output: Any, batch_req_state: Any, batch_desc: Any
    ) -> Any:
        connector = getattr(self, "_ple_gds_runner_connector", None)
        plan = None
        prefetch = None
        dummy_run = bool(getattr(self.model_state, "_ple_gds_dummy_run", False))
        mtp_enabled = _mtp_enabled(getattr(self, "pp_handler", None))
        if connector is not None and not dummy_run:
            if connector.control_worker is not None:
                connector.control_worker.raise_if_failed()
            req_ids = list(batch_req_state.req_ids)
            starts = self.req_states.num_computed_tokens_np[
                batch_req_state.idx_mapping_np
            ].astype(np.int64, copy=True)
            lengths = batch_req_state.num_scheduled_tokens.astype(np.int64, copy=True)
            effective_drafts: dict[str, Sequence[int]] = {}
            if mtp_enabled:
                actual_drafts = getattr(self, "_ple_gds_actual_drafts", {})
                for req_id in req_ids:
                    scheduled = tuple(
                        int(value)
                        for value in scheduler_output.scheduled_spec_decode_tokens.get(
                            req_id, ()
                        )
                    )
                    if scheduled and all(value < 0 for value in scheduled):
                        entry = actual_drafts.pop(req_id, None)
                        generation = connector.shadow.generation(req_id)
                        if entry is None or entry[0] != generation:
                            raise TicketError(
                                f"missing PP draft metadata for {req_id} generation "
                                f"{generation}"
                            )
                        resolved = entry[1][: len(scheduled)]
                        if len(resolved) != len(scheduled) or any(
                            value < 0 for value in resolved
                        ):
                            raise TicketError(
                                f"invalid PP draft metadata for {req_id}: {resolved}"
                            )
                        effective_drafts[req_id] = resolved
                    else:
                        effective_drafts[req_id] = scheduled
            if (
                connector.control_worker is not None
                and len(req_ids) == 1
                and int(lengths[0]) == 1
                and int(batch_desc.num_tokens) == 1
            ):
                generation = connector.shadow.generation(req_ids[0])
                prefetch = connector.find_prefetch(
                    req_id=req_ids[0],
                    generation=generation,
                    start=int(starts[0]),
                )
            else:
                plan = connector.shadow.plan(
                    step_id=connector.next_step_id(),
                    req_ids=req_ids,
                    starts=starts,
                    lengths=lengths,
                    drafts=effective_drafts,
                    padded_tokens=int(batch_desc.num_tokens),
                )
                connector.submit(plan)
        drafts = scheduler_output.scheduled_spec_decode_tokens if mtp_enabled else {}
        if mtp_enabled and drafts:
            bonus = int(self.model_state.num_new_sampled_tokens_per_step)
            draft_lengths = np.fromiter(
                (len(drafts.get(req_id, ())) for req_id in batch_req_state.req_ids),
                dtype=np.int32,
                count=len(batch_req_state.req_ids),
            )
            required = draft_lengths + bonus
            scheduled = batch_req_state.num_scheduled_tokens
            if not (scheduled >= required).all():
                if connector is not None:
                    connector.cancel_pending()
                logger.error(
                    "MTP verification input invariant failed: req_ids=%s "
                    "scheduled=%s draft_lengths=%s bonus=%s drafts=%s",
                    list(batch_req_state.req_ids),
                    scheduled.tolist(),
                    draft_lengths.tolist(),
                    bonus,
                    {key: list(value) for key, value in drafts.items()},
                )
        input_batch = original_prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc
        )
        if connector is not None and not dummy_run:
            actual = (
                tuple(input_batch.req_ids),
                tuple(int(value) for value in input_batch.num_computed_tokens_np),
                tuple(int(value) for value in input_batch.num_scheduled_tokens),
                int(input_batch.num_tokens),
                int(input_batch.num_tokens_after_padding),
            )
            if prefetch is not None:
                expected = (
                    (prefetch.req_id,),
                    (prefetch.start,),
                    (1,),
                    1,
                    1,
                )
            else:
                assert plan is not None
                expected = (
                    plan.identity.req_ids,
                    plan.identity.starts,
                    plan.identity.lengths,
                    plan.identity.valid_tokens,
                    plan.identity.padded_tokens,
                )
            if actual != expected:
                connector.cancel_pending()
                raise TicketError(
                    f"runner batch changed after ticket submission: {actual} != {expected}"
                )
            if prefetch is not None:
                input_batch._ple_gds_prefetch_identity = prefetch
            else:
                input_batch._ple_gds_ticket_identity = plan.identity
        return input_batch

    def shutdown(self: Any) -> None:
        connector = getattr(self, "_ple_gds_runner_connector", None)
        input_connector = getattr(self, "_ple_gds_input_connector", None)
        if connector is not None:
            if connector.control_worker is not None:
                connector.control_worker.close()
            connector.close()
            connector.layer.close_gds()
            self._ple_gds_runner_connector = None
        original_shutdown(self)
        if input_connector is not None:
            stats = input_connector.stats()
            if int(stats["close_count"]) != 1:
                raise TicketError(
                    "stock GPUModelRunner did not close input-driven PLE exactly once"
                )
            input_connector.layer.close_gds()
            self._ple_gds_input_connector = None
            self.model_state._ple_gds_input_connector = None

    GPUModelRunner.load_model = load_model
    GPUModelRunner.execute_model = execute_model
    GPUModelRunner.prepare_inputs = prepare_inputs
    GPUModelRunner.add_requests = add_requests
    GPUModelRunner._remove_request = remove_request
    GPUModelRunner.update_requests = update_requests
    GPUModelRunner.update_pp_decode_requests = update_pp_decode_requests
    GPUModelRunner.shutdown = shutdown

    original_pp_init = PPHandler.__init__
    original_pp_receive = PPHandler.receive
    original_pp_get = PPHandler.get_prev_sampled_outputs

    def pp_init(
        self: Any,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ) -> None:
        original_pp_init(self, max_num_reqs, num_speculative_steps, device)
        # ``speculative_config=None`` has zero speculative steps.  The sampled
        # token bridge is still required by the GDS CPU shadow for ordinary
        # decode, but its draft-token side channel remains disabled.
        self._ple_mtp_enabled = int(num_speculative_steps) > 0
        self._ple_gds_input_driven = _enabled("VLLM_PLE_GDS_INPUT_DRIVEN")
        if not self.is_last_rank:
            if self._ple_gds_input_driven:
                self._ple_cpu_bridge = None
                self._ple_last_cpu_metadata = None
                self._ple_shadow_pending = deque()
                self._ple_sample_control_worker = None
                return
            depth = get_pp_group().world_size + 1
            self._ple_cpu_bridge = PpPinnedSampleBridge(
                max_num_reqs, num_speculative_steps + 1, depth
            )
            self._ple_last_cpu_metadata = None
            self._ple_shadow_pending = deque()
            self._ple_sample_control_worker = None

    def pp_receive(self: Any, input_batch: Any) -> bool:
        result = original_pp_receive(self, input_batch)
        if bool(getattr(self, "_ple_gds_input_driven", False)):
            return result
        pending = self.queue[-1]
        if pending is not None:
            req_ids = tuple(input_batch.req_ids)
            if _are_vendor_warmup_req_ids(req_ids):
                return result
            with torch.cuda.stream(self.broadcast_stream):
                pending._ple_cpu_slot = self._ple_cpu_bridge.stage(
                    sampled_tokens=pending.sampled_tokens,
                    num_sampled=pending.num_sampled,
                    num_rejected=pending.num_rejected,
                    draft_tokens=pending.draft_tokens if _mtp_enabled(self) else None,
                    stream=self.broadcast_stream,
                )
            pending._ple_req_ids = req_ids
            shadow = getattr(self, "_ple_gds_shadow", None)
            if shadow is None:
                raise RuntimeError("PP0 GDS PLE shadow is not attached")
            generations = tuple(shadow.generation(req_id) for req_id in input_batch.req_ids)
            pending._ple_generations = generations
            if _mtp_enabled(self):
                self._ple_shadow_pending.append(pending)
            else:
                control_worker = self._ple_sample_control_worker
                if control_worker is None:
                    raise RuntimeError("PP0 GDS PLE sample control worker is not attached")
                control_worker.submit(
                    slot=pending._ple_cpu_slot,
                    req_ids=pending._ple_req_ids,
                    generations=pending._ple_generations,
                    idx_mapping_np=pending.idx_mapping_np,
                    need_sampled_mask=pending.need_sampled_mask,
                    gen_at_receive_np=pending.gen_at_receive_np,
                )
        return result

    def pp_get(self: Any) -> dict[str, torch.Tensor] | None:
        pending = self.queue[0] if self.queue else None
        result = original_pp_get(self)
        if _mtp_enabled(self) and pending is not None and hasattr(
            pending, "_ple_cpu_slot"
        ):
            shadow_pending = self._ple_shadow_pending
            if not shadow_pending or shadow_pending[0] is not pending:
                raise TicketError("PP sampled metadata FIFO diverged from vendor FIFO")
            metadata_pending = shadow_pending.popleft()
            rows = len(metadata_pending._ple_req_ids)
            sampled, num_sampled, num_rejected, draft_tokens = (
                self._ple_cpu_bridge.poll(metadata_pending._ple_cpu_slot, rows)
            )
            freed = (
                self.req_idx_gen_np[metadata_pending.idx_mapping_np]
                != metadata_pending.gen_at_receive_np
            )
            include = np.asarray(
                metadata_pending.need_sampled_mask & ~freed, dtype=np.bool_
            )
            self._ple_last_cpu_metadata = CpuSampleMetadata(
                req_ids=metadata_pending._ple_req_ids,
                generations=metadata_pending._ple_generations,
                sampled_tokens=sampled,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                draft_tokens=draft_tokens,
                include_mask=include,
            )
        return result

    PPHandler.__init__ = pp_init
    PPHandler.receive = pp_receive
    PPHandler.get_prev_sampled_outputs = pp_get

    from vllm.models.qwen3_8_flash_next.nvidia.model_state import (
        Qwen3_8FlashNextModelState,
    )

    original_prepare = Qwen3_8FlashNextModelState.prepare_inputs
    original_dummy = Qwen3_8FlashNextModelState.prepare_dummy_inputs

    def prepare_inputs(self: Any, input_batch: Any, req_states: Any) -> dict[str, Any]:
        result = original_prepare(self, input_batch, req_states)
        input_connector = getattr(self, "_ple_gds_input_connector", None)
        if input_connector is not None:
            return result
        connector = getattr(self, "_ple_gds_runner_connector", None)
        if connector is None:
            return result
        if bool(getattr(self, "_ple_gds_dummy_run", False)):
            connector.prepare_dummy(int(input_batch.num_tokens_after_padding))
            return result
        identity = getattr(input_batch, "_ple_gds_ticket_identity", None)
        prefetch = getattr(input_batch, "_ple_gds_prefetch_identity", None)
        if identity is None and prefetch is None:
            raise TicketError("runner did not attach a GDS PLE ticket identity")
        stream = torch.cuda.current_stream(connector.device)
        if prefetch is not None:
            connector.consume_prefetch(prefetch, stream)
        else:
            connector.consume(identity, stream)
        return result

    def prepare_dummy_inputs(self: Any, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        result = original_dummy(self, num_reqs, num_tokens)
        connector = getattr(self, "_ple_gds_runner_connector", None)
        if connector is not None:
            connector.prepare_dummy(num_tokens)
        return result

    Qwen3_8FlashNextModelState.prepare_inputs = prepare_inputs
    Qwen3_8FlashNextModelState.prepare_dummy_inputs = prepare_dummy_inputs
    from hc_gemv_runtime import install as install_hc_gemv
    install_hc_gemv(GPUModelRunner)
    from projection_runtime import install as install_projection
    install_projection(GPUModelRunner)
    from triton_preload import install as install_preload
    install_preload(GPUModelRunner)
    from draft_int8_runtime import install as install_draft_int8
    if os.environ.get("Q38_DRAFT_INT8", "1") == "1":
        install_draft_int8()
    _HOOK_INSTALLED = True
    logger.info("installed bounded MRV2 GDS PLE runner hooks")


__all__ = [
    "BatchIdentity",
    "CpuBatchPlan",
    "CpuTokenShadow",
    "GdsPleRunnerConnector",
    "InputDrivenGdsPleConnector",
    "LateSampleMetadata",
    "MailboxFull",
    "PpPinnedSampleBridge",
    "PpSampleControlWorker",
    "TicketError",
    "_are_vendor_warmup_req_ids",
    "_is_vendor_warmup_req_id",
    "_is_vendor_warmup_scheduler_output",
    "_mtp_enabled",
    "_shadow_add_requests",
    "_shadow_update_requests",
    "install_runner_hooks",
]
