"""Strict GDS-backed Qwen3.8 Flash Next N-gram embedding integration.

This module is opt-in and deliberately fail-closed.  It replaces only the
large FP8 embedding table; Qwen's hash buffers and PLE dequantization remain
the stock implementation.  The caller must provide CPU-resident token metadata
so the production path never synchronizes GPU token IDs back to the host.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import nn

from ple_gds.compact import load_current_compact
from ple_gds.persistent_reader import PersistentGdsPleReader

logger = logging.getLogger("vllm.ple_gds")

ENV_ENABLE = "VLLM_PLE_GDS"
EXPECTED_ROW_COUNT = 320_001_536
EXPECTED_ROW_BYTES = 160
EXPECTED_DTYPE = "F8_E4M3"
EXPECTED_SCALE_DTYPE = "BF16"

_PLE_SHARD_RE = re.compile(
    r"(?:^|\.)ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$"
)
_TRUE = frozenset(("1", "true", "yes", "on"))
_REGISTRY: weakref.WeakValueDictionary[str, nn.Module] = weakref.WeakValueDictionary()
_OP_NAME = "ple_gds_lookup"
_HOOK_LOCK = threading.Lock()
_WEIGHT_SKIP_INSTALLED = False


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def enabled() -> bool:
    return _enabled(ENV_ENABLE)


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bdf(value: str) -> str:
    text = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text):
        text = "0000:" + text
    match = re.fullmatch(r"([0-9a-f]{8}):([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", text)
    if match:
        text = match.group(1)[-4:] + ":" + match.group(2)
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text):
        raise ValueError(f"invalid PCI BDF {value!r}")
    return text


def _canonical_uuid(value: str) -> str:
    text = value.strip().upper()
    return text if text.startswith("GPU-") else "GPU-" + text


def _validate_mode_environment() -> None:
    conflicts = [
        name
        for name in ("VLLM_PLE_MMAP", "VLLM_PLE_CPU_OFFLOAD", "VLLM_PLE_NVFP4_GPU")
        if _enabled(name)
    ]
    if conflicts:
        raise RuntimeError("VLLM_PLE_GDS is mutually exclusive with " + ", ".join(conflicts))
    if _env_int("VLLM_PLE_GDS_CACHE_ROWS", 0) != 0:
        raise RuntimeError("VLLM_PLE_GDS_CACHE_ROWS must be exactly 0")
    if os.environ.get("CUFILE_ENV_PATH_JSON") is None:
        raise RuntimeError("CUFILE_ENV_PATH_JSON must name the accepted strict cuFile JSON")


def _validate_artifact(artifact: Path) -> tuple[dict[str, Any], Path]:
    identity_path = os.environ.get("Q38_PLE_IDENTITY")
    if not identity_path:
        raise RuntimeError("Q38_PLE_IDENTITY must name an enrolled PLE identity JSON")
    identity = json.loads(Path(identity_path).read_text())
    current = artifact / "CURRENT"
    if _sha256(current) != identity["current_sha256"]:
        raise RuntimeError("GDS PLE CURRENT identity mismatch")
    metadata, generation = load_current_compact(
        artifact,
        check_sources=_enabled("VLLM_PLE_GDS_OFFLINE_AUDIT"),
        check_data=_enabled("VLLM_PLE_GDS_OFFLINE_AUDIT"),
    )
    metadata_path = generation / "ple-gds-metadata.json"
    if generation.name != identity["generation"]:
        raise RuntimeError(f"unexpected GDS PLE generation {generation.name}")
    if _sha256(metadata_path) != identity["metadata_sha256"]:
        raise RuntimeError("GDS PLE metadata identity mismatch")
    component = next(
        (item for item in metadata["components"] if item["name"] == "ngram_embedding"),
        None,
    )
    scale = next((item for item in metadata["globals"] if item["name"] == "weight_scale"), None)
    checks = {
        "row_count": int(metadata["row_count"]) == EXPECTED_ROW_COUNT,
        "row_bytes": component is not None and int(component["row_bytes"]) == EXPECTED_ROW_BYTES,
        "dtype": component is not None and component.get("dtype") == EXPECTED_DTYPE,
        "scale": scale is not None and scale.get("dtype") == EXPECTED_SCALE_DTYPE
        and int(scale["size"]) == 2,
        "data_identity": metadata.get("data_sha256") == identity["data_sha256"],
        "source_identity": metadata.get("source_identity_sha256") == identity["source_identity_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("GDS PLE artifact validation failed: " + ", ".join(failed))
    return metadata, generation


def _query_gpu(uuid: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wanted = _canonical_uuid(uuid)
    matches: list[str] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and _canonical_uuid(fields[0]) == wanted:
            matches.append(_canonical_bdf(fields[1]))
    if len(matches) != 1:
        raise RuntimeError(f"nvidia-smi found {len(matches)} entries for target GPU {wanted}")
    return matches[0]


def _validate_gpu(uuid: str, bdf: str) -> None:
    actual_bdf = _query_gpu(uuid)
    expected_bdf = _canonical_bdf(bdf)
    if actual_bdf != expected_bdf:
        raise RuntimeError(f"target GPU BDF mismatch: {actual_bdf} != {expected_bdf}")
    wanted = _canonical_uuid(uuid).removeprefix("GPU-").lower()
    visible = [
        index
        for index in range(torch.cuda.device_count())
        if str(torch.cuda.get_device_properties(index).uuid).lower() == wanted
    ]
    if len(visible) != 1:
        raise RuntimeError(f"target GPU must be visible exactly once to PyTorch, got {visible}")


class _GdsNgramEmbedding(nn.Module):
    """Geometry-only replacement for VocabParallelEmbedding; no table weight."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("GDS placeholder embedding cannot perform a resident lookup")


def should_skip_ple_weight(name: str) -> bool:
    """Return whether *name* is exactly a checkpoint PLE shard payload."""

    return _PLE_SHARD_RE.search(name) is not None


def install_weight_skip_hook() -> None:
    """Skip PLE shard payloads before lazy safetensors calls get_tensor()."""

    global _WEIGHT_SKIP_INSTALLED
    with _HOOK_LOCK:
        if _WEIGHT_SKIP_INSTALLED:
            return
        import vllm.model_executor.model_loader.weight_utils as weight_utils

        original = weight_utils.should_skip_weight
        if getattr(original, "_ple_gds_wrapper", False):
            _WEIGHT_SKIP_INSTALLED = True
            return

        def wrapped(name: str, local_expert_ids: set[int] | None) -> bool:
            return should_skip_ple_weight(name) or original(name, local_expert_ids)

        wrapped._ple_gds_wrapper = True  # type: ignore[attr-defined]
        wrapped._ple_gds_original = original  # type: ignore[attr-defined]
        weight_utils.should_skip_weight = wrapped
        _WEIGHT_SKIP_INSTALLED = True


def _require_cpu_integer_tensor(name: str, value: torch.Tensor, dims: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_cuda or value.device.type != "cpu":
        raise TypeError(f"{name} must be prepared on CPU; GPU-to-CPU hot-path sync is forbidden")
    if value.ndim != dims:
        raise ValueError(f"{name} must be {dims}D")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    return value.contiguous().to(dtype=torch.int64)


def _plan_ngram_row_ids_reference(
    layer: nn.Module,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> torch.Tensor:
    """Compute stock-Qwen N-gram row IDs entirely on CPU.

    The returned one-dimensional int64 tensor is ordered exactly as the stock
    embedding lookup: token-major, then ngram-head-major. CUDA-graph padding is
    deterministic but never scattered into the valid-token pack workspace.
    """

    input_ids = _require_cpu_integer_tensor("input_ids", input_ids.reshape(-1), 1)
    query_start_loc = _require_cpu_integer_tensor("query_start_loc", query_start_loc, 1)
    ngram_context = _require_cpu_integer_tensor("ngram_context", ngram_context, 2)
    num_tokens = input_ids.numel()
    if query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must describe at least one request")
    num_reqs = query_start_loc.numel() - 1
    if ngram_context.shape != (num_reqs, int(layer.ngram_size) - 1):
        raise ValueError(
            "ngram_context shape mismatch: "
            f"got {tuple(ngram_context.shape)}, need {(num_reqs, int(layer.ngram_size) - 1)}"
        )
    if int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc must start at zero")
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    if torch.any(lengths < 0):
        raise ValueError("query_start_loc must be nondecreasing")
    num_valid_tokens = int(query_start_loc[-1])
    if num_valid_tokens < 0 or num_valid_tokens > num_tokens:
        raise ValueError("query_start_loc ends outside input_ids")
    if num_tokens == 1 and num_reqs == 1 and num_valid_tokens == 1:
        token = input_ids.numpy()[0]
        context_values = ngram_context.numpy()[0]
        eos = np.int64(layer.eos_token_id)
        previous_1 = eos if context_values[-1] == eos else context_values[-1]
        previous_2 = (
            eos
            if context_values[-1] == eos or context_values[-2] == eos
            else context_values[-2]
        )
        multipliers = layer._ple_gds_multipliers_numpy
        sizes = layer._ple_gds_vocab_sizes_numpy
        offsets = layer._ple_gds_offsets_numpy
        with np.errstate(over="ignore"):
            mixed_2 = np.bitwise_xor(token * multipliers[0], previous_1 * multipliers[1])
            mixed_3 = np.bitwise_xor(mixed_2, previous_2 * multipliers[2])
        first = np.remainder(mixed_2, sizes[: layer.heads_per_ngram])
        first = first + offsets[: layer.heads_per_ngram]
        second = np.remainder(mixed_3, sizes[layer.heads_per_ngram :])
        second = second + offsets[layer.heads_per_ngram :]
        return torch.from_numpy(np.concatenate((first, second)).astype(np.int64, copy=False))
    max_seq_len = max(1, int(lengths.max()))

    positions = torch.arange(num_tokens, dtype=torch.int64)
    packed = torch.full(
        (num_reqs, max_seq_len), int(layer.eos_token_id), dtype=torch.int64
    )
    request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
    request_indices.clamp_(min=0, max=num_reqs - 1)
    columns = (positions - query_start_loc[request_indices]).clamp(0, max_seq_len - 1)
    packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = input_ids[
        :num_valid_tokens
    ]
    context = torch.cat((ngram_context, packed), dim=-1)
    positions_2d, position_in_segment = layer._shift_precompute(
        context, int(layer.eos_token_id)
    )
    shifted = [context]
    for shift in range(1, int(layer.ngram_size)):
        shifted.append(
            layer._shift_apply(
                context,
                positions_2d,
                position_in_segment,
                shift,
                int(layer.eos_token_id),
            )
        )
    multipliers = layer._ple_gds_layer_multipliers_cpu
    vocab_sizes = layer._ple_gds_vocab_sizes_cpu
    offsets = layer._ple_gds_offsets_cpu
    adjusted_columns = columns + int(layer.ngram_size) - 1
    blocks: list[torch.Tensor] = []
    for ngram in range(2, int(layer.ngram_size) + 1):
        start = (ngram - 2) * int(layer.heads_per_ngram)
        end = start + int(layer.heads_per_ngram)
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
        ids = torch.remainder(mixed.unsqueeze(-1), vocab_sizes[start:end]) + offsets[start:end]
        blocks.append(ids[request_indices, adjusted_columns])
    result = torch.cat(blocks, dim=-1).reshape(-1).contiguous()
    if torch.any(result < 0) or torch.any(result >= int(layer._ple_gds_reader.row_count)):
        raise IndexError("Qwen N-gram planner produced an out-of-range artifact row")
    return result


def plan_ngram_row_ids(layer, input_ids, query_start_loc, ngram_context):
    if not (isinstance(input_ids, torch.Tensor) and isinstance(query_start_loc, torch.Tensor) and isinstance(ngram_context, torch.Tensor) and (input_ids.numel() == 7) and (query_start_loc.shape == (2,)) and (ngram_context.shape == (1, 2)) and (layer.ngram_size == 3) and all((t.device.type == 'cpu' and t.dtype in (torch.int32, torch.int64) for t in (input_ids, query_start_loc, ngram_context)))):
        return _plan_ngram_row_ids_reference(layer, input_ids, query_start_loc, ngram_context)
    tokens = _require_cpu_integer_tensor('input_ids', input_ids.reshape(-1), 1).numpy()
    qsl = _require_cpu_integer_tensor('query_start_loc', query_start_loc, 1).numpy()
    context = _require_cpu_integer_tensor('ngram_context', ngram_context, 2).numpy()[0]
    if qsl[0] != 0:
        raise ValueError('query_start_loc must start at zero')
    valid = int(qsl[1])
    if valid < 0:
        raise ValueError('query_start_loc must be nondecreasing')
    if valid > 7:
        raise ValueError('query_start_loc ends outside input_ids')
    eos = np.int64(layer.eos_token_id)
    seq = np.concatenate((context, tokens[:valid] if valid else np.array([eos])))
    indices = np.minimum(np.arange(7), max(1, valid) - 1) + 2
    current = seq[indices]
    previous_1 = seq[indices - 1]
    previous_2 = np.where(previous_1 == eos, eos, seq[indices - 2])
    mul = layer._ple_gds_multipliers_numpy
    sizes = layer._ple_gds_vocab_sizes_numpy
    offsets = layer._ple_gds_offsets_numpy
    h = layer.heads_per_ngram
    with np.errstate(over='ignore'):
        mixed_2 = current * mul[0] ^ previous_1 * mul[1]
        mixed_3 = mixed_2 ^ previous_2 * mul[2]
    result = np.empty((7, 2 * h), dtype=np.int64)
    result[:, :h] = mixed_2[:, None] % sizes[:h] + offsets[:h]
    result[:, h:] = mixed_3[:, None] % sizes[h:] + offsets[h:]
    if np.any(result < 0) or np.any(result >= layer._ple_gds_reader.row_count):
        raise IndexError('Qwen N-gram planner produced an out-of-range artifact row')
    return torch.from_numpy(result.reshape(-1))

def _lookup_into(layer: nn.Module, row_ids: torch.Tensor, output: torch.Tensor) -> dict[str, Any]:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cuFile host I/O cannot execute during CUDA Graph capture")
    reader = layer._ple_gds_reader
    if reader is None:
        raise RuntimeError("non-owner PP rank has no GDS PLE reader")
    stream = torch.cuda.current_stream(reader.device)
    cursor = 0
    chunks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for lease in reader.iter_read(row_ids, mode="sync_mt", stream=stream, measure_gpu=False):
        try:
            count = lease.tensor.shape[0]
            output.reshape(-1, layer.head_dim)[cursor : cursor + count].copy_(lease.tensor)
            cursor += count
            chunks.append(lease.metrics.as_dict())
        finally:
            lease.release(stream)
    if cursor != row_ids.numel():
        raise RuntimeError(f"GDS lookup wrote {cursor} rows, expected {row_ids.numel()}")
    return {
        "row_count": row_ids.numel(),
        "chunk_count": len(chunks),
        "lookup_seconds": time.perf_counter() - started,
        "chunks": chunks,
    }


def _lookup_impl(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = _REGISTRY.get(layer_name)
    if layer is None:
        raise RuntimeError(f"GDS PLE layer {layer_name!r} is not registered")
    planning_start = time.perf_counter()
    row_ids = plan_ngram_row_ids(layer, input_ids, query_start_loc, ngram_context)
    planning_seconds = time.perf_counter() - planning_start
    metrics = _lookup_into(layer, row_ids, output)
    metrics["row_id_planning_seconds"] = planning_seconds
    metrics["end_to_end_seconds"] = planning_seconds + metrics["lookup_seconds"]
    layer._ple_gds_last_metrics = metrics


def _lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_lookup_impl,
        mutates_args=["output"],
        fake_impl=_lookup_fake,
    )


def _owner_rank() -> bool:
    test_value = os.environ.get("VLLM_PLE_GDS_TEST_OWNER")
    if test_value is not None:
        if not _enabled("VLLM_PLE_GDS_TESTING"):
            raise RuntimeError("VLLM_PLE_GDS_TEST_OWNER requires VLLM_PLE_GDS_TESTING=1")
        return test_value.strip().lower() in _TRUE
    try:
        from vllm.distributed import get_pp_group

        if os.environ.get("Q38_TP_PLE") == "1":
            from vllm.distributed import get_tp_group
            assert get_pp_group().world_size == 1 and get_tp_group().world_size == 2
            return get_tp_group().rank_in_group == 0
        return bool(get_pp_group().is_first_rank)
    except Exception as group_error:
        try:
            from vllm.config import get_current_vllm_config

            pp_size = get_current_vllm_config().parallel_config.pipeline_parallel_size
        except Exception as config_error:
            raise RuntimeError("cannot determine GDS PLE PP ownership") from config_error
        if int(pp_size) == 1:
            return True
        raise RuntimeError("PP group is unavailable; refusing to guess GDS PLE ownership") from group_error


def _make_reader(artifact: Path, uuid: str, bdf: str) -> PersistentGdsPleReader:
    return PersistentGdsPleReader(
        artifact,
        component="ngram_embedding",
        expected_gpu_uuid=uuid,
        expected_gpu_bdf=bdf,
        slot_count=_env_int("VLLM_PLE_GDS_STAGING_SLOTS", 3),
        staging_bytes=_env_int("VLLM_PLE_GDS_STAGING_MIB", 32) * 1024 * 1024,
        output_bytes=_env_int("VLLM_PLE_GDS_OUTPUT_MIB", 32) * 1024 * 1024,
        worker_count=_env_int("VLLM_PLE_GDS_WORKERS", 32),
        pin_workers=True,
        poll_mode=True,
        max_gap_pages=0,
        max_range_bytes=1024 * 1024,
        cache_capacity_rows=0,
        enable_diagnostic_batch=False,
        enable_diagnostic_async=False,
        offline_integrity_audit=False,
    )


def apply(cls: type) -> None:
    """Patch ``Qwen3_8FlashNextNGramEmbedding`` when VLLM_PLE_GDS=1."""

    if not enabled():
        return
    if getattr(cls, "_ple_gds_patched", False):
        return
    _validate_mode_environment()
    artifact = Path(os.environ["VLLM_PLE_GDS_ARTIFACT"]).resolve()
    uuid = _canonical_uuid(os.environ["VLLM_PLE_GDS_GPU_UUID"])
    bdf = _canonical_bdf(os.environ["VLLM_PLE_GDS_GPU_BDF"])
    install_weight_skip_hook()
    _register_op()

    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def patched_init(
        self: nn.Module,
        config: Any,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        quant_config: Any = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        try:
            from vllm.config import get_current_vllm_config

            current_config = get_current_vllm_config()
        except Exception:
            # Standalone module tests do not install a current vLLM config.
            current_config = None
        if current_config is not None:
            strategy = current_config.load_config.safetensors_load_strategy
            if strategy not in (None, "lazy"):
                raise RuntimeError(
                    "VLLM_PLE_GDS requires lazy safetensors loading so PLE shards "
                    "are skipped before payload materialization"
                )
        owner = _owner_rank()
        if owner:
            metadata, _generation = _validate_artifact(artifact)
            _validate_gpu(uuid, bdf)
        real_embedding = mod.VocabParallelEmbedding
        mod.VocabParallelEmbedding = lambda n, d, **_kw: _GdsNgramEmbedding(n, d)
        try:
            orig_init(
                self,
                config,
                embedding_dim,
                ple_dense_layer_id,
                max_total_tokens,
                max_num_reqs,
                prefix,
                quant_config=None,
                params_dtype=params_dtype,
            )
        finally:
            mod.VocabParallelEmbedding = real_embedding
        if int(self.head_dim) != EXPECTED_ROW_BYTES:
            raise RuntimeError(f"Qwen head_dim {self.head_dim} != artifact row bytes {EXPECTED_ROW_BYTES}")
        if int(self.ngram_embedding.org_vocab_size) != EXPECTED_ROW_COUNT:
            raise RuntimeError(
                "Qwen N-gram vocabulary does not match artifact rows: "
                f"{self.ngram_embedding.org_vocab_size} != {EXPECTED_ROW_COUNT}"
            )
        self._ple_gds_prefix = prefix
        self._ple_gds_layer_multipliers_cpu = self.layer_multipliers.detach().cpu().contiguous()
        self._ple_gds_vocab_sizes_cpu = self.ngram_heads_vocab_sizes.detach().cpu().contiguous()
        self._ple_gds_offsets_cpu = self.ngram_heads_offsets.detach().cpu().contiguous()
        self._ple_gds_multipliers_numpy = self._ple_gds_layer_multipliers_cpu.numpy()
        self._ple_gds_vocab_sizes_numpy = self._ple_gds_vocab_sizes_cpu.numpy()
        self._ple_gds_offsets_numpy = self._ple_gds_offsets_cpu.numpy()
        self._ple_gds_reader = _make_reader(artifact, uuid, bdf) if owner else None
        self._ple_gds_last_metrics = None
        if self._ple_gds_reader is not None:
            self.register_buffer(
                "_offload_weight_scale",
                self._ple_gds_reader.global_scale.detach().clone(),
                persistent=False,
            )
            _REGISTRY[prefix] = self
        logger.info(
            "PLE GDS: %s owner=%s rows=%d head_dim=%d heads=%d backend=sync_mt",
            prefix,
            self._ple_gds_reader is not None,
            EXPECTED_ROW_COUNT,
            int(self.head_dim),
            int(self.ngram_heads),
        )

    def patched_load_weights(
        self: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        retained: set[str] = set()
        remaining: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if re.fullmatch(r"ngram_embedding\.shard_\d+\.weight", name):
                raise RuntimeError("PLE shard reached module loader; pre-read skip hook failed")
            if name == "ngram_embedding.weight_scale":
                if self._ple_gds_reader is None:
                    retained.add(name)
                    continue
                candidate = weight.detach().to(
                    device=self._offload_weight_scale.device, dtype=torch.bfloat16
                )
                if not torch.equal(candidate.view(torch.uint8), self._offload_weight_scale.view(torch.uint8)):
                    raise RuntimeError("checkpoint PLE weight_scale differs from GDS artifact")
                self._offload_weight_scale.copy_(candidate)
                retained.add(name)
                continue
            remaining.append((name, weight))
        retained.update(orig_load_weights(self, remaining))
        return retained

    def patched_forward_impl(
        self: nn.Module,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del output_buffer
        input_connector = getattr(self, "_ple_gds_input_connector", None)
        if input_connector is not None:
            return input_connector.consume_output(hidden_states, input_ids)
        runner_output = getattr(self, "_ple_gds_runner_output", None)
        if runner_output is not None:
            num_tokens = input_ids.reshape(-1).numel()
            # TorchDynamo must retain this dimension as symbolic for graph
            # capture.  Keep the fail-closed identity check in eager mode,
            # where the Python integer comparison is safe.
            if not torch.compiler.is_compiling():
                active_tokens = int(getattr(self, "_ple_gds_runner_active_tokens", 0))
                if active_tokens != num_tokens:
                    raise RuntimeError(
                        "GDS PLE runner output identity mismatch: "
                        f"active={active_tokens}, model={num_tokens}"
                    )
            return runner_output[:num_tokens, : int(self.embedding_dim)]
        reader = self._ple_gds_reader
        if reader is None:
            raise RuntimeError("non-owner PP rank attempted a GDS PLE lookup")
        if hidden_states.device != reader.device:
            raise RuntimeError(
                f"PLE consumer is on {hidden_states.device}, expected target GPU {reader.device}"
            )
        num_tokens = input_ids.reshape(-1).numel()
        output = torch.empty(
            (num_tokens, int(self.embedding_dim)),
            dtype=torch.float8_e4m3fn,
            device=reader.device,
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids, query_start_loc, ngram_context, output, self._ple_gds_prefix
        )
        return output

    def close_gds(self: nn.Module) -> None:
        reader = getattr(self, "_ple_gds_reader", None)
        if reader is not None:
            reader.close()
            self._ple_gds_reader = None
        _REGISTRY.pop(getattr(self, "_ple_gds_prefix", ""), None)

    cls._ple_gds_orig_forward_impl = cls.forward_impl
    cls.__init__ = patched_init
    cls.load_weights = patched_load_weights
    cls.forward_impl = patched_forward_impl
    cls.close_gds = close_gds
    cls.plan_gds_row_ids = plan_ngram_row_ids
    cls._ple_gds_patched = True
    logger.info("PLE GDS patch applied to %s.%s", cls.__module__, cls.__name__)


__all__ = [
    "apply",
    "enabled",
    "install_weight_skip_hook",
    "plan_ngram_row_ids",
    "should_skip_ple_weight",
]
