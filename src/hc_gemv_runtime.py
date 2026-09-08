"""Pinned SM80 HC projection replacement; ordinary vLLM graph capture/replay.

Q38_HC_GEMV=0 disables installation before compilation. No request controls,
alternative graphs, tensor snapshots, or per-token instrumentation are installed.
Unsupported tensor shapes/layouts/dtypes use the original torch.mm.
"""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path

import torch
from hc_gemv_kernels import row_gemv

ENABLED = os.environ.get('Q38_HC_GEMV', '1') == '1'
SHAPES = {(10240, 336), (10240, 320), (320, 10240)}
_installed = False
_record_capture = False
_captured = []


def supported(x, wt, out):
    return (x.ndim == wt.ndim == out.ndim == 2
            and tuple(wt.shape) in SHAPES
            and tuple(x.shape) == (1, wt.shape[0])
            and tuple(out.shape) == (1, wt.shape[1])
            and x.dtype == wt.dtype == out.dtype == torch.bfloat16
            and x.is_cuda and x.device == wt.device == out.device
            and x.stride(1) == out.stride(1) == 1
            and wt.stride() == (1, wt.shape[0]))


def mm(x, wt, *, out, site='direct'):
    if not ENABLED or not supported(x, wt, out):
        return torch.mm(x, wt, out=out)
    if torch.cuda.get_device_capability(x.device) != (8, 0):
        return torch.mm(x, wt, out=out)
    k, n = wt.shape
    bn, bk = (1, 1024) if k == 10240 else (16, 128)
    row_gemv[((n + bn - 1) // bn,)](x, wt.T, out, n, k, bn, bk, num_warps=4)
    if _record_capture and torch.cuda.is_current_stream_capturing():
        _captured.append(dict(site=site, weight_shape=[k, n]))
    return out


def instrument_source(source, key):
    # Restrict this version-specific rewrite to generated HC functions and exact
    # reinterpret_tensor shape/stride arguments; never use a loose text match.
    if 'qwen3_8_flash_next_hc_gate_mix.default' not in source:
        return source, 0
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name != 'call':
            continue
        for node in ast.walk(fn):
            value = node.value if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
            if not isinstance(value, ast.Call) or ast.unparse(value.func) != 'extern_kernels.mm':
                continue
            if len(value.args) != 2 or not isinstance(value.args[1], ast.Call):
                continue
            wt = value.args[1]
            if ast.unparse(wt.func) != 'reinterpret_tensor' or len(wt.args) < 3:
                continue
            try:
                shape, stride = ast.literal_eval(wt.args[1]), ast.literal_eval(wt.args[2])
            except (ValueError, TypeError):
                continue
            if shape not in SHAPES or stride != (1, shape[0]):
                continue
            assert node.lineno == node.end_lineno, 'Unsupported generated multiline HC call'
            value.func = ast.parse('_hc_gemv.mm', mode='eval').body
            value.keywords.append(ast.keyword(arg='site', value=ast.Constant(f'{key}:{node.lineno}')))
            edits.append((node.lineno - 1, ast.unparse(node)))
    for i, rewritten in sorted(edits, reverse=True):
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = indent + rewritten + '\n'
    if edits:
        assert 'from __future__ import' not in source
        return 'import hc_gemv_runtime as _hc_gemv\n' + ''.join(lines), len(edits)
    return source, 0


def install(_runner_cls=None):
    global _installed
    if _installed or not ENABLED:
        return
    # This isolated package is qualified only for SM80. Other architectures keep
    # the original implementation; no device tuning is queried or changed.
    import torch._inductor.codecache as cc
    import torch._inductor.runtime.compile_tasks as ct
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    old_reload = cc._reload_python_module
    changed = set()
    sources = []

    def reload(key, path, *args, **kwargs):
        if path not in changed:
            original = Path(path).read_text()
            if 'import hc_gemv_runtime as _hc_gemv' not in original:
                source, count = instrument_source(original, key)
                if count:
                    Path(path).write_text(source)
                    sources.append(dict(key=key, count=count,
                        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        patched_sha256=hashlib.sha256(source.encode()).hexdigest()))
            changed.add(path)
        return old_reload(key, path, *args, **kwargs)

    old_capture = CudaGraphManager.capture

    @functools.wraps(old_capture)
    def capture(self, create_forward_fn, *args, **kwargs):
        if self.vllm_config.speculative_config is not None:
            # The no-MTP M=1 coverage contract does not describe MTP's target
            # verification graph. Preserve ordinary capture and all dispatch.
            result = old_capture(self, create_forward_fn, *args, **kwargs)
            from vllm.distributed import get_tp_group
            rank = get_tp_group().rank_in_group
            record = dict(rank=rank, decode_query_len=self.decode_query_len,
                          manager=type(self).__name__,
                          graph_descriptors=[str(d) for d in self.graphs],
                          dispatch_unchanged=True, no_mtp_coverage_guard_skipped=True)
            Path(f'/evidence/hc-mtp-graphs-rank{rank}-{type(self).__name__}-{self.decode_query_len}.json').write_text(json.dumps(record, indent=2))
            print('hc_MTP_GRAPHS_READY', rank, self.decode_query_len, flush=True)
            return result
        if torch.cuda.get_device_capability(self.device) != (8, 0):
            return old_capture(self, create_forward_fn, *args, **kwargs)
        from vllm.config.compilation import CUDAGraphMode
        _captured.clear()
        def observed_create(desc, warmup):
            forward = create_forward_fn(desc, warmup=warmup)
            if warmup or desc.cg_mode != CUDAGraphMode.FULL or desc.num_tokens != 1:
                return forward
            def observed_forward(*forward_args, **forward_kwargs):
                global _record_capture
                # Piecewise graphs may be captured inside a warmup forward.
                # Only count the actual FULL M=1 capture, not every capture
                # performed anywhere during the manager's startup routine.
                _record_capture = torch.cuda.is_current_stream_capturing() and not _captured
                try:
                    return forward(*forward_args, **forward_kwargs)
                finally:
                    _record_capture = False
            return observed_forward
        result = old_capture(self, observed_create, *args, **kwargs)
        # Startup-only coverage guard for the pinned Qwen3.8 TP2 configuration.
        assert len(_captured) == 194, ('HC capture coverage changed', len(_captured))
        shapes = [tuple(row['weight_shape']) for row in _captured]
        assert shapes.count((10240, 336)) == 96
        assert shapes.count((10240, 320)) == 1
        assert shapes.count((320, 10240)) == 97
        from vllm.distributed import get_tp_group
        rank = get_tp_group().rank_in_group
        record = dict(enabled=True, rank=rank, projections=_captured,
                      sources=sources, graph_descriptors=[str(d) for d in self.graphs],
                      graph_replay='unmodified vLLM', request_hooks=False)
        evidence = Path(os.environ.get('Q38_HC_STARTUP_DIR', '/evidence'))
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f'hc-startup-rank{rank}.json').write_text(json.dumps(record, indent=2))
        print('HC_GEMV_CLEAN_CAPTURE_OK', rank, len(_captured), flush=True)
        _captured.clear()
        return result

    cc._reload_python_module = reload
    ct._reload_python_module = reload
    CudaGraphManager.capture = capture
    _installed = True
