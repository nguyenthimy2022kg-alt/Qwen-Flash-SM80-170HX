"""Frozen SM80 single-token GDN replay. No per-step file I/O or diagnostics.

Enable at process startup with QWEN_GDN_REPLAY=1. Keep the matched model wrappers
and repaired PLE runner from this release. Unsupported inputs use the original
method; captured bindings are retained for the lifetime of the worker.
"""
import os
import time
from types import SimpleNamespace
import torch

_setting=os.environ.get('QWEN_GDN_REPLAY','0')
if _setting not in ('0','1'):
    raise ValueError('QWEN_GDN_REPLAY must be 0 or 1')
ENABLED=_setting=='1'
_allow=False
_arch=None
_layers={}

# Preserve the frozen experiment's startup JIT-monitor compatibility setting.
# This is unrelated to replay timing and is identical in enabled/disabled runs.
if os.getenv('GDN_DIAG_DISABLE_JIT_MONITOR')=='1':
    import vllm.utils.jit_monitor as _jm
    _jm.activate=lambda *args,**kwargs:None

def runner_tick(dummy_run=False):
    global _allow
    _allow=not dummy_run

class Entry:
    def __init__(self):
        self.calls=0;self.graph=None;self.key=None;self.refs=None
        self.indices=None;self.capture_stream=None;self.capture_ms=None


def signature(tensors,layer):
    return (layer.activation,layer.head_k_dim,tuple(
        None if t is None else (t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device)
        for t in tensors))

def try_run(layer,q,b,a,out,meta,eager):
    global _arch
    if not ENABLED or not _allow:return False
    name=layer.prefix
    entry=_layers.get(name)
    if entry is None:entry=_layers.setdefault(name,Entry())
    if entry.graph is None:entry.calls+=1
    def reject(reason):
        return False
    if (meta.num_actual_tokens!=1 or meta.num_decodes!=1 or meta.num_prefills!=0
            or getattr(meta,'spec_sequence_masks',None) is not None):return reject('not_single_decode')
    if torch.cuda.is_current_stream_capturing():return reject('outer_capture')
    if _arch is None:_arch=torch.cuda.get_device_capability(q.device)
    if _arch!=(8,0):return reject('architecture')
    idx=meta.non_spec_state_indices_tensor
    if idx is None or idx.ndim!=1 or idx.dtype!=torch.int32 or idx.stride(0)!=1:return reject('indices')
    if q.dtype!=torch.bfloat16 or b.dtype!=q.dtype or a.dtype!=q.dtype or out.dtype!=q.dtype:return reject('dtype')
    if q.shape!=(1,10240) or a.shape!=(1,48) or b.shape!=(1,48) or out.shape!=(1,48,128):return reject('shape')
    caches=layer.kv_cache
    if len(caches)!=2 or caches[0].dtype!=q.dtype or caches[1].dtype!=q.dtype:return reject('cache_dtype')
    tensors=(q,b,a,out,caches[0],caches[1],layer.conv1d.weight,layer.conv1d.bias,layer.A_log,layer.dt_bias)
    if any(t is not None and t.device!=q.device for t in tensors) or idx.device!=q.device:return reject('device')
    key=signature(tensors,layer)
    if entry.graph is not None and key!=entry.key:return reject('binding_changed')
    if entry.graph is None:
        # Original execution must already have compiled the exact kernel path.
        if entry.calls<8:return reject('warmup')
        entry.refs=tensors  # Keep original buffers alive for the entire graph lifetime.
        entry.key=key
        entry.indices=torch.empty((1,),dtype=torch.int32,device=q.device)
        entry.capture_stream=torch.cuda.Stream(device=q.device)
        graph=torch.cuda.CUDAGraph()
        capture_meta=SimpleNamespace(num_actual_tokens=1,non_spec_state_indices_tensor=entry.indices)
        t0=time.perf_counter_ns()
        # Low-level capture avoids torch.cuda.graph's device-wide synchronize,
        # GC and empty_cache. Capture records kernels; it does not execute them.
        # Launching on the original stream below preserves all producer waits.
        with torch.cuda.stream(entry.capture_stream):
            graph.capture_begin(capture_error_mode='thread_local')
            try:eager(q,b,a,out,capture_meta)
            finally:graph.capture_end()
        entry.graph=graph;entry.capture_ms=(time.perf_counter_ns()-t0)/1e6
        print(f'GDN_REPLAY_CAPTURED device={q.device.index} layer={name}',flush=True)
    entry.indices.copy_(idx[:1])
    entry.graph.replay()
    return True
