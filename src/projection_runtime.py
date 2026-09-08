"""Pinned TP2/SM80 M=1 input projection; standard vLLM graphs, no request hooks."""
import ast
import functools
import hashlib
import json
from pathlib import Path
import torch
from projection_kernels import split_gemv

_installed = False
_record_capture = False
_captured = []


def supported(x, q, a, y, z):
    tensors = (x, q, a, y, z)
    return (tuple(x.shape) == (1, 2560) and tuple(q.shape) == (2560, 8192)
            and tuple(a.shape) == (2560, 48) and tuple(y.shape) == (1, 8192)
            and tuple(z.shape) == (1, 48) and q.stride() == a.stride() == (1, 2560)
            and all(t.is_cuda and t.device == x.device and t.dtype == torch.bfloat16 for t in tensors)
            and all(t.stride(-1) == 1 for t in (x, y, z)))


class Runtime:
    def __init__(self):
        self.kernel = None

    def proj(self, x, q, a, y, z, *, site):
        if not supported(x, q, a, y, z) or torch.cuda.get_device_capability(x.device) != (8, 0):
            torch.mm(x, q, out=y)
            torch.mm(x, a, out=z)
            return
        if self.kernel is None:
            # First warmup forward compiles this kernel before FULL capture.
            assert not torch.cuda.is_current_stream_capturing(), 'Projection requires warmup'
            self.kernel = split_gemv()
        self.kernel(x, q.T, a.T, y, z)
        if _record_capture and torch.cuda.is_current_stream_capturing():
            _captured.append(dict(site=site, q_ptr=q.data_ptr(), a_ptr=a.data_ptr()))


r = Runtime()
def instrument_source(source,key):
    if 'qwen3_8_flash_next_hc_gate_mix.default' not in source:return source,0
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[];count=0
    for fn in ast.walk(tree):
        if not isinstance(fn,ast.FunctionDef) or fn.name!='call':continue
        candidates=[]
        for node in ast.walk(fn):
            if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
            c=node.value
            if ast.unparse(c.func)!='extern_kernels.mm' or len(c.args)!=2:continue
            wt=c.args[1]
            if not isinstance(wt,ast.Call) or ast.unparse(wt.func)!='reinterpret_tensor' or len(wt.args)!=4:continue
            try:shape,stride,offset=[ast.literal_eval(v) for v in wt.args[1:]]
            except (ValueError,TypeError):continue
            if shape not in [(2560,8192),(2560,48)] or stride!=(1,2560) or offset!=0:continue
            assert len(c.keywords)==1 and c.keywords[0].arg=='out'
            candidates.append((node,c,shape,c.keywords[0].value))
        candidates.sort(key=lambda t:t[0].lineno)
        if not candidates:continue
        assert len(candidates)==2,('unexpected projection count',len(candidates))
        (nq,cq,sq,y),(na,ca,sa,z)=candidates
        assert sq==(2560,8192) and sa==(2560,48) and ast.unparse(cq.args[0])==ast.unparse(ca.args[0])
        assert all(n.lineno==n.end_lineno for n in (nq,na))
        # Moving QKVZ calculation until BA's output is allocated is valid only
        # if neither its input nor output is consumed/overwritten in between.
        between=ast.parse('def f():\n'+''.join('    '+line.lstrip() for line in lines[nq.end_lineno:na.lineno-1]))
        protected={ast.unparse(cq.args[0]),ast.unparse(y)}
        assert not any(isinstance(n,ast.Name) and n.id in protected for n in ast.walk(between))
        tmp='_projection_q_weight'
        assert tmp not in source
        edits.append((nq.lineno-1,f'{tmp} = {ast.unparse(cq.args[1])}'))
        expr=f'_projection.r.proj({ast.unparse(ca.args[0])}, {tmp}, {ast.unparse(ca.args[1])}, {ast.unparse(y)}, {ast.unparse(z)}, site={key!r})'
        edits.append((na.lineno-1,expr+'\n'+(' '*(len(lines[na.lineno-1])-len(lines[na.lineno-1].lstrip())))+f'del {tmp}'))
        count+=1
    for i,body in sorted(edits,reverse=True):
        indent=lines[i][:len(lines[i])-len(lines[i].lstrip())];lines[i]=indent+body+'\n'
    return ('import projection_runtime as _projection\n'+''.join(lines),count) if count else (source,0)


def install(_runner_cls=None):
    global _installed
    if _installed:
        return
    import torch._inductor.codecache as cc
    import torch._inductor.runtime.compile_tasks as ct
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    old_reload = cc._reload_python_module
    changed, sources = set(), []

    def reload(key, path, *args, **kwargs):
        if path not in changed:
            original = Path(path).read_text()
            if 'import projection_runtime as _projection' not in original:
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
            Path(f'/evidence/projection-mtp-graphs-rank{rank}-{type(self).__name__}-{self.decode_query_len}.json').write_text(json.dumps(record, indent=2))
            print('projection_MTP_GRAPHS_READY', rank, self.decode_query_len, flush=True)
            return result
        from vllm.config.compilation import CUDAGraphMode
        from vllm.distributed import get_tp_group
        assert get_tp_group().world_size == 2
        assert torch.cuda.get_device_capability(self.device) == (8, 0)
        _captured.clear()

        def observed_create(desc, warmup):
            forward = create_forward_fn(desc, warmup=warmup)
            if warmup or desc.cg_mode != CUDAGraphMode.FULL or desc.num_tokens != 1:
                return forward

            def observed_forward(*forward_args, **forward_kwargs):
                global _record_capture
                _record_capture = torch.cuda.is_current_stream_capturing() and not _captured
                try:
                    return forward(*forward_args, **forward_kwargs)
                finally:
                    _record_capture = False
            return observed_forward

        result = old_capture(self, observed_create, *args, **kwargs)
        assert len(_captured) == 36, ('Projection capture coverage changed', len(_captured))
        assert len({(p['q_ptr'], p['a_ptr']) for p in _captured}) == 36
        rank = get_tp_group().rank_in_group
        record = dict(rank=rank, count=len(_captured), projections=_captured, sources=sources,
                      extra_weight_bytes=0, request_hooks=False, alternative_graphs=False,
                      graph_replay='unmodified vLLM')
        Path(f'/evidence/projection-startup-rank{rank}.json').write_text(json.dumps(record, indent=2))
        print('PROJECTION_CLEAN_CAPTURE_OK', rank, len(_captured), flush=True)
        _captured.clear()
        return result

    cc._reload_python_module = reload
    ct._reload_python_module = reload
    CudaGraphManager.capture = capture
    _installed = True
