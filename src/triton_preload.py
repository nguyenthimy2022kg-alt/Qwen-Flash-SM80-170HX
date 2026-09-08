"""Preload pinned, previously compiled kernels before any model stream waits.

The binary cache returns the original native load_binary ABI tuple. Handles stay
owned for process lifetime. No kernel is executed by this loader. Unknown later
kernels keep the native loader, with start/end records for bounded diagnosis.
"""
import functools
import hashlib
import json
import time
from pathlib import Path

_initialized=False
_handles={}

def initialize(device, rank=None):
    global _initialized
    if _initialized:return
    import torch
    import triton
    from triton.runtime import driver
    assert torch.cuda.current_device()==device.index
    assert torch.cuda.get_device_capability(device)==(8,0)
    if rank is None:
        from vllm.distributed import get_tp_group
        rank=get_tp_group().rank_in_group
    import ctypes,os
    mode=ctypes.c_int()
    assert ctypes.CDLL('libcuda.so.1').cuModuleGetLoadingMode(ctypes.byref(mode))==0
    assert mode.value==1, ('GPU worker must use EAGER',mode.value)
    utils=driver.active.utils
    original=utils.load_binary
    root=Path(__file__).parent/'preload'
    manifest=json.loads((root/'manifest.json').read_text())
    records=[]
    def load(name,binary,shared,ordinal):
        digest=hashlib.sha256(binary).hexdigest()
        key=(name,digest,shared,ordinal)
        if key in _handles:return _handles[key]
        path=Path(f'/evidence/triton-load-rank{rank}.jsonl')
        def record(stage):
            with path.open('a') as f:f.write(json.dumps(dict(stage=stage,name=name,sha256=digest,device=ordinal,time=time.time()))+'\n')
        record('start');result=original(name,binary,shared,ordinal);record('end')
        assert len(result)==5
        _handles[key]=result
        return result
    for row in manifest:
        binary=(root/row['file']).read_bytes()
        assert hashlib.sha256(binary).hexdigest()==row['sha256']
        handles=load(row['name'],binary,row['shared'],device.index)
        records.append(dict(**row,loaded=True))
    utils.load_binary=load
    _initialized=True
    Path(f'/evidence/triton-preload-rank{rank}.json').write_text(json.dumps(dict(
        count=len(records),records=records,device=device.index,phase='before_model_load',
        cuda_loading_mode=mode.value, cuda_loading_env=os.environ.get('CUDA_MODULE_LOADING'),
        triton=triton.__version__, torch=torch.__version__,extra_model_weights=False),indent=2))
    print('TRITON_PRELOAD_READY',rank,len(records),flush=True)

def install(cls):
    original=cls.load_model
    @functools.wraps(original)
    def load_model(self,*args,**kwargs):
        initialize(self.device)
        return original(self,*args,**kwargs)
    cls.load_model=load_model
