"""Set EAGER before spawned GPU interpreters start; restore parent environment."""
from contextlib import contextmanager
import os
import threading

_lock=threading.RLock()

@contextmanager
def gpu_worker_loading(start_method):
    if start_method!='spawn':
        raise RuntimeError('Scoped CUDA loading requires spawn, not inherited CUDA state')
    with _lock:
        before=os.environ.get('CUDA_MODULE_LOADING')
        os.environ['CUDA_MODULE_LOADING']='EAGER'
        try:
            yield
        finally:
            if before is None:os.environ.pop('CUDA_MODULE_LOADING',None)
            else:os.environ['CUDA_MODULE_LOADING']=before
