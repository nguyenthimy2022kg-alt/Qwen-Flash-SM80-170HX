"""Trial: explicitly resolve Marlin functions while the GPU is still idle."""
import ctypes as C
import hashlib
import itertools
import json
import time
from pathlib import Path

class Scalar(C.Structure):
    # vLLM csrc/core/scalar_type.hpp; only for the pinned native libraries.
    _fields_=[('exponent',C.c_uint8),('mantissa',C.c_uint8),('signed',C.c_bool),
              ('bias',C.c_int32),('finite',C.c_bool),('nan_repr',C.c_uint8)]

def scalar(s):
    return Scalar(s.exponent,s.mantissa,s.signed,s.bias,s._finite_values_only,s.nan_repr.value)

def initialize(rank=0, resolve=True):
    import torch
    import vllm
    from vllm.scalar_type import scalar_types as types
    assert C.sizeof(Scalar)==12 and Scalar.bias.offset==4 and Scalar.finite.offset==8
    root=Path(vllm.__file__).parent
    cudart=C.CDLL(str((Path(torch.__file__).parent/'../nvidia/cu13/lib/libcudart.so.13').resolve()))
    attr=cudart.cudaFuncGetAttributes
    attr.argtypes=[C.c_void_p,C.c_void_p];attr.restype=C.c_int
    bf16=scalar(types.bfloat16)
    expected={'_C_stable_libtorch.abi3.so':'01baa4f3c42e1742bee6fb78d260e3e9d13b1a0ab3a6515c7c5f56737952d566','_moe_C_stable_libtorch.abi3.so':'b34e1e3ce1b26088e717d1483a3801f1737cc68fd99263d91349e47b779d032a'}
    rows=[];start=time.monotonic()
    for library,namespace in [('_C_stable_libtorch.abi3.so','6marlin'),('_moe_C_stable_libtorch.abi3.so','16marlin_moe_wna16')]:
        p=root/library
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==expected[library], (library,'unsupported native binary')
        lib=C.CDLL(str(p))
        fn=getattr(lib,f'_ZN{namespace}17get_marlin_kernelEN4vllm10ScalarTypeES1_S1_S1_iiibbbiibi')
        fn.argtypes=[Scalar]*4+[C.c_int]*3+[C.c_bool]*3+[C.c_int]*2+[C.c_bool,C.c_int]
        fn.restype=C.c_void_p
        default=fn(bf16,bf16,bf16,bf16,0,0,0,False,False,False,0,0,False,0)
        seen=set();configs=[]
        for weight,scale,group in [(types.float4_e2m1f,types.float8_e4m3fn,1),(types.uint8b128,types.bfloat16,8)]:
            b=scalar(weight);s=scalar(scale)
            for m,n,k,threads,m8,stages in itertools.product(range(1,9),(2,4,8,16),(2,4,8,16),(128,256),(False,True),(2,3,4,5)):
                ptr=fn(bf16,b,bf16,s,m,n,k,m8,False,False,group,threads,False,stages)
                if ptr==default or ptr in seen:continue
                assert ptr
                if resolve:
                    buf=C.create_string_buffer(512)
                    status=attr(C.byref(buf),ptr)
                    if status:raise RuntimeError(f'cudaFuncGetAttributes failed: {library} status={status}')
                seen.add(ptr);configs.append([str(weight),m,n,k,threads,m8,stages])
        assert len(seen)==30, (library,'no matching kernels; ABI/config mismatch')
        rows.append({'library':library,'sha256':digest,'kernels':len(seen),'configs':configs})
    result={'rank':rank,'seconds':time.monotonic()-start,'resolved':resolve,'libraries':rows}
    Path(f'/evidence/marlin-preload-rank{rank}.json').write_text(json.dumps(result,indent=2))
    print('MARLIN_PRELOAD_READY',rank,[(r['library'],r['kernels']) for r in rows],flush=True)
    return result

if __name__=='__main__':
    import torch
    torch.cuda.set_device(0)
    initialize()
