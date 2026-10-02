"""M7 HC fusion and split-storage GDN projection; retain original M1 routes."""
import ast
import torch
import triton
import triton.language as tl
from m7_joint_kernels import mm7, finish7, up_gate7

_seen=set()
_weights={}
def mark(kind, weight):
    _weights.setdefault(kind,set()).add(weight.data_ptr())
    if kind not in _seen:
        _seen.add(kind)
        print(f'M7_JOINT_ACTIVE device={torch.cuda.current_device()} kind={kind}',flush=True)

def valid(x,wt,out):
    return (x.ndim==wt.ndim==out.ndim==2 and x.shape==(7,wt.shape[0])
            and out.shape==(7,wt.shape[1]) and wt.stride()==(1,wt.shape[0])
            and all(t.is_cuda and t.device==x.device and t.dtype==torch.bfloat16 for t in (x,wt,out))
            and x.stride(1)==out.stride(1)==1)

def down_silu(x,wt,*,out):
    if not valid(x,wt,out):
        import hc_gemv_runtime as hc
        hc.mm(x,wt,out=out)
        return torch.ops.vllm.qwen3_8_flash_next_hc_silu.default(out[:,:320],4)
    n=wt.shape[1];assert wt.shape[0]==10240 and n in (320,336)
    low=torch.empty((7,320),dtype=x.dtype,device=x.device)
    partial=torch.empty((16,7,n),dtype=torch.float32,device=x.device)
    mm7[(triton.cdiv(n,32),16)](x,wt.T,out,partial,low,n,10240,x.stride(0),out.stride(0),32,128,16,True,num_warps=4,num_stages=3)
    finish7[(triton.cdiv(7*n,256),)](partial,out,low,n,out.stride(0),16,True,256)
    mark('hc_down_silu',wt)
    return low

def up_gate(low,wt,out,normalized):
    if not valid(low,wt,out):
        import hc_gemv_runtime as hc
        hc.mm(low,wt,out=out)
        return torch.ops.vllm.qwen3_8_flash_next_hc_gate_mix.default(normalized,out,4)
    assert wt.shape==(320,10240)
    result=torch.empty((7,2560),dtype=low.dtype,device=low.device)
    up_gate7[(40,)](low,wt.T,normalized,result,low.stride(0),normalized.stride(0),2560,64,64,num_warps=4,num_stages=3)
    mark('hc_up_gate',wt)
    return result

@triton.jit
def projection7(X,Q,A,Y,Z,SX:tl.constexpr,SY:tl.constexpr,SZ:tl.constexpr):
    n=tl.program_id(0)*64+tl.arange(0,64)
    m=tl.arange(0,16);kk=tl.arange(0,128)
    acc=tl.zeros((16,64),tl.float32)
    for j in range(20):
        k=j*128+kk
        x=tl.load(X+m[:,None]*SX+k[None,:],m[:,None]<7,other=0)
        q=tl.load(Q+n[None,:]*2560+k[:,None],n[None,:]<8192,other=0)
        a=tl.load(A+(n[None,:]-8192)*2560+k[:,None],(n[None,:]>=8192)&(n[None,:]<8240),other=0)
        w=tl.where(n[None,:]<8192,q,a)
        acc=tl.dot(x,w,acc)
    tl.store(Y+m[:,None]*SY+n[None,:],acc.to(tl.bfloat16),(m[:,None]<7)&(n[None,:]<8192))
    tl.store(Z+m[:,None]*SZ+n[None,:]-8192,acc.to(tl.bfloat16),(m[:,None]<7)&(n[None,:]>=8192)&(n[None,:]<8240))

def project(x,q,a,y,z):
    if not (valid(x,q,y) and valid(x,a,z) and q.shape==(2560,8192) and a.shape==(2560,48)):
        return False
    projection7[(129,)](x,q.T,a.T,y,z,x.stride(0),y.stride(0),z.stride(0),num_warps=4,num_stages=3)
    mark('qkvz_ba_projection',q)
    return True

def instrument(source):
    """Fuse only recognized generated HC chains; maintain injection output."""
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[]
    for fn in ast.walk(tree):
        if not isinstance(fn,ast.FunctionDef) or fn.name!='call':continue
        nodes=sorted(ast.walk(fn),key=lambda n:getattr(n,'lineno',0))
        calls=[n for n in nodes if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='_hc_gemv.mm']
        for node in calls:
            c=node.value;shape=ast.literal_eval(c.args[1].args[1]);out=next(k.value for k in c.keywords if k.arg=='out');name=ast.unparse(out)
            if shape[0]==10240:
                matches=[n for n in nodes if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.ops.vllm.qwen3_8_flash_next_hc_silu.default' and name in {x.id for x in ast.walk(n.value.args[0]) if isinstance(x,ast.Name)}]
                assert len(matches)==1
                later=matches[0];assert ast.literal_eval(later.value.args[1])==4
                temp='_m7_low_'+name
                edits.append((node.lineno-1,f'{temp} = _m7_joint.down_silu({ast.unparse(c.args[0])}, {ast.unparse(c.args[1])}, out={name})'))
                edits.append((later.lineno-1,f'{ast.unparse(later.targets[0])} = {temp}; del {temp}'))
            elif shape==(320,10240):
                matches=[n for n in nodes if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.ops.vllm.qwen3_8_flash_next_hc_gate_mix.default' and ast.unparse(n.value.args[1])==name]
                assert len(matches)==1
                later=matches[0];assert ast.literal_eval(later.value.args[2])==4
                temp='_m7_up_'+name
                edits.append((node.lineno-1,f'{temp} = ({ast.unparse(c.args[0])}, {ast.unparse(c.args[1])}, {name})'))
                edits.append((later.lineno-1,f'{ast.unparse(later.targets[0])} = _m7_joint.up_gate(*{temp}, {ast.unparse(later.value.args[0])}); del {temp}'))
    for i,body in sorted(edits,reverse=True):
        indent=lines[i][:len(lines[i])-len(lines[i].lstrip())];lines[i]=indent+body+'\n'
    return ('import m7_joint_runtime as _m7_joint\n'+''.join(lines)) if edits else source
