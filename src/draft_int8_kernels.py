"""Full-vocabulary weight-only INT8 GEMV. BF16 inputs/output; FP32 accumulation."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def quantize(W,Q,S, N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)
    col=tl.arange(0,B)
    v=tl.load(W+group*G+col,col<G,other=0).to(tl.float32)
    a=tl.max(tl.abs(v),0)
    s=tl.where(a>0,a/127.,1.)
    q=tl.minimum(tl.maximum(libdevice.nearbyint(v/s),-127.),127.).to(tl.int8)
    tl.store(Q+group*G+col,q,col<G)
    tl.store(S+group,s)

@triton.jit
def gemv(X,Q,S,Y,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
         BN:tl.constexpr,BK:tl.constexpr):
    ns=tl.program_id(0)*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK)
    acc=tl.zeros((BN,BK),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+ks
        x=tl.load(X+kk,kk<K,other=0).to(tl.float32)
        q=tl.load(Q+ns[:,None]*K+kk[None,:],(ns[:,None]<N)&(kk[None,:]<K),other=0).to(tl.float32)
        s=tl.load(S+ns[:,None]*(K//G)+kk[None,:]//G,(ns[:,None]<N)&(kk[None,:]<K),other=0)
        acc=acc+(q*s)*x[None,:]
    tl.store(Y+ns,tl.sum(acc,1),ns<N)
