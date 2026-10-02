"""SM80 M7 BF16 HC kernels; FP32 accumulation and BF16 stage boundaries."""
import triton
import triton.language as tl

@triton.jit
def mm7(X,W,Y,P,Low,N:tl.constexpr,K:tl.constexpr,SX:tl.constexpr,SY:tl.constexpr,
        BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr,FUSE:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN)
 m=tl.arange(0,16)
 kk=tl.arange(0,BK)
 split=tl.program_id(1)
 acc=tl.zeros((16,BN),tl.float32)
 for j in range(tl.cdiv(K,BK*SK)):
  k=(j*SK+split)*BK+kk
  a=tl.load(X+m[:,None]*SX+k[None,:],(m[:,None]<7)&(k[None,:]<K),other=0)
  b=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),other=0)
  acc=tl.dot(a,b,acc)
 if SK==1:
  y=acc.to(tl.bfloat16)
  tl.store(Y+m[:,None]*SY+n[None,:],y,(m[:,None]<7)&(n[None,:]<N))
  if FUSE:
   z=y.to(tl.float32)/4
   tl.store(Low+m[:,None]*320+n[None,:],z*tl.sigmoid(z),(m[:,None]<7)&(n[None,:]<320))
 else:
  tl.store(P+split*7*N+m[:,None]*N+n[None,:],acc,(m[:,None]<7)&(n[None,:]<N))

@triton.jit
def finish7(P,Y,Low,N:tl.constexpr,SY:tl.constexpr,SK:tl.constexpr,FUSE:tl.constexpr,B:tl.constexpr):
 idx=tl.program_id(0)*B+tl.arange(0,B);s=tl.arange(0,SK)
 vals=tl.load(P+s[:,None]*7*N+idx[None,:],idx[None,:]<7*N,other=0)
 y=tl.sum(vals,axis=0).to(tl.bfloat16);m=idx//N;n=idx%N
 tl.store(Y+m*SY+n,y,idx<7*N)
 if FUSE:
  z=y.to(tl.float32)/4
  tl.store(Low+m*320+n,z*tl.sigmoid(z),(m<7)&(n<320))

@triton.jit
def up_gate7(Low,W,X,Out,SLOW:tl.constexpr,SX:tl.constexpr,SO:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN);m=tl.arange(0,16);kk=tl.arange(0,BK)
 a0=tl.zeros((16,BN),tl.float32);a1=tl.zeros((16,BN),tl.float32)
 a2=tl.zeros((16,BN),tl.float32);a3=tl.zeros((16,BN),tl.float32)
 for j in range(tl.cdiv(320,BK)):
  k=j*BK+kk
  a=tl.load(Low+m[:,None]*SLOW+k[None,:],(m[:,None]<7)&(k[None,:]<320),other=0)
  b0=tl.load(W+n[None,:]*320+k[:,None],(n[None,:]<2560)&(k[:,None]<320),other=0)
  b1=tl.load(W+(n[None,:]+2560)*320+k[:,None],(n[None,:]<2560)&(k[:,None]<320),other=0)
  b2=tl.load(W+(n[None,:]+5120)*320+k[:,None],(n[None,:]<2560)&(k[:,None]<320),other=0)
  b3=tl.load(W+(n[None,:]+7680)*320+k[:,None],(n[None,:]<2560)&(k[:,None]<320),other=0)
  a0=tl.dot(a,b0,a0);a1=tl.dot(a,b1,a1);a2=tl.dot(a,b2,a2);a3=tl.dot(a,b3,a3)
 g0=a0.to(tl.bfloat16).to(tl.float32);g1=a1.to(tl.bfloat16).to(tl.float32)
 g2=a2.to(tl.bfloat16).to(tl.float32);g3=a3.to(tl.bfloat16).to(tl.float32)
 mask=(m[:,None]<7)&(n[None,:]<2560)
 x0=tl.load(X+m[:,None]*SX+n[None,:],mask,other=0).to(tl.float32)
 x1=tl.load(X+m[:,None]*SX+n[None,:]+2560,mask,other=0).to(tl.float32)
 x2=tl.load(X+m[:,None]*SX+n[None,:]+5120,mask,other=0).to(tl.float32)
 x3=tl.load(X+m[:,None]*SX+n[None,:]+7680,mask,other=0).to(tl.float32)
 mix=tl.sigmoid(g0)*x0
 mix+=tl.sigmoid(g1)*x1
 mix+=tl.sigmoid(g2)*x2
 mix+=tl.sigmoid(g3)*x3
 tl.store(Out+m[:,None]*SO+n[None,:],mix/4,mask)

@triton.jit
def simt7(X,W,Y,N:tl.constexpr,K:tl.constexpr,SX:tl.constexpr,SY:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.arange(0,8);n=tl.program_id(0)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
 acc=tl.zeros((8,BN,BK),tl.float32)
 for j in range(tl.cdiv(K,BK)):
  k=j*BK+kk
  x=tl.load(X+m[:,None]*SX+k[None,:],(m[:,None]<7)&(k[None,:]<K),other=0).to(tl.float32)
  w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),other=0).to(tl.float32)
  acc+=x[:,None,:]*w[None,:,:]
 y=tl.sum(acc,axis=2)
 tl.store(Y+m[:,None]*SY+n[None,:],y,(m[:,None]<7)&(n[None,:]<N))
