"""Same selected 4/256/128 GEMV, directly reading/writing two existing buffers."""
import tilelang
import tilelang.language as T

@tilelang.jit(target={'kind':'cuda','arch':'sm_80'})
def split_gemv():
    @T.prim_func
    def main(X:T.Tensor((1,2560),'bfloat16'), Q:T.Tensor((8192,2560),'bfloat16'),
             A:T.Tensor((48,2560),'bfloat16'), Y:T.Tensor((1,8192),'bfloat16'),
             Z:T.Tensor((1,48),'bfloat16')):
        with T.Kernel(2060,threads=128) as bn:
            accum=T.alloc_fragment((4,256),'float32')
            reduced=T.alloc_fragment((4,),'float32')
            T.clear(accum)
            if bn < 2048:
                for block in T.serial(10):
                    for i,j in T.Parallel(4,256):
                        accum[i,j]+=T.cast(Q[bn*4+i,block*256+j],'float32')*T.cast(X[0,block*256+j],'float32')
            else:
                for block in T.serial(10):
                    for i,j in T.Parallel(4,256):
                        accum[i,j]+=T.cast(A[(bn-2048)*4+i,block*256+j],'float32')*T.cast(X[0,block*256+j],'float32')
            T.reduce_sum(accum,reduced,dim=1)
            if bn < 2048:
                for i in T.Parallel(4):Y[0,bn*4+i]=reduced[i]
            else:
                for i in T.Parallel(4):Z[0,(bn-2048)*4+i]=reduced[i]
    return main
