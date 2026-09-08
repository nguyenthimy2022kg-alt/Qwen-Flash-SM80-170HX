"""Bounded, unquantized SM80 HC GEMV candidates. Row-major weights."""
import triton
import triton.language as tl


@triton.jit
def row_gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + cols
        x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
        acc += w * x[None, :]
    y = tl.sum(acc, axis=1)
    tl.store(Y + rows, y, rows < N)
