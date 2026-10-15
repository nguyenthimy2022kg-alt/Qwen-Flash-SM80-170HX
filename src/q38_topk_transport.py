"""Lossless local top-k transport, including arbitrarily many boundary ties.

Each shard sends its k largest scores/IDs and a bit per vocabulary entry
equal to its kth score. Everything discarded is below the *local* kth score,
so it cannot survive the global top-k filter. No host read or conditional
collective is required. The ordinary sampler still performs global top-k,
top-p, and speculative verification using its original reduction layout.
"""
import torch
import triton as tr
import triton.language as tl

K = 20


@tr.jit
def _pack_ties(X, V, I, P, WIDTH: tl.constexpr, XSTRIDE: tl.constexpr,
               WORDS: tl.constexpr, PACK: tl.constexpr, B: tl.constexpr, K: tl.constexpr = 20):
    row, block = tl.program_id(0), tl.program_id(1)
    word = block * B + tl.arange(0, B)
    bit = tl.arange(0, 32)
    col = word[:, None] * 32 + bit[None, :]
    threshold = tl.load(V + row * K + K - 1).to(tl.float32)
    x = tl.load(X + row * XSTRIDE + col, col < WIDTH, other=-float('inf')).to(tl.float32)
    tied = ((x == threshold) | ((x != x) & (threshold != threshold))) & (col < WIDTH)
    bits = tl.sum(tied.to(tl.uint32) << bit[None, :], 1)
    tl.store(P + row * PACK + 2 * K + 1 + word, bits, word < WORDS)
    if block == 0:
        j = tl.arange(0, 32)
        score = tl.load(V + row * K + j, j < K, other=0).to(tl.float32)
        index = tl.load(I + row * K + j, j < K, other=0).to(tl.int32)
        tl.store(P + row * PACK + j, score.to(tl.int32, bitcast=True), j < K)
        tl.store(P + row * PACK + K + j, index, j < K)
        tl.store(P + row * PACK + 2 * K, threshold.to(tl.int32, bitcast=True))


@tr.jit
def _unpack_ties(P, Y, WIDTH: tl.constexpr, PACK: tl.constexpr, B: tl.constexpr, K: tl.constexpr = 20):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * B + tl.arange(0, B)
    shard, local = col // WIDTH, col % WIDTH
    ptr = P + row * (2 * PACK) + shard * PACK
    threshold = tl.load(ptr + 2 * K, col < 2 * WIDTH, other=0).to(tl.float32, bitcast=True)
    bits = tl.load(ptr + 2 * K + 1 + local // 32, col < 2 * WIDTH, other=0).to(tl.uint32)
    tied = ((bits >> (local % 32)) & 1) != 0
    tl.store(Y + row * (2 * WIDTH) + col,
             tl.where(tied, threshold, -float('inf')), col < 2 * WIDTH)


@tr.jit
def _unpack_top(P, Y, WIDTH: tl.constexpr, PACK: tl.constexpr, K: tl.constexpr = 20):
    row, shard = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, 32)
    ptr = P + row * (2 * PACK) + shard * PACK
    value = tl.load(ptr + j, j < K, other=0).to(tl.float32, bitcast=True)
    index = tl.load(ptr + K + j, j < K, other=0)
    tl.store(Y + row * (2 * WIDTH) + shard * WIDTH + index, value, j < K)


def pack_topk(local):
    rows, width = local.shape
    words = tr.cdiv(width, 32)
    size = 2 * K + 1 + words
    values, indices = local.topk(K, dim=-1, sorted=True)
    result = torch.empty((rows, size), dtype=torch.int32, device=local.device)
    _pack_ties[(rows, tr.cdiv(words, 32))](
        local, values, indices, result, width, local.stride(0), words, size, 32)
    return result


def unpack_topk(gathered, width, dtype):
    rows = gathered.shape[0]
    size = 2 * K + 1 + tr.cdiv(width, 32)
    result = torch.empty((rows, 2 * width), dtype=dtype, device=gathered.device)
    _unpack_ties[(rows, tr.cdiv(2 * width, 1024))](gathered, result, width, size, 1024)
    _unpack_top[(rows, 2)](gathered, result, width, size)
    return result
