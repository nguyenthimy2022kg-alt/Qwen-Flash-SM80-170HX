"""Restricted, lossless target-logit transport for the pinned SM80 runtime.

Unsupported sampling configurations use the original implementation. State is
scoped to a single target sampling call, including exceptional exits.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import numpy as np
import torch
from q38_compact_ops import CAP, pack, compact_filter

_ENABLED = os.getenv('Q38_COMPACT_VOCAB', '1') == '1'
_CURRENT = ContextVar('q38_compact_target_sampling', default=None)
_parameters = None  # One-entry device parameter cache, bounded across requests.

@dataclass
class SamplingContext:
    top_p: float
    output: object = None

def eligible_top_p(runner, batch, grammar):
    if not _ENABLED or grammar is not None:
        return None
    if batch.num_reqs != 1 or batch.num_tokens != 7 or batch.num_draft_tokens != 6:
        return None
    if runner.rejection_sampler is None:
        return None
    sampler = runner.sampler
    ix = batch.idx_mapping_np
    states = sampler.sampling_states
    if not np.all(states.temperature.np[ix] == 1):
        return None
    if not np.all(states.top_k.np[ix] == 20) or not np.all(states.min_p.np[ix] == 0):
        return None
    if np.any(states.num_logprobs[ix] != -1):
        return None
    if sampler.logprob_token_ids_state.max_num_token_ids(ix) > 0:
        return None
    if np.any(sampler.penalties_state.use_penalty[ix]):
        return None
    if np.any(sampler.logit_bias_state.use_logit_bias[ix]):
        return None
    if np.any(sampler.bad_words_state.num_bad_words.np[ix] > 0):
        return None
    budget = sampler.thinking_budget_state
    if budget.enabled and np.any(budget.use_thinking_budget[ix]):
        return None
    top_p = float(states.top_p.np[ix][0])
    return top_p if 0 < top_p <= 1 else None

@contextmanager
def target_sampling(runner, batch, grammar):
    top_p = eligible_top_p(runner, batch, grammar)
    token = _CURRENT.set(SamplingContext(top_p) if top_p is not None else None)
    try:
        yield
    finally:
        _CURRENT.reset(token)

def already_filtered(logits):
    state = _CURRENT.get()
    if state is None or state.output is None:
        return False
    expected = state.output
    # The single request verifier passes a full tensor view; do not exempt a
    # different tensor or a partial chunk from normal processing.
    return (logits.shape == expected.shape and logits.dtype == expected.dtype
            and logits.device == expected.device and logits.stride() == expected.stride()
            and logits.data_ptr() == expected.data_ptr())

def try_logits(processor, lm_head, local):
    global _parameters
    state = _CURRENT.get()
    if state is None:
        return None
    if lm_head.tp_size != 2 or processor.org_vocab_size != 248320:
        return None
    if local.shape != (7,124160) or local.dtype != torch.bfloat16:
        return None
    if processor.scale != 1 or processor.soft_cap is not None:
        return None
    from vllm.distributed import tensor_model_parallel_all_gather
    key = (state.top_p, local.device)
    if _parameters is None or _parameters[0] != key:
        _parameters = (key, torch.full((7,),20,device=local.device,dtype=torch.int32),
                       torch.full((7,),state.top_p,device=local.device))
    packed = pack(local, lm_head.shard_indices.org_vocab_start_index)
    gathered = tensor_model_parallel_all_gather(packed, dim=-1)
    output = compact_filter(list(gathered.split(2*CAP+1,dim=-1)),248320,
                            _parameters[1],_parameters[2])
    if output is not None:
        state.output = output
    return output
