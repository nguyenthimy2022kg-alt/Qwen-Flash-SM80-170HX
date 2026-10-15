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
from q38_compact_ops import CAP, pack, compact_filter, pack_greedy, greedy_filter

_ENABLED = os.getenv('Q38_COMPACT_VOCAB', '1') == '1'
_GREEDY_ENABLED = os.getenv('Q38_GREEDY_VOCAB', '1') == '1'
_TOPK_ENABLED = os.getenv('Q38_TOPK_VOCAB', '1') == '1'
_CURRENT = ContextVar('q38_compact_target_sampling', default=None)
_parameters = None  # One-entry device parameter cache, bounded across requests.

@dataclass
class SamplingContext:
    top_p: float | None
    output: object = None
    greedy: bool = False
    max_rows: int = 7
    topk_transport: bool = False

def eligible_greedy(runner, batch, grammar):
    if not _ENABLED or not _GREEDY_ENABLED or grammar is not None:
        return False
    if not 1 <= batch.num_reqs <= 10 or batch.has_structured_output_reqs:
        return False
    if not 0 <= batch.num_draft_tokens <= 6 * batch.num_reqs:
        return False
    verifier = runner.rejection_sampler
    if (verifier is None or verifier.synthetic_conditional_rates is not None
            or verifier.use_block_verification):
        return False
    sampler = runner.sampler
    if sampler.compute_nans or sampler.return_sampling_mask:
        return False
    ix = batch.idx_mapping_np
    states = sampler.sampling_states
    if (not np.all(states.temperature.np[ix] == 0)
            or np.any(sampler.needs_logits_processing[ix])
            or np.any(states.num_logprobs[ix] != -1)
            or sampler.logprob_token_ids_state.max_num_token_ids(ix) > 0):
        return False
    return True

def eligible_sampled(runner, batch, grammar):
    if not _ENABLED or not _TOPK_ENABLED or grammar is not None:
        return False
    if not 2 <= batch.num_reqs <= 10 or batch.has_structured_output_reqs:
        return False
    if not 0 <= batch.num_draft_tokens <= 6 * batch.num_reqs:
        return False
    verifier = runner.rejection_sampler
    if (verifier is None or verifier.synthetic_conditional_rates is not None
            or verifier.use_block_verification):
        return False
    sampler = runner.sampler
    if sampler.compute_nans or sampler.return_sampling_mask:
        return False
    ix = batch.idx_mapping_np
    states = sampler.sampling_states
    if (not np.all(states.temperature.np[ix] == 1)
            or not np.all(states.top_k.np[ix] == 20)
            or not np.all(states.min_p.np[ix] == 0)
            or not np.all((states.top_p.np[ix] > 0) & (states.top_p.np[ix] <= 1))
            or np.any(states.num_logprobs[ix] != -1)
            or sampler.logprob_token_ids_state.max_num_token_ids(ix) > 0
            or np.any(sampler.penalties_state.use_penalty[ix])
            or np.any(sampler.logit_bias_state.use_logit_bias[ix])
            or np.any(sampler.bad_words_state.num_bad_words.np[ix] > 0)):
        return False
    budget = sampler.thinking_budget_state
    if budget.enabled and np.any(budget.use_thinking_budget[ix]):
        return False
    return True

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
    if eligible_greedy(runner, batch, grammar):
        state = SamplingContext(None, greedy=True, max_rows=7 * batch.num_reqs)
    elif eligible_sampled(runner, batch, grammar):
        state = SamplingContext(None, max_rows=7 * batch.num_reqs, topk_transport=True)
    else:
        top_p = eligible_top_p(runner, batch, grammar)
        state = SamplingContext(top_p) if top_p is not None else None
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        _CURRENT.reset(token)

def already_filtered(logits):
    state = _CURRENT.get()
    if state is None or state.output is None:
        return False
    expected = state.output
    # The verifier passes a full tensor view; do not exempt a
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
    if local.ndim != 2 or local.dtype != torch.bfloat16:
        return None
    if processor.scale != 1 or processor.soft_cap is not None:
        return None
    from vllm.distributed import tensor_model_parallel_all_gather
    if state.greedy:
        shard = lm_head.shard_indices
        if (local.shape[1] != 124160 or not 1 <= local.shape[0] <= state.max_rows
                or shard.num_org_vocab_padding != 0
                or shard.org_vocab_start_index not in (0, 124160)
                or lm_head.num_embeddings != processor.org_vocab_size):
            return None
        gathered = tensor_model_parallel_all_gather(
            pack_greedy(local, shard.org_vocab_start_index), dim=-1)
        output = greedy_filter(gathered, processor.org_vocab_size, local.dtype)
        if output is not None:
            state.output = output
        return output
    if state.topk_transport:
        shard = lm_head.shard_indices
        if (local.shape[1] != 124160 or not 1 <= local.shape[0] <= state.max_rows
                or local.stride(1) != 1
                or shard.num_org_vocab_padding != 0
                or shard.org_vocab_start_index not in (0, 124160)
                or lm_head.num_embeddings != processor.org_vocab_size):
            return None
        from q38_topk_transport import pack_topk, unpack_topk
        gathered = tensor_model_parallel_all_gather(pack_topk(local), dim=-1)
        # Do not mark already_filtered: the ordinary sampler must still apply
        # global top-k and each request's top-p before verification.
        return unpack_topk(gathered, local.shape[1], local.dtype)
    if local.shape != (7, 124160):
        return None
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
