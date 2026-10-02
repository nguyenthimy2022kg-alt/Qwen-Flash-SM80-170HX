"""Single-request M7 adapter for a 33-column speculative state table."""
from vllm.triton_utils import triton, tl

@triton.jit
def pack(Full, Accepted, Small, NewAccepted):
    i = tl.arange(0, 8)
    accepted = tl.load(Accepted)
    valid = (accepted > 0) & (accepted <= 33)
    index = tl.where(i < 7, i, tl.maximum(0, tl.minimum(32, accepted - 1)))
    value = tl.load(Full + index)
    tl.store(Small + i, value)
    tl.store(NewAccepted, tl.where(valid, 8, 0))

def eligible(metadata, state_indices):
    return (state_indices.size(0) == 1
            and state_indices.size(1) == 33
            and metadata.num_spec_decodes == 1
            and metadata.num_actual_tokens == 7
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0)

def adapt(layer, state_indices, accepted):
    small = layer._wide_mtp_state_indices
    new_accepted = layer._wide_mtp_num_accepted
    pack[(1,)](state_indices, accepted, small, new_accepted)
    return small, new_accepted
