"""Real two-rank transport plus pinned upstream greedy sampler equivalence."""
import os,json,copy,time,statistics
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch.distributed as dist
import vllm.distributed as vd
import q38_compact_vocab as cv
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl');device=torch.device('cuda',rank)
def gather(x,dim=-1):
    parts=[torch.empty_like(x) for _ in range(2)];dist.all_gather(parts,x)
    return torch.cat(parts,dim=dim)
vd.tensor_model_parallel_all_gather=gather

def setup(n):
    ix=np.array([9,3,1,6,0,2,8,7,4,5][:n])
    states=NS(temperature=NS(np=np.zeros(10)),num_logprobs=np.full(10,-1),top_k=NS(np=np.full(10,248320)),min_p=NS(np=np.zeros(10)),top_p=NS(np=np.ones(10)))
    sampler=NS(sampling_states=states,compute_nans=False,return_sampling_mask=False,needs_logits_processing=np.zeros(10,dtype=bool),logprob_token_ids_state=NS(max_num_token_ids=lambda ix:0))
    runner=NS(sampler=sampler,rejection_sampler=NS(synthetic_conditional_rates=None,use_block_verification=False))
    batch=NS(num_reqs=n,num_tokens=n*7,num_draft_tokens=n*6,idx_mapping_np=ix,has_structured_output_reqs=False)
    return runner,batch
processor=NS(org_vocab_size=248320,scale=1,soft_cap=None)
head=NS(tp_size=2,num_embeddings=248320,shard_indices=NS(org_vocab_start_index=rank*124160,num_org_vocab_padding=0))

r,b=setup(10)
assert cv.eligible_greedy(r,b,None)
for key,val in [('compute_nans',True),('return_sampling_mask',True)]:
    setattr(r.sampler,key,val);assert not cv.eligible_greedy(r,b,None);setattr(r.sampler,key,False)
for array,val in [(r.sampler.sampling_states.temperature.np,1),(r.sampler.sampling_states.num_logprobs,0),(r.sampler.needs_logits_processing,True)]:
    old=array[3];array[3]=val;assert not cv.eligible_greedy(r,b,None);array[3]=old
assert not cv.eligible_greedy(r,b,object())
b.has_structured_output_reqs=True;assert not cv.eligible_greedy(r,b,None);b.has_structured_output_reqs=False
for key,val in [('synthetic_conditional_rates',object()),('use_block_verification',True)]:
    old=getattr(r.rejection_sampler,key);setattr(r.rejection_sampler,key,val);assert not cv.eligible_greedy(r,b,None);setattr(r.rejection_sampler,key,old)
r.sampler.logprob_token_ids_state.max_num_token_ids=lambda ix:1
assert not cv.eligible_greedy(r,b,None)
r.sampler.logprob_token_ids_state.max_num_token_ids=lambda ix:0
try:
    with cv.target_sampling(r,b,None):
        assert cv._CURRENT.get().greedy
        with cv.target_sampling(r,b,object()):assert cv._CURRENT.get() is None
        assert cv._CURRENT.get().greedy
        raise ValueError('intentional unwind')
except ValueError:pass
assert cv._CURRENT.get() is None

counts=dict(cases=0,rows=0,sampled_tokens=0,fallbacks=0)
for n in (1,2,3,8,10,3,1):
 for variant in ('random','ties','variable'):
    r,b=setup(n)
    lengths=[(i%7)+1 if variant=='variable' else 7 for i in range(n)]
    cu=np.array([0]+list(np.cumsum(lengths)),dtype=np.int32);rows=int(cu[-1])
    torch.manual_seed(123+rank+n);local=torch.randn(rows,124160,device=device,dtype=torch.bfloat16)
    if variant=='ties':
        for i in range(rows):
            for j in (0,8191,8192,124159):local[i,j]=20
    ref=gather(local)
    with cv.target_sampling(r,b,None):
        out=cv.try_logits(processor,head,local)
        assert out is not None and cv.already_filtered(out[:])
        assert not cv.already_filtered(out.clone())
        assert torch.equal(ref.argmax(-1),out.argmax(-1))
    idx=torch.tensor(b.idx_mapping_np,device=device,dtype=torch.int32)
    expanded=idx.repeat_interleave(torch.tensor(lengths,device=device))
    pos=torch.arange(rows,device=device,dtype=torch.int64)+11000
    localpos=torch.cat([torch.arange(k,device=device,dtype=torch.int32) for k in lengths])
    temps=torch.zeros(10,device=device);seeds=torch.arange(10,device=device,dtype=torch.int64)
    a=gumbel_sample(ref,expanded,temps,seeds,pos,False)
    z=gumbel_sample(out,expanded,temps,seeds,pos,False)
    assert torch.equal(a,z)
    draft=torch.zeros(rows,device=device,dtype=torch.int64)
    for req,(lo,hi) in enumerate(zip(cu[:-1],cu[1:])):
        draft[lo+1:hi]=a[lo:hi-1]
        if hi-lo>1 and req%3==1:draft[hi-1]=-1
        if hi-lo>1 and req%3==2:draft[lo+1]=(draft[lo+1]+1)%248320
    args=(None,draft,torch.tensor(cu,device=device),pos,idx,expanded,localpos,temps,seeds,6)
    ta,na=rejection_sample(ref,*args);tb,nb=rejection_sample(out,*args)
    assert torch.equal(na,nb)
    mask=torch.arange(7,device=device)[None,:]<na[:,None]
    assert torch.equal(ta[mask],tb[mask])
    counts['sampled_tokens']+=int(na.sum());counts['rows']+=rows;counts['cases']+=1
# Infinite logits have defined greedy winners, including all-masked shards.
r,b=setup(10)
for kind in ('positive_inf','one_masked','both_masked'):
    local=torch.randn(70,124160,device=device,dtype=torch.bfloat16)
    if kind=='both_masked' or (kind=='one_masked' and rank==1):local[0].fill_(-float('inf'))
    if kind=='positive_inf':local[0,123]=float('inf')
    ref=gather(local)
    with cv.target_sampling(r,b,None):
        out=cv.try_logits(processor,head,local)
        assert out is not None and torch.equal(ref.argmax(-1),out.argmax(-1))
    counts['cases']+=1
# Inspect the device assertion predicate without poisoning the test context.
local=torch.randn(70,124160,device=device,dtype=torch.bfloat16)
if rank==1:local[0,123]=float('nan')
saved_assert=torch._assert_async
def expected_nan(predicate,message):
    assert not bool(predicate.item()) and 'NaN target logits' in message
    raise ValueError('expected NaN rejection')
torch._assert_async=expected_nan
try:
    with cv.target_sampling(r,b,None):cv.try_logits(processor,head,local)
    raise AssertionError('NaN did not trigger rejection')
except ValueError as exc:assert str(exc)=='expected NaN rejection'
finally:torch._assert_async=saved_assert
counts['nan_rejection_predicate']=True
# A pinned-shape mismatch falls back before communication.
with cv.target_sampling(r,b,None):
    assert cv.try_logits(processor,head,torch.empty(71,124160,device=device,dtype=torch.bfloat16)) is None
    assert cv.try_logits(NS(org_vocab_size=248320,scale=.5,soft_cap=None),head,local) is None
# Micro latency for exactly the observed target batch; same local logits.
local=torch.randn(70,124160,device=device,dtype=torch.bfloat16)
times={}
for mode in ('full','compact'):
    values=[]
    for i in range(25):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        if mode=='full':result=gather(local)
        else:
            with cv.target_sampling(r,b,None):result=cv.try_logits(processor,head,local)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        if i>=5:values.append(elapsed)
    times[mode]=dict(mean_ms=statistics.mean(values),p50_ms=statistics.median(values))
print(json.dumps(dict(rank=rank,status='PASS',**counts,timing=times)),flush=True)
dist.destroy_process_group()
