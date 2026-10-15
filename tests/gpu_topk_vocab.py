"""Two-rank exact post-filter, probabilities, sampling and MTP checks."""
import os,json,time,statistics,copy
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch.distributed as dist
import vllm.distributed as vd
import q38_compact_vocab as cv
from q38_topk_transport import pack_topk,unpack_topk
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as reference
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');device=torch.device('cuda',rank)
def gather(x,dim=-1):
    parts=[torch.empty_like(x) for _ in range(2)];dist.all_gather(parts,x);return torch.cat(parts,dim=dim)
vd.tensor_model_parallel_all_gather=gather

def setup(n):
    ix=np.array([9,3,1,6,0,2,8,7,4,5][:n])
    states=NS(temperature=NS(np=np.ones(10)),num_logprobs=np.full(10,-1),top_k=NS(np=np.full(10,20)),min_p=NS(np=np.zeros(10)),top_p=NS(np=np.full(10,.95)))
    sampler=NS(sampling_states=states,compute_nans=False,return_sampling_mask=False,needs_logits_processing=np.ones(10,dtype=bool),logprob_token_ids_state=NS(max_num_token_ids=lambda ix:0),penalties_state=NS(use_penalty=np.zeros(10,dtype=bool)),logit_bias_state=NS(use_logit_bias=np.zeros(10,dtype=bool)),bad_words_state=NS(num_bad_words=NS(np=np.zeros(10))),thinking_budget_state=NS(enabled=True,use_thinking_budget=np.zeros(10,dtype=bool)))
    return NS(sampler=sampler,rejection_sampler=NS(synthetic_conditional_rates=None,use_block_verification=False)),NS(num_reqs=n,num_tokens=n*7,num_draft_tokens=n*6,idx_mapping_np=ix,has_structured_output_reqs=False)
processor=NS(org_vocab_size=248320,scale=1,soft_cap=None)
head=NS(tp_size=2,num_embeddings=248320,shard_indices=NS(org_vocab_start_index=rank*124160,num_org_vocab_padding=0))
r,b=setup(10);assert cv.eligible_sampled(r,b,None)
for array,val in [(r.sampler.sampling_states.temperature.np,.7),(r.sampler.sampling_states.top_k.np,21),(r.sampler.sampling_states.min_p.np,.1),(r.sampler.sampling_states.num_logprobs,0),(r.sampler.penalties_state.use_penalty,True),(r.sampler.logit_bias_state.use_logit_bias,True),(r.sampler.bad_words_state.num_bad_words.np,1),(r.sampler.thinking_budget_state.use_thinking_budget,True)]:
    old=array[3];array[3]=val;assert not cv.eligible_sampled(r,b,None);array[3]=old
assert not cv.eligible_sampled(r,b,object())
b.has_structured_output_reqs=True;assert not cv.eligible_sampled(r,b,None);b.has_structured_output_reqs=False
for obj,key,val in [(r.sampler,'compute_nans',True),(r.sampler,'return_sampling_mask',True),(r.rejection_sampler,'synthetic_conditional_rates',object()),(r.rejection_sampler,'use_block_verification',True)]:
    old=getattr(obj,key);setattr(obj,key,val);assert not cv.eligible_sampled(r,b,None);setattr(obj,key,old)
counts=dict(cases=0,rows=0,probability_rows=0,rejection_rows=0)
for n in (1,2,3,8,10,3,1):
 for variant in ('random','ties40','ties500','all_equal','variable','one_sided'):
    r,b=setup(n);lengths=[(i%7)+1 if variant=='variable' else 7 for i in range(n)]
    cu=np.array([0]+list(np.cumsum(lengths)),dtype=np.int32);rows=int(cu[-1])
    torch.manual_seed(777+rank+n);local=(torch.randn(rows,124160,device=device)*3).bfloat16()
    if variant=='ties40':local[:,:20]=30
    if variant=='ties500':local[:,::400]=30
    if variant=='all_equal':local.fill_(1)
    if variant=='one_sided' and rank==0:local.fill_(-float('inf'))
    full=gather(local)
    with cv.target_sampling(r,b,None):
        if n == 1:
            # Kernel supports one row group, but deployment keeps the prior
            # single-request path because no speed improvement was established.
            assert not cv.eligible_sampled(r,b,None)
            out=unpack_topk(gather(pack_topk(local)),124160,local.dtype)
        else:
            assert cv._CURRENT.get().topk_transport
            out=cv.try_logits(processor,head,local)
        assert out is not None and not cv.already_filtered(out)
    assert cv._CURRENT.get() is None
    k=torch.full((rows,),20,device=device,dtype=torch.int32)
    p=torch.tensor([.05,.5,.8,.95,1.][:(min(rows,5))],device=device).repeat((rows+4)//5)[:rows]
    a=reference(full.float(),k,p);z=reference(out.float(),k,p)
    assert torch.equal(a,z),(n,variant,'post-filter',int((a!=z).sum()))
    assert torch.equal(a.softmax(-1),z.softmax(-1)),(n,variant,'probabilities')
    idx=torch.tensor(b.idx_mapping_np,device=device,dtype=torch.int32)
    expanded=idx.repeat_interleave(torch.tensor(lengths,device=device))
    pos=torch.arange(rows,device=device,dtype=torch.int64)+10000
    localpos=torch.cat([torch.arange(k,device=device,dtype=torch.int32) for k in lengths])
    temps=torch.ones(10,device=device);seeds=torch.arange(10,device=device,dtype=torch.int64)+100
    sa=gumbel_sample(a,expanded,temps,seeds,pos,False)
    sz=gumbel_sample(z,expanded,temps,seeds,pos,False);assert torch.equal(sa,sz)
    draft=torch.zeros(rows,device=device,dtype=torch.int64)
    for req,(lo,hi) in enumerate(zip(cu[:-1],cu[1:])):
        draft[lo+1:hi]=sa[lo:hi-1]
        if hi-lo>1 and req%3==1:draft[hi-1]=-1
        if hi-lo>1 and req%3==2:draft[lo+1]=(draft[lo+1]+1)%248320
    args=(None,draft,torch.tensor(cu,device=device),pos,idx,expanded,localpos,temps,seeds,6)
    ta,na=rejection_sample(a,*args);tz,nz=rejection_sample(z,*args)
    assert torch.equal(na,nz);mask=torch.arange(7,device=device)[None,:]<na[:,None];assert torch.equal(ta[mask],tz[mask])
    counts['cases']+=1;counts['rows']+=rows;counts['probability_rows']+=rows;counts['rejection_rows']+=int(na.sum())
    del full,out,a,z
# Transport retains arbitrary boundary ties even for nonfinite scores. Compare
# top-k masks/scores only: probability distributions may be undefined.
for kind in ('nan_many','positive_inf','negative_inf'):
    local=torch.randn(7,124160,device=device,dtype=torch.bfloat16)
    if kind=='negative_inf':local.fill_(-float('inf'))
    else:local[:,::400]=float('nan') if kind=='nan_many' else float('inf')
    out=unpack_topk(gather(pack_topk(local)),124160,local.dtype);full=gather(local)
    thresh=full.topk(20,-1).values[:,-1:]
    expected=full.masked_fill(full<thresh,-float('inf'))
    threshold_out=out.topk(20,-1).values[:,-1:];actual=out.masked_fill(out<threshold_out,-float('inf'))
    if kind=='nan_many':assert torch.equal(torch.isnan(full),torch.isnan(out))
    else:assert torch.equal(expected,actual)
    counts['cases']+=1
# End-to-end transport + ordinary filtering micro latency.
r,b=setup(10);local=torch.randn(70,124160,device=device,dtype=torch.bfloat16)
k=torch.full((70,),20,device=device,dtype=torch.int32);p=torch.full((70,),.95,device=device)
times={}
for mode in ('full','compact'):
    elapsed=[]
    for i in range(20):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        if mode=='full':result=gather(local)
        else:
            with cv.target_sampling(r,b,None):result=cv.try_logits(processor,head,local)
        result=reference(result.float(),k,p)
        torch.cuda.synchronize()
        if i>=5:elapsed.append((time.perf_counter()-start)*1000)
    times[mode]={'mean_ms':statistics.mean(elapsed),'p50_ms':statistics.median(elapsed)}
print(json.dumps(dict(rank=rank,status='PASS',**counts,timing=times)),flush=True);dist.destroy_process_group()
