"""Single-request sparse-probability MTP6 + verified history-copy drafts.

Only accepted request history is searched. MTP prefill still runs on every
iteration to catch its KV/state up to target-verified tokens before any skip.
"""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.triton_utils import triton, tl

@triton.jit
def _find(H, L, IDX, NS, TEMP, PART, STRIDE:tl.constexpr, MAXLEN:tl.constexpr,
          K:tl.constexpr, B:tl.constexpr, ALLOW_STOCHASTIC:tl.constexpr=False):
    r=tl.load(IDX);n=tl.load(L+r);sampled=tl.load(NS)
    pos=tl.program_id(0)*B+tl.arange(0,B)
    valid=(pos+K+16<=n)&(pos<MAXLEN)&(n>K)&(sampled>0)&((tl.load(TEMP+r)==0)|(ALLOW_STOCHASTIC & (tl.load(TEMP+r)>0)))
    match=valid
    for j in range(K):
        a=tl.load(H+r*STRIDE+pos+j,valid,other=-1)
        b=tl.load(H+r*STRIDE+n-K+j,n>=K,other=-2)
        match=match&(a==b)
    tl.store(PART+tl.program_id(0),tl.max(tl.where(match,pos,-1),axis=0))

@triton.jit
def _gather(H,L,IDX,NS,PART,OUT,STRIDE:tl.constexpr,MAXLEN:tl.constexpr,
            K:tl.constexpr,NPART:tl.constexpr,R:tl.constexpr,CAP:tl.constexpr):
    z=tl.arange(0,R);pos=tl.max(tl.load(PART+z,z<NPART,other=-1),axis=0)
    r=tl.load(IDX);n=tl.load(L+r);available=n-pos-K
    size=tl.where((pos>=0)&(available>=CAP),CAP,tl.where((pos>=0)&(available>=16),16,0))
    size=tl.minimum(size,tl.maximum(MAXLEN-n-1,0))
    size=tl.where(size>=16,size,0)
    tl.store(OUT,size);tl.store(OUT+1,n);tl.store(OUT+2,tl.load(NS));tl.store(OUT+3,pos)
    i=tl.arange(0,CAP);v=tl.load(H+r*STRIDE+pos+K+i,i<size,other=-1)
    tl.store(OUT+4+i,v)

class HybridController:
    def __init__(self, runner):
        assert runner.max_num_reqs==1,'Hybrid drafting requires max_num_seqs=1'
        assert runner.num_speculative_steps==32
        assert not runner.scheduler_config.async_scheduling
        # Both MTP argmax and history-copy proposals are deterministic point masses.
        # Reject unsupported probability/synthetic modes rather than reusing stale logits.
        cfg=runner.speculative_config
        assert cfg.draft_sample_method == 'greedy'
        assert cfg.rejection_sample_method == 'standard'
        assert runner.speculator.draft_logits is None
        assert runner.rejection_sampler.synthetic_conditional_rates is None
        assert not runner.rejection_sampler.use_block_verification
        self.stochastic=True
        self.runner=runner;self.cap=32;self.key=16
        self.parts=triton.cdiv(runner.max_model_len,128)
        self.partial=torch.empty(self.parts,dtype=torch.int32,device=runner.device)
        self.output=torch.empty(36,dtype=torch.int64,device=runner.device)
        self.last_req=None;self.mode='baseline';self.calls=0;self.matches=0
        self.audit=True;self.trace_enabled=True
        self.rank=get_tp_group().rank_in_group
        self.trace=Path('/evidence')/f'hybrid-rank{self.rank}.jsonl'
    def prepare(self,batch,num_sampled):
        r=self.runner;sp=r.speculator;sp.hybrid_candidate=None;sp.hybrid_length=6
        assert batch.num_reqs==1
        req=batch.req_ids[0]
        if req!=self.last_req:
            self.last_req=req
            cfg=json.loads(Path(__file__).with_name('hybrid-defaults.json').read_text())
            from sparse_draft import prepare_mode
            prepare_mode(sp,cfg)
            self.mode=cfg.get('mode','baseline');assert self.mode in ('baseline','hybrid')
            self.audit=cfg.get('audit',True);self.trace_enabled=cfg.get('trace',True)
            self.stochastic=cfg.get('stochastic',True)
            if self.trace_enabled:
                with self.trace.open('a') as f:f.write(json.dumps({'event':'request','req':req,'mode':self.mode,'stochastic':self.stochastic,'draft_mode':getattr(sp,'_sparse_mode','sparse')})+'\n')
        sp._sparse_q_active = getattr(sp, '_sparse_mode', 'sparse') == 'sparse'
        if self.mode=='baseline':return
        states=r.req_states
        _find[(self.parts,)](states.all_token_ids.gpu,states.total_len.gpu,batch.idx_mapping,
            num_sampled,r.sampler.sampling_states.temperature.gpu,self.partial,
            states.all_token_ids.gpu.stride(0),r.max_model_len,self.key,128,self.stochastic)
        _gather[(1,)](states.all_token_ids.gpu,states.total_len.gpu,batch.idx_mapping,
            num_sampled,self.partial,self.output,states.all_token_ids.gpu.stride(0),
            r.max_model_len,self.key,self.parts,triton.next_power_of_2(self.parts),32)
        data=self.output.cpu().tolist() if (self.audit or self.trace_enabled) else [int(self.output[0].item())]
        count=int(data[0])
        # Explicit CPU rendezvous prevents divergent collective paths between TP ranks.
        if self.audit:
            check=(req,tuple(data));checks=[None]*get_tp_group().world_size
            dist.all_gather_object(checks,check,group=get_tp_group().cpu_group)
            if any(v!=check for v in checks):raise RuntimeError('Hybrid TP proposal mismatch')
        if count:
            sp._sparse_q_active=False  # deterministic history copy remains one-hot
            sp.hybrid_candidate=self.output[4:4+count]
            sp.hybrid_length=count;self.matches+=1
        self.calls+=1
        if self.trace_enabled:
            with self.trace.open('a') as f:f.write(json.dumps({'event':'proposal','req':req,'length':count,'total_len':data[1],'sampled':data[2],'source':data[3]})+'\n')

def prepare_hybrid(runner,batch,num_sampled):
    if not hasattr(runner,'_ngram_hybrid'):runner._ngram_hybrid=HybridController(runner)
    runner._ngram_hybrid.prepare(batch,num_sampled)
