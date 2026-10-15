"""MTP6: INT8 screening, BF16 top-8 q, native exact rejection.

q has support ONLY on the selected eight tokens. It is not an approximation
passed off as full softmax: the exact same sparse q samples drafts and verifies
acceptance/residuals. Target logits and target sampling remain untouched.
"""
import types,json
import torch,triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.distributed import tensor_model_parallel_all_gather
import vllm._custom_ops as ops
from vllm.scalar_type import scalar_types
N,K,TOP=124160,2560,32

@triton.jit
def clear_row(Q,IDX,STEP,V:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 r=tl.load(IDX+tl.program_id(1)).to(tl.int64);step=tl.load(STEP).to(tl.int64)
 i=tl.program_id(0)*B+tl.arange(0,B)
 tl.store(Q+(r*S+step)*V+i,-float('inf'),(i<V)&(r>=0))

@triton.jit
def choose(VALUES,IDS,Q,OUT,IDX,TEMP,SEED,POS,STEP,ENABLED,
           V:tl.constexpr,S:tl.constexpr,C:tl.constexpr=8):
 b=tl.program_id(0);r=tl.load(IDX+b).to(tl.int64);step=tl.load(STEP).to(tl.int64)
 j=tl.arange(0,C);ids=tl.load(IDS+b*C+j).to(tl.int64);raw=tl.load(VALUES+b*C+j).to(tl.float32)
 temp=tl.load(TEMP+r,r>=0,other=0);enabled=tl.load(ENABLED)!=0
 # A separate counter-based stream from target, rejection, and bonus sampling.
 # Native verifier uses (seed, position); this uses a salted seed + token id.
 seed=tl.load(SEED+r,r>=0,other=0)^0x53A91B27;pos=tl.load(POS+b)+1
 gseed=tl.randint(seed,pos);u=tl_rand32(gseed,ids,includes_zero=False)
 noise=-tl.log(-libdevice.log1p(-u))
 scores=tl.where((temp>0)&enabled,raw/tl.maximum(temp,1.e-20)+noise,raw)
 mx=tl.max(scores,0);token=tl.min(tl.where(scores==mx,ids,2147483647),0)
 tl.store(OUT+b,token)
 # Store raw pre-temperature logits as required by native rejection sampling.
 tl.store(Q+(r*S+step)*V+ids,raw,r>=0)


def sample(self,hidden_states,positions,idx_mapping,temperature,seeds,current_draft_step,draft_logits):
 batch=hidden_states.shape[0]
 m=self.model
 approximate=ops.marlin_gemm(hidden_states,None,m.draft_int8_weight,None,m.draft_int8_scales,None,None,None,None,None,m.draft_int8_workspace,scalar_types.uint8b128,batch,N,K,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True)
 ids=torch.topk(approximate,TOP,dim=-1).indices
 weights=m.lm_head.weight.index_select(0,ids.reshape(-1))
 if batch==1:
  raw=torch.nn.functional.linear(hidden_states,weights).float()
 else:
  raw=torch.bmm(weights.reshape(batch,TOP,K),hidden_states.unsqueeze(-1)).squeeze(-1).float()
 packet=torch.cat((raw,(ids+m.draft_int8_rank*N).float()),dim=-1)
 both=tensor_model_parallel_all_gather(packet,dim=-1).reshape(batch,2,2,TOP)
 scores=both[:,:,0,:].reshape(batch,2*TOP);allids=both[:,:,1,:].reshape(batch,2*TOP)
 values,indices=torch.topk(scores,8,dim=-1)
 selected=allids.gather(-1,indices).to(torch.int64)
 result=torch.empty((batch,),dtype=torch.int64,device=hidden_states.device)
 clear_row[(triton.cdiv(self.vocab_size,1024),batch)](self.sparse_logits,idx_mapping,current_draft_step,self.vocab_size,self.num_speculative_steps,1024)
 choose[(batch,)](values,selected,self.sparse_logits,result,idx_mapping,temperature,seeds,positions,current_draft_step,self.sparse_enabled,self.vocab_size,self.num_speculative_steps)
 return result


def setup(sp):
 assert 1<=sp.max_num_reqs<=10 and sp.num_speculative_steps==6 and sp.vocab_size==2*N
 assert not sp.use_fp64_gumbel,'FP64 sampling is outside this release scope'
 assert sp.draft_logits is None
 sp.sparse_logits=torch.full((sp.max_num_reqs,sp.num_speculative_steps,sp.vocab_size),-torch.inf,device=sp.device,dtype=torch.float32)
 sp.sparse_enabled=torch.ones((),dtype=torch.int32,device=sp.device)
 sp._sparse_q_active=False
 sp.sample_draft=types.MethodType(sample,sp)
 print('SPARSE_DRAFT_READY '+json.dumps({'support':8,'prefilter_per_rank':32,'capacity':sp.num_speculative_steps,'draft_steps':6,'q_bytes':sp.sparse_logits.numel()*4}),flush=True)


def prepare_draft(runner):
 """Enable exact sparse-q verification without searching or copying history."""
 sp=runner.speculator
 if not getattr(sp,'_sparse_runtime_checked',False):
  cfg=runner.speculative_config
  assert cfg.method=='mtp' and cfg.num_speculative_tokens==6
  assert not runner.scheduler_config.async_scheduling
  assert cfg.draft_sample_method=='greedy'
  assert cfg.rejection_sample_method=='standard'
  assert sp.draft_logits is None
  assert runner.rejection_sampler.synthetic_conditional_rates is None
  assert not runner.rejection_sampler.use_block_verification
  sp._sparse_runtime_checked=True
 sp._sparse_q_active=True


def verification_logits(sp):
 return sp.sparse_logits if sp._sparse_q_active else None
