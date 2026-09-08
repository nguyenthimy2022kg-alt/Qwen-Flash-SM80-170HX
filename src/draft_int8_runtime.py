"""Draft-only full-vocabulary INT8 screening followed by original BF16 reranking.

M=1 fast path matches the isolated screen. Other shapes retain native behavior.
Target weights/logits/verification are unchanged. Packing precedes cache profiling.
"""
import json,time,types,os
from pathlib import Path
import torch,triton
import vllm._custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size,tensor_model_parallel_all_gather
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_weights,get_weight_perm
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales,marlin_make_workspace_new
from draft_int8_kernels import quantize

N,K,G,TOP=124160,2560,128,32
def get_top_tokens(self,hidden_states):
    if hidden_states.shape[0]!=1:
        return self.compute_logits(hidden_states).argmax(dim=-1)
    approximate=ops.marlin_gemm(hidden_states,None,self.draft_int8_weight,None,self.draft_int8_scales,None,None,None,None,None,self.draft_int8_workspace,scalar_types.uint8b128,1,N,K,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True)
    local_ids=torch.topk(approximate,TOP,dim=-1).indices[0]
    exact_weights=self.lm_head.weight.index_select(0,local_ids)
    logits=torch.nn.functional.linear(hidden_states,exact_weights)
    values=logits.max(-1).values
    token=torch.where(logits[0]==values[0],local_ids+self.draft_int8_rank*N,2*N).min().reshape(1)
    pair=torch.stack((values.float(),token.float()),-1)
    both=tensor_model_parallel_all_gather(pair,dim=-1).reshape(1,2,2)
    best=both[:,:,0].argmax(-1,keepdim=True)
    return both[:,:,1].gather(-1,best).squeeze(-1).to(torch.int64)

def install():
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    if getattr(DraftModelSpeculator,'_draft_int8_installed',False):return
    original=DraftModelSpeculator._validate_local_argmax_reduction
    def validate(self):
        assert self.use_local_argmax_reduction
        assert self.speculative_config.num_speculative_tokens==6
        assert self.speculative_config.draft_sample_method!='probabilistic'
        assert get_tensor_model_parallel_world_size()==2
        model=self.model;head=model.lm_head;weight=head.weight;rank=get_tensor_model_parallel_rank()
        assert tuple(weight.shape)==(N,K) and weight.dtype==torch.bfloat16
        assert head.shard_indices.org_vocab_start_index==rank*N
        assert head.shard_indices.num_org_vocab_padding==0
        p=model.logits_processor
        assert p.scale==1.0 and p.soft_cap is None and p.head_dtype in (None,torch.bfloat16) and not p.logits_as_input
        assert not hasattr(model,'draft_int8_weight')
        started=time.perf_counter();free_before,total=torch.cuda.mem_get_info(weight.device)
        # Only ~323 MB persists; chunked packing bounds both host and device scratch.
        if free_before<768*1024**2:raise RuntimeError('Insufficient free GPU memory for bounded draft INT8 setup')
        ptr=weight.data_ptr()
        qw=torch.empty((K//16,N*4),device=weight.device,dtype=torch.int32)
        scales=torch.empty((K//G,N),device=weight.device,dtype=torch.bfloat16)
        perm=get_weight_perm(8)
        with torch.no_grad():
            for first in range(0,N,4096):
                last=min(first+4096,N);rows=last-first
                q=torch.empty((rows,K),device=weight.device,dtype=torch.int8)
                s=torch.empty((rows,K//G),device=weight.device,dtype=torch.float32)
                quantize[(rows*(K//G),)](weight[first:last],q,s,rows,K,G,triton.next_power_of_2(G),num_warps=4)
                packed=marlin_weights(q.T.contiguous().to(torch.int32)+128,K,rows,8,perm)
                qw[:,first*4:last*4].copy_(packed)
                scales[:,first:last].copy_(s.T)
                del q,s,packed
            scales=marlin_permute_scales(scales,K,N,G)
        model.register_buffer('draft_int8_weight',qw,persistent=False)
        model.register_buffer('draft_int8_scales',scales,persistent=False)
        model.register_buffer('draft_int8_workspace',marlin_make_workspace_new(weight.device,4),persistent=False)
        model.draft_int8_rank=rank
        model.get_top_tokens=types.MethodType(get_top_tokens,model)
        assert model.lm_head.weight.data_ptr()==ptr
        original(self)
        # Optional saved-state validation; normal startup requires no private fixture.
        retained = None
        sample_count = 0
        fixture_dir = os.environ.get('Q38_DRAFT_VALIDATION_DIR')
        if fixture_dir:
            saved = Path(fixture_dir) / f'draft-hidden-rank{rank}.pt'
            fixture = torch.load(saved, weights_only=True, map_location='cpu')
            states = fixture['states'][16:].to(weight.device)
            actual = fixture['tokens'][16:]
            sample_count = len(states)
            if not sample_count or sample_count != len(actual):
                raise RuntimeError('Draft validation fixture is empty or malformed')
            retained = 0
            with torch.inference_mode():
                for i in range(sample_count):
                    token = int(model.get_top_tokens(states[i:i+1]))
                    retained += int(token == int(actual[i]))
            if retained != sample_count:
                raise RuntimeError(f'Draft selections retained {retained}/{sample_count}')
            del states, actual, fixture
        torch.cuda.synchronize();free_after,_=torch.cuda.mem_get_info(weight.device)
        record={'rank':rank,'group_size':G,'rerank_per_gpu':TOP,'samples':sample_count,'retained':retained,'weight_bytes':qw.numel()*qw.element_size()+scales.numel()*scales.element_size(),'free_before_bytes':free_before,'free_after_bytes':free_after,'total_gpu_bytes':total,'setup_seconds':time.perf_counter()-started,'target_head_pointer_unchanged':True,'other_batch_sizes':'original compute_logits().argmax fallback'}
        Path(f'/evidence/draft-int8-rank{rank}.json').write_text(json.dumps(record,indent=2))
        print('DRAFT_INT8_READY '+json.dumps(record),flush=True)
    DraftModelSpeculator._validate_local_argmax_reduction=validate
    DraftModelSpeculator._draft_int8_installed=True
