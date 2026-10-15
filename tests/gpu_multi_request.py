"""Ten-slot MTP6 isolation, structured handoff and padded graph sampling checks."""
from types import SimpleNamespace as NS
import json
import torch
import triton
import sparse_draft as sparse
from draft_int8_kernels import quantize
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_weights, get_weight_perm
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_make_workspace_new
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def handoff():
    order=[9,3,1,6,0,2,8,7,4,5]
    tokens=torch.tensor([[r*100+s for s in range(6)] for r in order],device='cuda')
    handler=DraftTokensHandler(torch.device('cuda'))
    batch=NS(num_reqs=10,req_ids=[f'req-{r}' for r in order],has_structured_output_reqs=False)
    handler.set_draft_tokens(batch,tokens)
    result=handler.get_draft_tokens()
    assert result.req_ids==batch.req_ids and result.draft_token_ids==[[-1]*6 for _ in order]
    batch.has_structured_output_reqs=True
    handler.set_draft_tokens(batch,tokens)
    saved_ids=list(batch.req_ids)
    batch.req_ids[0]='new-request-in-reused-slot'
    copied=handler.get_draft_tokens()
    assert copied.req_ids==saved_ids and copied.draft_token_ids==tokens.tolist()
    # A shrinking batch must not return prior rows or prior grammar data.
    smaller=NS(num_reqs=3,req_ids=['new-A','new-B','new-C'],has_structured_output_reqs=False)
    handler.set_draft_tokens(smaller,tokens[:3])
    result=handler.get_draft_tokens()
    assert result.req_ids==smaller.req_ids and result.draft_token_ids==[[-1]*6]*3
    # Removing history copying must not disable the exact q used by verification.
    sp=NS(draft_logits=None,_sparse_q_active=False)
    runner=NS(speculator=sp,speculative_config=NS(method='mtp',num_speculative_tokens=6,draft_sample_method='greedy',rejection_sample_method='standard'),scheduler_config=NS(async_scheduling=False),rejection_sampler=NS(synthetic_conditional_rates=None,use_block_verification=False))
    sparse.prepare_draft(runner)
    assert sp._sparse_q_active and sp._sparse_runtime_checked
    sparse.prepare_draft(runner)
    assert sp._sparse_q_active
    print('10-request structured handoff, slot reuse and sparse-q activation: PASS',flush=True)


def padded_sampling(cap=6):
    # Small genuine Marlin INT8 projection; fake the other TP rank's packet only.
    # This checks packing, batch dimensions, reranking and masked q writes.
    torch.manual_seed(18)
    sparse.N,sparse.K=256,256
    n,k,g=256,256,128
    weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
    quant=torch.empty((n,k),device='cuda',dtype=torch.int8)
    scale=torch.empty((n,k//g),device='cuda')
    quantize[(n*(k//g),)](weight,quant,scale,n,k,g,g,num_warps=4)
    packed=marlin_weights(quant.T.contiguous().to(torch.int32)+128,k,n,8,get_weight_perm(8))
    scales=marlin_permute_scales(scale.T.to(torch.bfloat16),k,n,g)
    model=NS(draft_int8_weight=packed,draft_int8_scales=scales,
             draft_int8_workspace=marlin_make_workspace_new(weight.device,4),
             lm_head=NS(weight=weight),draft_int8_rank=0)
    sp=NS(model=model,vocab_size=2*n,num_speculative_steps=cap,
          sparse_logits=torch.full((10,cap,2*n),42.,device='cuda'),
          sparse_enabled=torch.ones((),device='cuda',dtype=torch.int32))
    def gather(packet,dim=-1):
        other=torch.cat((packet[:,:32]-100,packet[:,32:]+n),dim=-1)
        return torch.cat((packet,other),dim=-1)
    sparse.tensor_model_parallel_all_gather=gather
    hidden=torch.randn(16,k,device='cuda',dtype=torch.bfloat16)
    idx=torch.tensor([9,3,1,6,0,2,8,7,4,5]+[-1]*6,device='cuda')
    pos=torch.arange(16,device='cuda')
    temp=torch.zeros(10,device='cuda')
    seeds=torch.arange(10,device='cuda',dtype=torch.int64)
    step=torch.tensor(2,device='cuda',dtype=torch.int64)
    def run():
        return sparse.sample(sp,hidden,pos,idx,temp,seeds,step,None)
    for _ in range(3):result=run()
    for b,r in enumerate([9,3,1,6,0,2,8,7,4,5]):
        q=sp.sparse_logits[r,2]
        assert torch.isfinite(q).sum()==8
        assert result[b].item()==q.argmax().item()
    assert (sp.sparse_logits[:,:2]==42).all()
    assert (sp.sparse_logits[:,3:]==42).all()
    # Replay with different persistent request indices, including padded rows.
    graph=torch.cuda.CUDAGraph()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph,stream=stream):
        graph_result=run()
    sp.sparse_logits.fill_(42)
    idx.copy_(torch.tensor([1,-1,3]+[-1]*13,device='cuda'))
    graph.replay();torch.cuda.synchronize()
    assert (sp.sparse_logits[[0,2,4,5,6,7,8,9]]==42).all()
    assert torch.isfinite(sp.sparse_logits[1,2]).sum()==8
    assert torch.isfinite(sp.sparse_logits[3,2]).sum()==8
    assert graph_result[0]==sp.sparse_logits[1,2].argmax()
    print(f'batched INT8 screening, reranking, padded rows and graph replay (capacity={cap}): PASS',flush=True)


if __name__=='__main__':
    handoff()
    padded_sampling()
    print(json.dumps({'status':'PASS','request_slots':10,'capacity':6}),flush=True)
