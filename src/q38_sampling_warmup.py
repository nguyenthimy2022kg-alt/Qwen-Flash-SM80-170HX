"""Warm bounded optional sampling kernels before model collectives can wait.

No model weights, request history or per-step logging is retained.
"""
_initialized=False

def initialize(device):
    global _initialized
    if _initialized:return
    import gc
    import numpy as np
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
    vocab=248320
    req=RequestState(1,256,64,32,vocab,device)
    req.add_request('sampling-warmup',16,list(range(16)),16,128)
    req.apply_staged_writes()
    sampler=Sampler(1,vocab,device,req,num_speculative_tokens=32)
    cases=[dict(temperature=.6),dict(temperature=0),dict(top_k=50),dict(top_k=-1),
           dict(min_p=.05),dict(presence_penalty=.2,frequency_penalty=.2,repetition_penalty=1.1),
           dict(logit_bias={42:-1.}),dict(allowed_token_ids=[42,43]),
           dict(logprobs=1),dict(logprob_token_ids=[42,43]),dict(top_p=1.),dict(top_p=.5),dict(bad_words=['x'])]
    ix=torch.zeros(1,dtype=torch.int64,device=device);ixnp=np.array([0],dtype=np.intp)
    for changes in cases:
        values=dict(temperature=1.,top_k=20,top_p=.95,seed=42,max_tokens=128);values.update(changes)
        params=SamplingParams(**values)
        if changes.get('bad_words'):params._bad_words_token_ids=[[42],[43,44]]
        sampler.add_request(0,16,params);sampler.apply_staged_writes()
        for n in (1,7,17,33):
            logits=torch.zeros((n,vocab),dtype=torch.bfloat16,device=device)
            expanded=torch.zeros(n,dtype=torch.int64,device=device)
            positions=torch.arange(16,16+n,dtype=torch.int64,device=device)
            ids=torch.ones(n,dtype=torch.int32,device=device)
            local=torch.arange(n,dtype=torch.int32,device=device)
            processed=sampler.apply_sampling_params(logits,expanded,ix,ixnp,positions,ids,local)
            # Prefill's non-rejection sampler and optional returned-logprob kernels.
            if n==1:
                sampled,processed=sampler.sample(logits,expanded,ix,ixnp,positions,ids,local,return_logprobs=True)
                for count in (0,1,2,5):
                    compute_topk_scores(logits,count,sampled,None,
                        logprob_token_ids_state=sampler.logprob_token_ids_state,
                        expanded_idx_mapping=expanded,
                        max_per_req_token_ids=sampler.logprob_token_ids_state.max_num_token_ids(ixnp))
            elif 'logprobs' in changes or 'logprob_token_ids' in changes:
                sampled=torch.ones(n,dtype=torch.int64,device=device)
                compute_topk_scores(logits,1,sampled,[0,n],
                    logprob_token_ids_state=sampler.logprob_token_ids_state,
                    expanded_idx_mapping=expanded,
                    max_per_req_token_ids=sampler.logprob_token_ids_state.max_num_token_ids(ixnp))
    # Exercise both compact success and overflow branches before model streams
    # can wait on peer collectives. PyTorch CUDA kernels are lazy-loaded too.
    from q38_compact_ops import pack, compact_filter
    x=torch.randn((7,124160),device=device,dtype=torch.bfloat16)
    y=torch.randn_like(x)
    k=torch.full((7,),20,device=device,dtype=torch.int32)
    for value in (.95,1.,.5):
        p=torch.full((7,),value,device=device)
        compact=compact_filter([pack(x,0),pack(y,124160)],vocab,k,p)
        assert compact is not None
    assert compact_filter([pack(torch.zeros_like(x)),pack(torch.zeros_like(y),124160)],vocab,k,p) is None
    del x,y,k,p,compact
    torch.cuda.synchronize(device)
    del sampler,req,logits,processed,ix,expanded,positions,ids,local
    gc.collect();torch.cuda.empty_cache()
    _initialized=True
    print("SAMPLING_WARMUP_READY",device,flush=True)
