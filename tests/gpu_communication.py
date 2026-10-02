"""GPU regression for the pinned two-rank PCIe communication path.

Run with torchrun --standalone --nproc-per-node=2; Q38_PCIE_IPC=1.
Mount a writable /evidence directory for the JSON results.
"""
import os,json
from pathlib import Path
import torch,torch.distributed as dist
from vllm.config import VllmConfig,ParallelConfig,SchedulerConfig,set_current_vllm_config
from vllm.distributed import init_distributed_environment,initialize_model_parallel,get_tp_group,destroy_model_parallel
from vllm.distributed.parallel_state import graph_capture,GraphCaptureContext
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
ctx=set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2),scheduler_config=SchedulerConfig(async_scheduling=False,max_model_len=16384,is_encoder_decoder=False)));ctx.__enter__()
init_distributed_environment(world_size=2,rank=rank,local_rank=rank)
initialize_model_parallel(tensor_model_parallel_size=2,pipeline_model_parallel_size=1)
tp=get_tp_group();pcie=tp.device_communicator.pcie_ipc_comm;assert pcie is not None
rows=[];graphs=[];default=torch.cuda.current_stream()
for m,n,dtype,eligible in [(1,2560,torch.bfloat16,True),(7,2560,torch.bfloat16,True),(33,2560,torch.bfloat16,True),(2,2560,torch.bfloat16,False),(64,2560,torch.bfloat16,False),(7,1280,torch.bfloat16,False),(7,2560,torch.float32,False)]:
 x=torch.full((m,n),rank+1,device='cuda',dtype=dtype);expect=torch.full_like(x,3)
 assert pcie.supports(x)==eligible
 out=tp.all_reduce(x);torch.cuda.synchronize();assert torch.equal(out,expect)
 # Outputs from consecutive calls may coexist; no shared output aliasing.
 out2=tp.all_reduce(x);torch.cuda.synchronize();assert out.data_ptr()!=out2.data_ptr()
 stream=torch.cuda.Stream()
 with graph_capture(torch.device('cuda',rank),GraphCaptureContext(stream)) as gc:
  for _ in range(2):y=tp.all_reduce(x)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=gc.stream):y=tp.all_reduce(x)
 for i in range(3):
  x.fill_((rank+1)*(i+1));g.replay();torch.cuda.synchronize();assert torch.equal(y,expect*(i+1))
 graphs.append((g,x,y,expect));rows.append(dict(shape=[m,n],dtype=str(dtype),candidate=eligible,ok=True))
for i in range(21):
 g,x,y,expect=graphs[i%len(graphs)];x.fill_((rank+1)*(i%5+1));g.replay();torch.cuda.synchronize();assert torch.equal(y,expect*(i%5+1))
# A strided shape falls back without participating in the IPC protocol.
x=torch.ones(7,5120,device='cuda',dtype=torch.bfloat16)[:,::2]*(rank+1)
# Multiplication may make it contiguous; explicitly retain a strided view.
buf=torch.full((7,5120),rank+1,device='cuda',dtype=torch.bfloat16);x=buf[:,::2]
assert not pcie.supports(x)
y=tp.all_reduce(x);torch.cuda.synchronize();assert torch.equal(y,torch.full_like(y,3))
Path(f'/evidence/integration-rank{rank}.json').write_text(json.dumps(dict(status='pass',cases=rows,interleaved_replays=21,strided_fallback=True),indent=2))
print('INTEGRATION PASS',rank,flush=True)
graphs.clear();torch.cuda.synchronize();dist.barrier(group=tp.cpu_group)
destroy_model_parallel();dist.destroy_process_group()
