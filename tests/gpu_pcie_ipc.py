"""GPU-only regression for the TP2 PCIe wrapper; does not load a model.

Run with torchrun --standalone --nproc_per_node=2 and Q38_PCIE_IPC=1.
Requires an idle, compatible two-GPU installation; writes rank reports beneath
Q38_PCIE_TEST_OUTPUT (default /tmp/q38-pcie-ipc-test).
"""
import os,json,gc,statistics
from pathlib import Path
import torch,torch.distributed as dist
from vllm.config import VllmConfig,ParallelConfig,SchedulerConfig,set_current_vllm_config
from vllm.distributed import init_distributed_environment,initialize_model_parallel,get_tp_group,destroy_model_parallel
from vllm.distributed.parallel_state import graph_capture,GraphCaptureContext
torch.set_num_threads(2)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
ctx=set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2),scheduler_config=SchedulerConfig(async_scheduling=False,max_num_seqs=10,max_model_len=16384,is_encoder_decoder=False)));ctx.__enter__()
init_distributed_environment(world_size=2,rank=rank,local_rank=rank)
initialize_model_parallel(tensor_model_parallel_size=2,pipeline_model_parallel_size=1)
tp=get_tp_group();comm=tp.device_communicator;pcie=comm.pcie_ipc_comm;assert pcie is not None
output=Path(os.environ.get('Q38_PCIE_TEST_OUTPUT','/tmp/q38-pcie-ipc-test'));output.mkdir(parents=True,exist_ok=True)
rows=[];graphs=[]
def rendezvous():
 torch.cuda.synchronize();dist.barrier(group=tp.cpu_group)
def equal(y,e):
 ok=torch.tensor(int(torch.equal(y,e)));dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=tp.cpu_group);assert bool(ok)
def capture(x,active,calls=1):
 comm.pcie_ipc_comm=pcie if active else None
 stream=torch.cuda.Stream()
 with graph_capture(device,GraphCaptureContext(stream)) as cap:
  for _ in range(2):y=tp.all_reduce(x)
  rendezvous();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=cap.stream):
   for _ in range(calls):y=tp.all_reduce(x)
 comm.pcie_ipc_comm=pcie
 return g,y
def measure(g,calls):
 values=[]
 for _ in range(3):
  rendezvous();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(4):g.replay()
  b.record();b.synchronize();v=torch.tensor(a.elapsed_time(b)*1000/4/calls,dtype=torch.float64);dist.all_reduce(v,op=dist.ReduceOp.MAX,group=tp.cpu_group);values.append(float(v))
 return values
eligible_rows=sorted(set(range(1,11))|{14,21,28,33,35,42,49,56,63,70})
cases=[(m,2560,torch.bfloat16,True) for m in eligible_rows]+[(11,2560,torch.bfloat16,False),(64,2560,torch.bfloat16,False),(71,2560,torch.bfloat16,False),(7,1280,torch.bfloat16,False),(7,2560,torch.float32,False)]
for m,n,dtype,eligible in cases:
 gen=torch.Generator().manual_seed(3800+m+n)
 full=torch.randn(2,m,n,generator=gen).to(dtype);x=full[rank].to(device);expected=(full[0].float()+full[1].float()).to(dtype).to(device)
 assert pcie.supports(x)==eligible
 y=tp.all_reduce(x);torch.cuda.synchronize();equal(y,expected)
 # Consecutive eager outputs must remain independent.
 y2=tp.all_reduce(x);torch.cuda.synchronize();assert y.data_ptr()!=y2.data_ptr();equal(y,expected)
 g,y=capture(x,True)
 for k in (1,2,3):
  x.copy_(full[rank]*k);g.replay();torch.cuda.synchronize();equal(y,(full[0]*k+full[1]*k).to(device))
 row=dict(shape=[m,n],dtype=str(dtype),candidate=eligible,ok=True)
 # Screen only previously unmeasured shapes; preserve the prior 1/7/33 scope.
 if eligible and m not in (10,14,21,35,49,63,70):
  timings={}
  for label,active in [('current',False),('pcie',True)]:
   tg,ty=capture(x,active,48);timings[label]=measure(tg,48);equal(ty,(full[0]*3+full[1]*3).to(device));del tg,ty
  row.update(samples_us=timings,median_us={k:statistics.median(v) for k,v in timings.items()})
  if rank==0:print('TIMING',m,row['median_us'],flush=True)
 graphs.append((g,x,y,m,n));rows.append(row)
 # Release all eager outputs before the next capture.
 del y2;gc.collect()
for i in range(150):
 # Shrink and refill; alternate fast and fallback graphs on the same stream.
 g,x,y,m,n=graphs[(i*7)%len(graphs)];value=(rank+1)*(i%17+1);x.fill_(value)
 if rank==i%2:torch.cuda._sleep(200000)
 g.replay();torch.cuda.synchronize();equal(y,torch.full_like(y,3*(i%17+1)))
buf=torch.full((7,5120),rank+1,device=device,dtype=torch.bfloat16);x=buf[:,::2]
assert not pcie.supports(x);y=tp.all_reduce(x);torch.cuda.synchronize();equal(y,torch.full_like(y,3))
(output/f'integration-rank{rank}.json').write_text(json.dumps(dict(status='pass',cases=rows,interleaved_replays=150,strided_fallback=True,workspace_bytes=pcie.bytes),indent=2))
print('INTEGRATION PASS',rank,flush=True)
graphs.clear();del g;gc.collect();rendezvous();destroy_model_parallel();dist.destroy_process_group()
