"""TP2 PLE transport: one strict GDS producer, one dedicated NCCL broadcast stream.

Main model streams wait at PLE consumption; release waits for broadcast reads
before allowing the owner output to be reused. No PLE payload crosses CPU memory.
"""
import json,os,hashlib
import torch.distributed as dist
from pathlib import Path
import torch
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore
from vllm_ple_gds_runner import _stream_wait_value32,_stream_write_value32,_aligned_cuda_output

class TpPleTransport:
 def __init__(self,layer,device,max_tokens,owner=None,input_sources=None):
  self.layer=layer;self.device=torch.device(device);self.max_tokens=max_tokens;self.owner=owner
  group=get_tp_group();self.rank=group.rank_in_group
  assert group.world_size==2 and (owner is not None)==(self.rank==0)
  self.closed=False;self.close_count=0;self.active=False;self.dummy=False;self.tokens=0;self.sequence=0
  self.embedding_dim=layer.embedding_dim
  if owner is None:
   self._output_owner,self.output=_aligned_cuda_output(max_tokens,layer.embedding_dim,self.device)
   self.sem=CpuGpuSemaphore(self.device)
  else:self.output=owner.output;self.sem=owner._sem
  self.stream=torch.cuda.Stream(device=self.device);self.done=torch.cuda.Event()
  self.comm=PyNcclCommunicator(group.cpu_group,self.device)
  assert self.comm.available and not self.comm.disabled
  # Copy the two-byte global scale once during setup, before model profiling.
  if self.rank==0:scale=layer._offload_weight_scale
  else:
   scale=torch.empty((),dtype=torch.bfloat16,device=self.device)
   layer.register_buffer('_offload_weight_scale',scale,persistent=False)
  self.comm.broadcast(scale.reshape(-1).view(torch.uint8),src=0)
  torch.cuda.current_stream(self.device).synchronize()
  self.input_sources=input_sources;self.verify_remaining=int(os.getenv('Q38_TP_PLE_VERIFY_STEPS','0'))
  self._counts={'requests':0,'decode':0,'dummy':0,'bytes':0,'verified':0}
  layer._ple_gds_input_connector=self
  print('TP_PLE_TRANSPORT_READY',self.rank,'owner',self.rank==0,'bytes_per_token',self.embedding_dim,flush=True)
 def prepare_forward(self,num_reqs,num_tokens,dummy_run):
  assert not self.closed and not self.active
  assert 0<num_tokens<=self.max_tokens
  if dummy_run:return self.signal_dummy_outputs(num_tokens)
  self.active=True;self.dummy=False;self.tokens=num_tokens;self.sequence+=1
  if self.owner:self.owner.prepare_forward(num_reqs,num_tokens,False)
  with torch.cuda.stream(self.stream):
   if self.owner:
    # Bind this transfer to the current input generation. This event is recorded
    # only after the preceding main-stream release/reset and current input
    # staging. Without it, flag==1 can still refer to the previous generation.
    self.stream.wait_event(self.owner._d2h_done_event)
    _stream_wait_value32(self.stream,self.sem.flag_tensor,1)
   else:
    # Do not overwrite the remote output while the previous step still consumes it.
    _stream_wait_value32(self.stream,self.sem.flag_tensor,0)
   self.comm.broadcast(self.output[:num_tokens].view(torch.uint8).reshape(-1),src=0,stream=self.stream)
   if not self.owner:_stream_write_value32(self.stream,self.sem.flag_tensor,1)
   self.done.record(self.stream)
  if self.verify_remaining:
   # Initial warmup-only cross-rank checks, excluded from formal timings.
   self.stream.synchronize()
   def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
   record={'sequence':self.sequence,'tokens':num_tokens,'requests':num_reqs,'payload':digest(self.output[:num_tokens])}
   if self.input_sources is not None:
    ids,qsl,context=self.input_sources
    record['inputs']=[digest(ids[:num_tokens]),digest(qsl[:num_reqs+1]),digest(context[:num_reqs])]
   pairs=[None,None];dist.all_gather_object(pairs,record,group=get_tp_group().cpu_group)
   assert pairs[0]==pairs[1],('TP PLE data mismatch',pairs)
   self.verify_remaining-=1;self._counts['verified']+=1
   if not self.verify_remaining:print('TP_PLE_INITIAL_CHECKS_PASSED',self.rank,flush=True)
  self._counts['requests']+=1;self._counts['decode']+=int(num_tokens==1);self._counts['bytes']+=num_tokens*self.embedding_dim
 def signal_dummy_outputs(self,num_tokens):
  assert not self.closed and not self.active
  self.active=True;self.dummy=True;self.tokens=num_tokens
  if self.owner:self.owner.signal_dummy_outputs(num_tokens)
  else:
   self.output[:num_tokens].zero_();self.sem.signal(torch.cuda.current_stream(self.device))
  self._counts['dummy']+=1
 def consume_output(self,hidden_states,input_ids):
  n=input_ids.reshape(-1).numel()
  if not torch.compiler.is_compiling():
   assert self.active and n<=self.tokens
  torch.ops.vllm.ple_offload_wait(self.sem.flag_tensor,self.output,hidden_states)
  return self.output[:n,:self.embedding_dim]
 def release_outputs(self):
  assert self.active
  stream=torch.cuda.current_stream(self.device)
  if not self.dummy:stream.wait_event(self.done)
  if self.owner:self.owner.release_outputs()
  else:self.sem.reset(stream)
  self.active=False
 def stats(self):return dict(self._counts,close_count=self.close_count,rank=self.rank,active=self.active,owner=self.owner is not None)
 def close(self):
  if self.closed:return
  if self.active:raise RuntimeError('closing TP PLE with an active output')
  self.stream.synchronize()
  if self.owner:self.owner.close()
  self.closed=True;self.close_count+=1
  Path(f'/evidence/tp-ple-rank{self.rank}.json').write_text(json.dumps(self.stats()))
  self.layer._ple_gds_input_connector=None
  self.comm.destroy()
