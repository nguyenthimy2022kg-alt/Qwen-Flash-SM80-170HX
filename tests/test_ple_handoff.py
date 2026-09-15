"""Host-only regressions for TP PLE ordering; never loads a model or CUDA."""
import ast,types,threading,time,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"src"

def load_method(path,cls,method,env):
 c=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
 f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
 exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),env)
 return env[method]
class Tensor:
 def __getitem__(self,k):return self
 def view(self,*a):return self
 def reshape(self,*a):return self

def scenario(rank,n,fail=False,mismatch=False):
 events=[]
 class Stream:
  def __init__(self,name):self.name=name
  def synchronize(self):events.append(self.name+'_sync')
  def wait_event(self,e):events.append('wait_event')
  def __enter__(self):return self
  def __exit__(self,*args):pass
 class Owner:
  _d2h_done_event=object()
  def prepare_forward(self,*a):events.append('owner_prepare')
  def wait_for_output_ready(self):
   events.append('owner_ready')
   if fail:raise RuntimeError('injected producer error')
 def gather(out,item,group):
  events.append('cpu_ready' if isinstance(item,tuple) else 'cpu_done')
  out[:]=[item,item]
  if mismatch and isinstance(item,tuple):out[1]=(item[0],item[1],item[2]+1,None)
 cuda=types.SimpleNamespace(current_stream=lambda d:Stream('main'),stream=lambda s:s)
 env={'torch':types.SimpleNamespace(cuda=cuda,uint8=object()),'trace':lambda *a,**k:None,'dist':types.SimpleNamespace(all_gather_object=gather),'_stream_wait_value32':lambda *a:events.append('wait_flag'),'_stream_write_value32':lambda *a:events.append('write_flag')}
 f=load_method(p/'tp_ple_transport.py','TpPleTransport','prepare_forward',env)
 o=types.SimpleNamespace(closed=False,active=False,max_tokens=8192,sequence=30,rank=rank,device=rank,serialize_large_inputs=True,owner=Owner() if rank==0 else None,cpu_group='gloo',stream=Stream('transfer'),sem=types.SimpleNamespace(flag_tensor=Tensor()),output=Tensor(),done=types.SimpleNamespace(record=lambda s:events.append('done_record')),comm=types.SimpleNamespace(broadcast=lambda *a,**k:events.append('broadcast')),verify_remaining=0,embedding_dim=512,_counts={'requests':0,'decode':0,'bytes':0})
 try:f(o,1,n,False)
 except RuntimeError:
  assert fail or mismatch
  assert 'broadcast' not in events
  return
 assert not fail and not mismatch
 if n>112:
  assert events.index('main_sync')<events.index('cpu_ready')<events.index('broadcast')<events.index('transfer_sync')<events.index('cpu_done')
  if rank==0:assert events.index('owner_ready')<events.index('cpu_ready')
 else:assert not any(x in events for x in ['main_sync','cpu_ready','owner_ready','transfer_sync','cpu_done'])
class PleHandoffTests(unittest.TestCase):
 def test_large_and_decode_order(self):
  for rank in [0,1]:
   for tokens in [7,112,113,4536,8184]:
    with self.subTest(rank=rank,tokens=tokens):scenario(rank,tokens)
 def test_producer_error_stops_before_broadcast(self):scenario(0,4536,fail=True)
 def test_rank_shape_mismatch_stops_before_broadcast(self):scenario(1,4536,mismatch=True)
 def test_wait_requires_gpu_completion_and_times_out(self):
  wait=load_method(p/'vllm_ple_gds_runner.py','InputDrivenGdsPleConnector','wait_for_output_ready',{'time':time,'TicketError':RuntimeError})
  o=types.SimpleNamespace(_condition=threading.Condition(),_active_sequence=5,_error=None,_closed=False,_records={5:{'completed':False,'submitted':True}})
  finished=threading.Event()
  def waiter():wait(o,1);finished.set()
  t=threading.Thread(target=waiter);t.start();assert not finished.wait(.03)
  with o._condition:o._records[5]['completed']=True;o._condition.notify_all()
  t.join(1);assert finished.is_set()
  o._records[5]['completed']=False
  try:wait(o,.01)
  except RuntimeError as e:assert 'timed out' in str(e)
  else:raise AssertionError('missing timeout')

if __name__=="__main__":unittest.main()
