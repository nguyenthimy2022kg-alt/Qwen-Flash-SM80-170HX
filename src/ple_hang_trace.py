"""Host-only breadcrumbs. No GPU reads, event queries or synchronization."""
import os,time,json,threading
_lock=threading.Lock()
_fd=None
_rows=0
_enabled=os.getenv('Q38_PLE_HANG_TRACE')=='1'
def trace(stage,sequence,tokens,**extra):
 global _fd,_rows
 if not _enabled or (tokens<=112 and sequence>16):return
 try:
  with _lock:
   if _rows>=20000:return
   if _fd is None:_fd=os.open('/evidence/ple-trace-%d.jsonl'%os.getpid(),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
   os.write(_fd,(json.dumps(dict(time=time.time(),thread=threading.current_thread().name,stage=stage,sequence=sequence,tokens=tokens,**extra))+'\n').encode())
   _rows+=1
 except OSError:pass
