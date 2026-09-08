#!/usr/bin/env python3
"""启动、停止社区运行包；仅操作带有本项目标签的容器。"""
from pathlib import Path
import argparse,ctypes,json,os,re,subprocess as sp,sys,time,datetime
ROOT=Path(__file__).resolve().parents[1]
LABEL='qwen-flash-sm80.managed'

def read_config(path):
    c=json.loads(path.read_text())
    for field in ('models_root','ple_artifact','cufile_config','ple_identity','validation_dir'):
        if c.get(field):
            p=Path(c[field]).expanduser()
            c[field]=str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if not 1024<=c['port']<=65535:raise ValueError('端口必须在 1024～65535')
    if c['mode'] not in ('tep2','tp2'):raise ValueError('仅支持 tep2 / tp2')
    if len(c['gpu_ids'])!=2 or len(set(c['gpu_ids']))!=2:raise ValueError('需要两张不同的显卡')
    if Path(c['model_subdir']).is_absolute() or '..' in Path(c['model_subdir']).parts:raise ValueError('model_subdir 必须位于 models_root 内')
    if not 0<c['container_memory_gib']<=c['container_memory_and_swap_gib']:raise ValueError('内存与含交换空间上限无效')
    if c['min_host_available_gib']<=0:raise ValueError('主机可用内存保护必须为正值')
    if not isinstance(c['draft_int8'],bool):raise ValueError('draft_int8 必须为布尔值')
    for f in ('models_root','ple_artifact','cufile_config','ple_identity','validation_dir'):
        if c.get(f) and (':' in c[f] or '\n' in c[f]):raise ValueError('挂载路径不能含冒号或换行')
    return c

def gpu_inventory():
    output=sp.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True)
    result={}
    for line in output.splitlines():
        index,uuid,bdf=[x.strip() for x in line.split(',')]
        bdf=bdf[-12:].lower()
        result[index]=result[uuid]={'uuid':uuid,'bdf':bdf}
    return result

def build_command(c,name,run,devices):
    selected=[devices[str(x)] for x in c['gpu_ids']]
    if len({x['uuid'] for x in selected})!=2:raise ValueError('所选 GPU 实际为同一设备')
    model='/models/'+c['model_subdir']
    cmd=['docker','run','-d','--pull=never','--name',name,'--label',LABEL+'=1',
         '--memory',str(c['container_memory_gib'])+'g','--memory-swap',str(c['container_memory_and_swap_gib'])+'g',
         '--gpus','all','--ipc=host','-p',f"127.0.0.1:{c['port']}:8000"]
    mounts=[(c['models_root'],'/models','ro'),(c['ple_artifact'],'/ple','ro'),(c['cufile_config'],'/config/cufile.json','ro'),(c['ple_identity'],'/config/ple-identity.json','ro'),
            ('/run/udev','/run/udev','ro'),(str(run/'service'),'/evidence','rw'),(str(run/'cache'),'/root/.cache/vllm','rw')]
    if c.get('validation_dir'):mounts.append((c['validation_dir'],'/validation','ro'))
    for src,dst,mode in mounts:cmd+=['-v',f'{src}:{dst}:{mode}']
    env=json.loads((ROOT/'config/runtime-env.json').read_text())
    env.update(CUDA_VISIBLE_DEVICES=','.join(x['uuid'] for x in selected),VLLM_PLE_GDS_GPU_UUID=selected[0]['uuid'],VLLM_PLE_GDS_GPU_BDF=selected[0]['bdf'],VLLM_PLE_GDS_ARTIFACT='/ple',Q38_DRAFT_INT8=str(int(c['draft_int8'])),Q38_PLE_IDENTITY='/config/ple-identity.json')
    if c.get('validation_dir'):env['Q38_DRAFT_VALIDATION_DIR']='/validation'
    for k,v in env.items():cmd+=['-e',k+'='+v]
    args=json.loads((ROOT/'config/vllm-args.json').read_text())
    args=[x.replace('{MODEL}',model) for x in args]
    if c['mode']=='tp2':args.remove('--enable-expert-parallel')
    idx=args.index('--speculative-config')+1;spec=json.loads(args[idx]);spec['use_local_argmax_reduction']=c['draft_int8'];args[idx]=json.dumps(spec)
    return cmd+[c['image'],'vllm','serve']+args

def check_offset(guard):
    if not guard:return
    lib=ctypes.CDLL('libnvidia-ml.so.1')
    if lib.nvmlInit_v2()!=0:raise RuntimeError('NVML 初始化失败')
    try:
        h=ctypes.c_void_p();lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
        if lib.nvmlDeviceGetHandleByUUID(guard['gpu_uuid'].encode(),ctypes.byref(h))!=0:raise RuntimeError('偏移保护 GPU 未找到')
        lib.nvmlDeviceGetGpcClkVfOffset.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];v=ctypes.c_int()
        if lib.nvmlDeviceGetGpcClkVfOffset(h,ctypes.byref(v))!=0:raise RuntimeError('无法读取核心偏移')
        if v.value!=guard['expected']:raise RuntimeError('核心偏移不符合本机配置；未修改硬件')
    finally:lib.nvmlShutdown()

def available_gib():
    value=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(value.split()[1])/1024**2

def inspect(name):return json.loads(sp.check_output(['docker','inspect',name]))[0]
def stop(name):
    info=inspect(name)
    if info['Config'].get('Labels',{}).get(LABEL)!='1':raise RuntimeError('拒绝停止不属于本项目的容器')
    sp.run(['docker','stop','-t','20',name],check=True)

def supervise(run):
    c=json.loads((run/'config.json').read_text());cmd=json.loads((run/'command.json').read_text());name=cmd[cmd.index('--name')+1];started=False
    try:
        check_offset(c.get('core_offset_guard'))
        if available_gib()<c['min_host_available_gib']:raise RuntimeError('主机可用内存低于保护阈值')
        sp.run(cmd,check=True);started=True
        while True:
            state=inspect(name)['State'];(run/'state.json').write_text(json.dumps(state))
            if not state['Running']:break
            if available_gib()<c['min_host_available_gib']:
                (run/'memory-stop.txt').write_text('主机可用内存低于保护阈值，已请求停止。');stop(name);break
            time.sleep(5)
    except BaseException as exc:
        (run/'error.txt').write_text(str(exc))
        if started:stop(name)
        raise
    finally:
        if started:
            with (run/'server.log').open('wb') as f:sp.run(['docker','logs',name],stdout=f,stderr=sp.STDOUT)
            (run/'state.json').write_text(json.dumps(inspect(name)['State']))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('start');a.add_argument('--config',type=Path,default=ROOT/'config/local.json');a.add_argument('--name');a.add_argument('--dry-run',action='store_true')
    a=sub.add_parser('stop');a.add_argument('--name',required=True)
    a=sub.add_parser('_supervise');a.add_argument('run',type=Path)
    args=p.parse_args()
    if args.action=='stop':stop(args.name);return
    if args.action=='_supervise':supervise(args.run);return
    c=read_config(args.config);name=args.name or 'qwen-flash-sm80-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]+',name):raise ValueError('容器名称无效')
    run=ROOT/'runs'/name;cmd=build_command(c,name,run,gpu_inventory())
    if args.dry_run:print(json.dumps(cmd,ensure_ascii=False,indent=2));return
    for path in [Path(c['models_root'])/c['model_subdir'],Path(c['ple_artifact'])/'CURRENT',Path(c['cufile_config']),Path(c['ple_identity'])]:
        if not path.exists():raise FileNotFoundError(path)
    if c.get('validation_dir'):
        for rank in (0,1):
            if not (Path(c['validation_dir'])/f'draft-hidden-rank{rank}.pt').is_file():raise FileNotFoundError('缺少可选验证样本')
    check_offset(c.get('core_offset_guard'))
    if available_gib()<c['min_host_available_gib']:raise RuntimeError('主机可用内存不足')
    if sp.run(['docker','inspect',name],stdout=sp.DEVNULL,stderr=sp.DEVNULL).returncode==0:raise RuntimeError('同名容器已存在，请换名')
    run.mkdir(parents=True,exist_ok=False);(run/'service').mkdir();(run/'cache').mkdir()
    (run/'command.json').write_text(json.dumps(cmd,indent=2));(run/'config.json').write_text(json.dumps(c,indent=2))
    with (run/'supervisor.log').open('w') as log:
        proc=sp.Popen([sys.executable,str(Path(__file__).resolve()),'_supervise',str(run)],stdout=log,stderr=sp.STDOUT,start_new_session=True)
    print(json.dumps({'容器':name,'地址':f"http://127.0.0.1:{c['port']}",'日志':str(run),'监控进程':proc.pid,'状态':'启动已提交，请通过日志确认就绪'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
