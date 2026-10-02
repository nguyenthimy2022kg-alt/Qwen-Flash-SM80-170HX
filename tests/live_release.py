#!/usr/bin/env python3
"""Serial release smoke/regression checks against an already running server.

No GPU settings are changed. Creates synthetic conversations only. The output
records response content, timings and finish reasons; it is not a quality suite.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',default='http://127.0.0.1:18420')
    ap.add_argument('--model',default='qwen3.8-flash-next')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--long-context',action='store_true')
    ap.add_argument('--soak-rounds',type=int,default=0,help='Run only a growing multi-turn conversation for this many rounds')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    def post(path,body):
        req=urllib.request.Request(args.base_url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        return urllib.request.urlopen(req,timeout=600)
    def tokens(messages):
        with post('/tokenize',dict(model=args.model,messages=messages,chat_template_kwargs={'enable_thinking':False})) as f:
            return json.load(f)['count']
    def run(name,messages,cap=512,extra=None,cancel=False):
        body=dict(model=args.model,messages=messages,temperature=0,top_p=1,top_k=-1,seed=42,max_tokens=cap,
                  chat_template_kwargs={'enable_thinking':False},stream=True,stream_options={'include_usage':True})
        body.update(extra or {});content='';reason='';calls={};done=False;finish=None;usage=None;start=time.monotonic();first=last=None;events=0
        with post('/v1/chat/completions',body) as f:
            for line in f:
                if not line.startswith(b'data:'):continue
                raw=line[5:].strip()
                if raw==b'[DONE]':done=True;break
                data=json.loads(raw)
                if data.get('error'):raise RuntimeError(data['error'])
                usage=data.get('usage') or usage
                for ch in data.get('choices',[]):
                    d=ch.get('delta',{});c=d.get('content') or '';r=d.get('reasoning') or d.get('reasoning_content') or ''
                    if c or r:first=first or time.monotonic();last=time.monotonic();events+=1
                    content+=c;reason+=r;finish=ch.get('finish_reason') or finish
                    for call in d.get('tool_calls') or []:
                        dst=calls.setdefault(call['index'],dict(id='',type='function',function=dict(name='',arguments='')))
                        dst['id']=call.get('id') or dst['id'];fn=call.get('function') or {}
                        dst['function']['name']+=fn.get('name') or '';dst['function']['arguments']+=fn.get('arguments') or ''
                if cancel and events>=16:break
        result=dict(name=name,done=done,finish=finish,content=content,reasoning=reason,tool_calls=list(calls.values()),
                    seconds=time.monotonic()-start,ttft_s=first-start if first else None,output_s=last-first if first else None,usage=usage)
        (args.output/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        if not cancel:assert done and finish in ('stop','length','tool_calls'),result
        if not body.get('chat_template_kwargs',{}).get('enable_thinking',True):assert not reason,name
        rows.append({k:v for k,v in result.items() if k not in ('content','reasoning','tool_calls')})
        (args.output/'summary.json').write_text(json.dumps(rows,indent=2))
        print(name,finish,'complete',flush=True)
        return result
    def user(text):return dict(role='user',content=text)
    if args.soak_rounds:
        history=[]
        tasks=['用Python实现一个内存记账类，支持新增和余额，不超过70行代码。',
               '在上一版的基础上增加按分类汇总，只给需要替换的代码，不超过50行。',
               '为当前版本写出3个单元测试，保持简短。',
               '检查刚才的实现有哪些边界问题，用不超过200字回答。',
               '给出本轮改动的简短JSON说明，包含changes和tests两个数组。']
        for i in range(args.soak_rounds):
            text=tasks[i%len(tasks)]
            if i and i%len(tasks)==0:text='重新写一个独立版本。'+text
            history.append(user(text))
            res=run(f'soak-{i+1:02d}',history,1200,{'temperature':.6,'top_p':.95,'top_k':20})
            assert res['content'] and res['finish'] in ('stop','length')
            history.append(dict(role='assistant',content=res['content']))
        assert urllib.request.urlopen(args.base_url+'/health',timeout=10).status==200
        (args.output/'PASS').write_text(f'{len(rows)} growing-history requests passed\n')
        print('SOAK PASS',len(rows),flush=True)
        return
    history=[]
    prompts=['写一个单文件网页，三张可筛选的旅行卡片，中文界面，原生HTML/CSS/JS，无外部依赖，代码不超过80行。',
             '加入收藏功能，只输出完整替换的脚本，保持精简。',
             '让收藏在刷新后仍能保存。只给修改部分。',
             '检查空值处理是否正确，给出两个测试步骤。',
             '用JSON数组写出刚才的两个测试步骤，仅输出JSON。',
             '用三句话总结这个网页。']
    for i,prompt in enumerate(prompts):
        history.append(user(prompt));res=run(f'history-{i+1}',history,2048 if i==0 else 768)
        assert res['content'];history.append(dict(role='assistant',content=res['content']))
    tool=dict(type='function',function=dict(name='get_weather',description='查询天气',parameters=dict(type='object',properties={'city':{'type':'string'}},required=['city'])))
    h=[user('必须调用工具查询上海天气。')]
    res=run('tool-call',h,256,{'tools':[tool],'tool_choice':'required'});assert res['finish']=='tool_calls' and len(res['tool_calls'])==1
    call=res['tool_calls'][0];assert call['function']['name']=='get_weather';json.loads(call['function']['arguments'])
    h.extend([dict(role='assistant',content=res['content'] or None,tool_calls=res['tool_calls']),dict(role='tool',tool_call_id=call['id'],content='{"city":"上海","weather":"晴","temperature_c":26}')])
    assert '26' in run('tool-followup',h,128,{'tools':[tool],'tool_choice':'none'})['content']
    run('cancel',[user('写一个完整Python记账程序，包含保存、统计和命令行功能。')],2048,cancel=True)
    time.sleep(2)
    assert run('after-cancel',[user('15加27等于多少？只输出数字。')],32)['content'].strip()=='42'
    run('sampling',[user('列出整理房间的五个简短建议。')],256,{'temperature':.6,'top_p':.95,'top_k':20})
    res=run('json',[user('输出JSON对象，steps是3个字符串构成的数组。')],256,{'response_format':{'type':'json_object'}})
    assert len(json.loads(res['content'])['steps'])==3
    # Repeated known text exercises long history proposals and their verification.
    block='\n'.join(f'item_{i:03d} = "color_{i%7}"' for i in range(80))
    res=run('history-copy',[user('请原样复制下面80行，只给代码，不要解释：\n'+block)],1800)
    assert all(line in res['content'] for line in block.splitlines())
    assert run('after-copy',[user('17乘以19，只给数字。')],32)['content'].strip()=='323'
    # Two synthetic local images; no outside network or private photos.
    def png(rgb):
        def chunk(name,data):return struct.pack('!I',len(data))+name+data+struct.pack('!I',zlib.crc32(name+data)&0xffffffff)
        raw=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes(rgb)*64)*64))+chunk(b'IEND',b'')
        return 'data:image/png;base64,'+base64.b64encode(raw).decode()
    images=[{'type':'text','text':'按图片顺序说出两张图片的主要颜色，只回答颜色名称。'}]+[{'type':'image_url','image_url':{'url':png(c)}} for c in [(255,0,0),(0,0,255)]]
    res=run('images-two',[user(images)],128);assert '红' in res['content'] and '蓝' in res['content']
    res=run('thinking',[user('15加27等于多少？')],1536,{'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True,'reasoning_effort':'xhigh'},'temperature':1,'top_p':.95,'top_k':20})
    assert res['reasoning'] and '42' in res['content'] and res['finish']=='stop'
    if args.long_context:
        for target in (8192,32768,131072,260000):
            def message(n):
                lines='\n'.join(f'记录{i:05d}：项目状态正常，颜色为蓝色，负责人是小林，预算等待复核。' for i in range(n))
                return [user(lines+'\n最终核对码：73921。只回答最终核对码。')]
            lo,hi=1,14000
            while lo<hi:
                mid=(lo+hi+1)//2
                if tokens(message(mid))<=target:lo=mid
                else:hi=mid-1
            h=message(lo);count=tokens(h)
            (args.output/f'context-{target}-input.json').write_text(json.dumps({'tokens':count,'target':target,'records':lo}))
            for i,prompt in enumerate([None,'负责人叫什么？只回答名字。','最终核对码是多少？只回答数字。']):
                if prompt:h.append(user(prompt))
                res=run(f'context-{target}-turn-{i+1}',h,128)
                assert ('小林' if i==1 else '73921') in res['content']
                h.append(dict(role='assistant',content=res['content']))
    assert urllib.request.urlopen(args.base_url+'/health',timeout=10).status==200
    (args.output/'PASS').write_text(f'{len(rows)} serial requests passed\n')
    print('RELEASE LIVE PASS',len(rows),flush=True)

if __name__=='__main__':main()
