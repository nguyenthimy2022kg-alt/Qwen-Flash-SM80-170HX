#!/usr/bin/env python3
"""为已有 FP8 PLE 检查点生成转换映射，或登记转换后的数据身份。"""
import argparse,json,re,sys,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ple_gds.manifest import _parse_safetensors_header
from ple_gds.compact import load_current_compact

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def mapping(index_path):
    weights=json.loads(index_path.read_text())['weight_map'];parts=[];scales=[];headers={}
    for key,filename in weights.items():
        match=re.search(r'ngram_embedding\.shard_(\d+)\.weight$',key)
        if match:parts.append((int(match[1]),key,filename))
        if key.endswith('ngram_embedding.weight_scale'):scales.append(key)
    if [x[0] for x in sorted(parts)]!=list(range(128)) or len(scales)!=1:
        raise ValueError('需要当前模型 128 个 FP8 PLE 分片及一个全局 scale')
    shards=[];rows=0
    for _,key,filename in sorted(parts):
        path=index_path.parent/filename
        if filename not in headers:headers[filename]=_parse_safetensors_header(path)[0]
        entry=headers[filename][key]
        if entry['dtype']!='F8_E4M3' or len(entry['shape'])!=2 or entry['shape'][1]!=160:
            raise ValueError('需要 F8_E4M3、160 列的 PLE；本脚本不转换量化格式')
        shards.append({'tensor':key,'row_start':rows});rows+=entry['shape'][0]
    if rows!=320001536:raise ValueError('PLE 总行数与当前模型不符')
    filename=weights[scales[0]]
    if filename not in headers:headers[filename]=_parse_safetensors_header(index_path.parent/filename)[0]
    scale=headers[filename][scales[0]]
    if scale['dtype']!='BF16' or math.prod(scale['shape'])!=1:
        raise ValueError('PLE 全局 scale 必须为单个 BF16 值')
    return {'row_count':rows,'components':[{'name':'ngram_embedding','dtype':'F8_E4M3','row_bytes':160,'shards':shards},{'name':'weight_scale','global':True,'dtype':'BF16','shards':[{'tensor':scales[0]}]}]}

def enroll(artifact,verify_data):
    m,g=load_current_compact(artifact,check_sources=False,check_data=verify_data)
    return {'generation':g.name,'current_sha256':sha(artifact/'CURRENT'),'metadata_sha256':sha(g/'ple-gds-metadata.json'),'data_sha256':m['data_sha256'],'source_identity_sha256':m['source_identity_sha256'],'data_bytes_verified':verify_data}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('mapping');a.add_argument('--index',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('enroll');a.add_argument('--artifact',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--metadata-only',action='store_true',help='只对已完成完整校验的数据使用，跳过大文件重复读取')
    args=p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        p.error(f'输出已存在，未读取或修改数据；请指定新文件：{args.output}')
    if not args.output.parent.is_dir():p.error(f'输出目录不存在：{args.output.parent}')
    value=mapping(args.index) if args.action=='mapping' else enroll(args.artifact,not args.metadata_only)
    with args.output.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print('已生成',args.output)

if __name__=='__main__':main()
