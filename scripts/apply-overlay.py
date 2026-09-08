#!/usr/bin/env python3
"""校验固定上游版本，然后安装增量源码；不加载模型。"""
from pathlib import Path
import argparse, hashlib, json, shutil

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'patches/upstream-sha256.json').read_text())
    errors=[]
    for rel,digest in expected.items():
        target=args.target/rel
        if digest is None:
            if target.exists():errors.append(rel+': expected a new file')
        elif not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            errors.append(rel+': upstream content mismatch')
    if errors:raise SystemExit('\n'.join(errors))
    shutil.copytree(root/'src',args.target,dirs_exist_ok=True)
    print(f'上游校验通过，已安装 {len(expected)} 个增量 Python 文件。')

if __name__=='__main__':main()
