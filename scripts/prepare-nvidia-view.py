#!/usr/bin/env python3
"""Create a GDS loading view, keeping PLE rows out of the mixed MTP shard."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

PLE_ROW = re.compile(r'ngram_embedding\.shard_\d+\.weight$')


def header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > 128 * 1024**2:
            raise ValueError('Safetensors header is unexpectedly large')
        return json.loads(f.read(n)), 8 + n


def copy_payload(source, output, offset, size):
    source.seek(offset)
    digest = hashlib.sha256()
    while size:
        data = source.read(min(size, 8 * 1024**2))
        if not data:
            raise ValueError('Truncated safetensors payload')
        digest.update(data)
        if output is not None:
            output.write(data)
        size -= len(data)
    return digest.hexdigest()


def create_view(source, target):
    source, target = Path(source).resolve(), Path(target).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(f'Refusing to overwrite existing view: {target}')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    for name in set(weights.values()):
        if Path(name).name != name:
            raise ValueError(f'Invalid shard filename: {name}')
    row_keys = [k for k in weights if PLE_ROW.search(k)]
    if len(row_keys) != 128:
        raise ValueError('Expected the complete original NVIDIA checkpoint with 128 PLE rows shards')
    mixed = {weights[k] for k in row_keys}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.' + target.name + '-', dir=target.parent))
    records = {}
    try:
        replacements = {}
        for number, name in enumerate(sorted(mixed)):
            data, start = header(source / name)
            keep = {k: v for k, v in data.items() if k != '__metadata__' and not PLE_ROW.search(k)}
            compact, offset = {'__metadata__': data.get('__metadata__', {'format': 'pt'})}, 0
            for key, value in keep.items():
                size = value['data_offsets'][1] - value['data_offsets'][0]
                compact[key] = dict(value, data_offsets=[offset, offset + size])
                offset += size
            encoded = json.dumps(compact, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            output_name = f'gds-retained-{number:02d}.safetensors'
            hashes = {}
            with (source / name).open('rb') as src, (temp / output_name).open('xb') as dst:
                dst.write(struct.pack('<Q', len(encoded)))
                dst.write(encoded)
                for key, value in keep.items():
                    begin, end = value['data_offsets']
                    hashes[key] = copy_payload(src, dst, start + begin, end - begin)
            with (temp / output_name).open('rb') as f:
                for key in keep:
                    begin, end = compact[key]['data_offsets']
                    if copy_payload(f, None, 8 + len(encoded) + begin, end - begin) != hashes[key]:
                        raise ValueError(f'Copied payload hash mismatch: {key}')
            records[output_name] = {'tensors': len(keep), 'bytes': offset, 'sha256': hashes}
            replacements[name] = output_name
        index['weight_map'] = {k: replacements.get(v, v) for k, v in weights.items() if not PLE_ROW.search(k)}
        index.get('metadata', {}).pop('total_size', None)
        (temp / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
        for path in source.iterdir():
            if not path.is_file() or path.name in mixed or path.name == 'model.safetensors.index.json':
                continue
            if path.suffix == '.safetensors':
                try:
                    os.link(path, temp / path.name)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copy2(path, temp / path.name)
            else:
                shutil.copy2(path, temp / path.name)
        (temp / 'GDS_VIEW_MANIFEST.json').write_text(json.dumps({'schema': 1, 'source_index_sha256': hashlib.sha256((source / 'model.safetensors.index.json').read_bytes()).hexdigest(), 'retained': records}, indent=2) + '\n')
        os.rename(temp, target)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return {'view': str(target), 'removed_ple_tensors': len(row_keys), 'retained_tensors': sum(r['tensors'] for r in records.values()), 'copied_bytes': sum(r['bytes'] for r in records.values()), 'verified': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(create_view(args.source, args.target), indent=2))


if __name__ == '__main__':
    main()
