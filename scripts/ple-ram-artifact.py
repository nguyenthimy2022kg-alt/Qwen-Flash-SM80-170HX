#!/usr/bin/env python3
"""Validate/copy only an enrolled PLE generation with bounded page-cache use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

CHUNK = 8 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        while data := f.read(CHUNK):
            h.update(data)
            os.posix_fadvise(f.fileno(), f.tell() - len(data), len(data), os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def layout(artifact, identity):
    ident = json.loads(identity.read_text())
    generation = ident['generation']
    if not re.fullmatch(r'gen-[0-9a-f]{32}', generation):
        raise ValueError('invalid generation name')
    current = artifact / 'CURRENT'
    gen = artifact / 'generations' / generation
    metadata = gen / 'ple-gds-metadata.json'
    for path in (current, metadata):
        if not path.resolve().is_relative_to(artifact.resolve()):
            raise ValueError('artifact file escapes the source directory')
    if current.read_text().strip() != generation or digest(current) != ident['current_sha256']:
        raise ValueError('CURRENT does not match the enrolled identity')
    if digest(metadata) != ident['metadata_sha256']:
        raise ValueError('metadata does not match the enrolled identity')
    meta = json.loads(metadata.read_text())
    name = meta['data_file']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid payload filename')
    data = gen / name
    if not data.resolve().is_relative_to(artifact.resolve()):
        raise ValueError('payload escapes the source directory')
    if data.stat().st_size != meta['data_size'] or meta['data_size'] <= 0:
        raise ValueError('payload size mismatch')
    if meta['data_sha256'] != ident['data_sha256']:
        raise ValueError('payload digest disagrees with enrolled identity')
    return ident, meta, data


def verify(artifact, identity):
    ident, meta, data = layout(artifact, identity)
    if digest(data) != ident['data_sha256']:
        raise ValueError('payload checksum mismatch')
    return meta['data_size']


def copy(artifact, identity, destination):
    if destination.resolve().is_relative_to(artifact.resolve()):
        raise ValueError('destination must be outside the source artifact')
    ident, meta, source = layout(artifact, identity)
    destination.mkdir(exist_ok=False)
    try:
        gen = destination / 'generations' / ident['generation']
        gen.mkdir(parents=True)
        target = gen / meta['data_file']
        h = hashlib.sha256()
        with source.open('rb', buffering=0) as src, target.open('xb', buffering=0) as dst:
            while data := src.read(CHUNK):
                h.update(data)
                view = memoryview(data)
                while view:
                    n = dst.write(view)
                    if not n:
                        raise OSError('short payload write')
                    view = view[n:]
                # brd already occupies RAM; bound the additional ext4 page cache.
                os.fdatasync(dst.fileno())
                os.posix_fadvise(dst.fileno(), dst.tell() - len(data), len(data), os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(src.fileno(), src.tell() - len(data), len(data), os.POSIX_FADV_DONTNEED)
        if target.stat().st_size != meta['data_size'] or h.hexdigest() != ident['data_sha256']:
            raise ValueError('source payload failed its enrolled checksum')
        shutil.copyfile(source.parent / 'ple-gds-metadata.json', gen / 'ple-gds-metadata.json')
        shutil.copyfile(artifact / 'CURRENT', destination / 'CURRENT')
        verify(destination, identity)
        for f in (target, gen / 'ple-gds-metadata.json', destination / 'CURRENT'):
            with f.open('rb') as handle:
                os.fsync(handle.fileno())
            f.chmod(0o444)
    except BaseException:
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('size', 'verify', 'copy'))
    parser.add_argument('artifact', type=Path)
    parser.add_argument('identity', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    if args.action == 'size':
        print(layout(args.artifact, args.identity)[1]['data_size'])
    elif args.action == 'verify':
        print(verify(args.artifact, args.identity))
    else:
        if args.destination is None:
            parser.error('copy requires destination')
        copy(args.artifact, args.identity, args.destination)


if __name__ == '__main__':
    main()
