#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Prepare pinned driver sources. Never installs or loads a kernel module."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(path, expected):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def prepare(args):
    manifest = json.loads((HERE / 'sources.json').read_text())
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    if not shutil.which('patch') or not shutil.which('curl'):
        raise ValueError('Install patch and curl first.')
    # Check before downloading or modifying any output tree.
    for name, digest in manifest['patch_sha256'].items():
        verify(HERE / 'patches' / name, digest)
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cmp-bar1-') as tmp:
        extracted = Path(tmp)
        for name, item in manifest['archives'].items():
            archive = cache / f'{name}-{item["sha256"][:16]}.tar.gz'
            if not archive.exists():
                partial = archive.with_suffix('.partial')
                subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                                '--output', str(partial), item['url']], check=True)
                verify(partial, item['sha256'])
                partial.rename(archive)
            verify(archive, item['sha256'])
            with tarfile.open(archive) as tf:
                tf.extractall(extracted, filter='data')
        source = extracted / manifest['archives']['nvidia']['root']
        upstream = extracted / manifest['archives']['cmpunlocker']['root']
        src_dst = source / 'src/nvidia/src/kernel/gpu/cmpunlock'
        inc_dst = source / 'src/nvidia/inc/kernel/gpu/cmpunlock'
        src_dst.mkdir(parents=True)
        inc_dst.mkdir(parents=True)
        shutil.copy2(upstream / 'driver/src/cmpunlock.c', src_dst)
        shutil.copy2(upstream / 'driver/src/cmpunlock.h', inc_dst)
        # Deliberately do not import the local clock/timing configuration.
        (inc_dst / 'cmpunlock_config.h').write_text(
            '#ifndef CMPUNLOCK_CONFIG_H\n#define CMPUNLOCK_CONFIG_H\n'
            '#define CMPUNLOCK_ENABLE_P2P 1\n#endif\n')
        with (source / 'src/nvidia/srcs.mk').open('a') as f:
            f.write('\nSRCS += src/kernel/gpu/cmpunlock/cmpunlock.c\n')
        selected = ['0011-p2p-bar1.patch', '0013-skip-mailbox-peer-preinit.patch']
        if args.allow_topology_override:
            selected.append('0015-bar1p2p-readcap-override.patch')
        patches = sorted((upstream / 'driver/patches').glob('*.patch'))
        patches += [HERE / 'patches' / name for name in selected]
        logs = []
        for patch in patches:
            result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=0',
                                     '-p1', '-i', str(patch)], cwd=source,
                                    capture_output=True, text=True)
            logs.append(f'{patch.name}\n{result.stdout}{result.stderr}')
            if result.returncode:
                raise ValueError('Patch failed:\n' + logs[-1])
        (source / 'cmp-bar1-patches.log').write_text('\n'.join(logs))
        (source / 'cmp-bar1-build.json').write_text(json.dumps({
            'driver_version': manifest['driver_version'],
            'upstream_commit': manifest['upstream_commit'],
            'patches': selected, 'clock_overrides': False,
            'topology_override': args.allow_topology_override,
            'status': 'prepared; not installed or runtime-validated',
        }, indent=2) + '\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output)
    print(f'Prepared: {output}\nNo modules installed or loaded.')
    print('Build: make -C "' + str(output) + '" -j1 modules SYSSRC="/lib/modules/$(uname -r)/build"')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cache', type=Path, default=Path.home() / '.cache/qwen-cmp-bar1')
    p.add_argument('--allow-topology-override', action='store_true',
                   help='Apply CMP read-capability override; this does not validate PCIe routing.')
    args = p.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
