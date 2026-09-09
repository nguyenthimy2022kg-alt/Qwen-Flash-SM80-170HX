#!/usr/bin/env python3
"""Read-only P2P/GDS inventory; does not run transfers or declare GDS working."""
import argparse
import glob
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(argv):
    if not shutil.which(argv[0]):
        return {'status': 'missing', 'command': argv}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {'status': 'ok' if p.returncode == 0 else 'error',
                'command': argv, 'returncode': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'error', 'command': argv, 'detail': str(exc)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def inventory(data_path=None):
    result = {'schema': 1, 'acceptance': 'NOT_TESTED',
              'meaning': 'Inventory only. P2P capability OK does not prove data transfer or strict GDS.',
              'kernel': platform.release(),
              'kernel_cmdline': read('/proc/cmdline'),
              'iommu_groups': len(glob.glob('/sys/kernel/iommu_groups/[0-9]*')),
              'nvme_multipath': read('/sys/module/nvme_core/parameters/multipath'),
              'nvidia_params': read('/proc/driver/nvidia/params'),
              'force_compat_env': os.environ.get('CUFILE_FORCE_COMPAT_MODE'),
              'pci_devices': []}
    for dev in sorted(Path('/sys/bus/pci/devices').glob('*')):
        vendor, cls = read(dev / 'vendor'), read(dev / 'class') or ''
        if vendor != '0x10de' and not cls.startswith('0x0108'):
            continue
        result['pci_devices'].append({
            'bdf': dev.name, 'vendor': vendor, 'device': read(dev / 'device'),
            'class': cls, 'sysfs_path': str(dev.resolve()),
            'numa_node': read(dev / 'numa_node'),
            'link_speed': read(dev / 'current_link_speed'),
            'link_width': read(dev / 'current_link_width'),
            'resources': read(dev / 'resource')})
    result['checks'] = {name: command(argv) for name, argv in {
        'gpu': ['nvidia-smi', '--query-gpu=index,name,pci.bus_id,driver_version,memory.total', '--format=csv'],
        'bar1': ['nvidia-smi', '-q', '-d', 'MEMORY'],
        'topology': ['nvidia-smi', 'topo', '-m'],
        'p2p_read': ['nvidia-smi', 'topo', '-p2p', 'r'],
        'p2p_write': ['nvidia-smi', 'topo', '-p2p', 'w'],
        'pci_tree': ['lspci', '-t'],
        'cufile_libraries': ['ldconfig', '-p'],
    }.items()}
    libs = result['checks']['cufile_libraries']
    if libs.get('stdout'):
        libs['stdout'] = '\n'.join(l for l in libs['stdout'].splitlines() if 'cufile' in l)
    gds = {Path(p).resolve() for pattern in ['/usr/local/cuda*/gds/tools/gdscheck.py',
          '/usr/local/cuda*/gds/tools/gdsio'] for p in glob.glob(pattern)}
    result['gds_tools'] = sorted(map(str, gds))
    conf = os.environ.get('CUFILE_ENV_PATH_JSON', '/etc/cufile.json')
    result['cufile_config_path'] = conf
    try:
        config = json.loads(Path(conf).read_text())
        result['cufile_config'] = {k: config[k] for k in ('properties', 'fs', 'logging') if k in config}
    except (OSError, ValueError) as exc:
        result['cufile_config_error'] = str(exc)
    if data_path is not None:
        target = Path(data_path).expanduser().resolve()
        result['data_path'] = str(target)
        result['data_path_exists'] = target.exists()
        if target.exists():
            result['checks']['data_mount'] = command(['findmnt', '-T', str(target), '-o', 'SOURCE,TARGET,FSTYPE,OPTIONS'])
        else:
            result['checks']['data_mount'] = {'status': 'missing', 'detail': 'Data path does not exist.'}
    result['next_steps'] = [
        'Run actual CUDA peer-copy data validation in BOTH directions.',
        'Run strict cuFile NVMe read/data validation on the actual PLE filesystem and reading GPU.',
        'Check completed I/O counts and P2PDMA logs; reject CPU compatibility fallback.',
        'See docs/CMP_P2P.md and docs/GDS_NVME_P2PDMA_REPRODUCTION.md (English versions available).']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', help='Existing PLE directory or file; contents are not read.')
    args = p.parse_args()
    print(json.dumps(inventory(args.data_path), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
