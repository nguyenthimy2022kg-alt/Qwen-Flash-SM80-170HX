import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

host = load('gds_host', ROOT / 'scripts/check-gds-host.py')
prepare_path = ROOT / 'drivers/cmp-bar1/prepare.py'
prepare = load('cmp_prepare', prepare_path) if prepare_path.exists() else None

class HostInventoryTests(unittest.TestCase):
    def test_missing_tool_is_not_success(self):
        with patch.object(host.shutil, 'which', return_value=None):
            self.assertEqual(host.command(['nvidia-smi'])['status'], 'missing')

    def test_timeout_is_reported(self):
        with patch.object(host.shutil, 'which', return_value='/bin/tool'), patch.object(
            host.subprocess, 'run', side_effect=subprocess.TimeoutExpired('tool', 20)
        ):
            self.assertEqual(host.command(['tool'])['status'], 'error')

    def test_capability_success_never_becomes_acceptance(self):
        with patch.object(host, 'command', return_value={'status': 'ok', 'stdout': 'OK'}):
            result = host.inventory('/definitely-missing-gds-test-path')
        self.assertEqual(result['acceptance'], 'NOT_TESTED')
        self.assertFalse(result['data_path_exists'])
        self.assertEqual(result['checks']['data_mount']['status'], 'missing')

@unittest.skipIf(prepare is None, 'Optional host driver package is excluded from inference images')
class DriverPreparationTests(unittest.TestCase):
    def test_modified_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'archive'
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                prepare.verify(p, '0' * 64)

    def test_packaged_patch_hashes(self):
        manifest = json.loads((prepare.HERE / 'sources.json').read_text())
        for name, digest in manifest['patch_sha256'].items():
            prepare.verify(prepare.HERE / 'patches' / name, digest)

    def test_existing_output_is_never_overwritten(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=Path(tmp), cache=Path(tmp) / 'cache', allow_topology_override=False)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare.prepare(args)
            self.assertFalse(args.cache.exists())
