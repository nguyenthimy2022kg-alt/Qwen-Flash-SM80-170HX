import importlib.util
import subprocess
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
