"""RAM artifact integrity and non-destructive setup guards; no root/GPU needed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ple_ram_artifact', ROOT / 'scripts/ple-ram-artifact.py')
ram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ram)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RamFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'
        self.name = 'gen-' + 'a' * 32
        self.gen = self.src / 'generations' / self.name
        self.gen.mkdir(parents=True)
        self.data = bytes(range(256)) * 64
        self.payload = self.gen / 'ple-gds-data.bin'
        self.payload.write_bytes(self.data)
        meta = dict(data_file=self.payload.name, data_size=len(self.data), data_sha256=sha(self.data))
        metadata = json.dumps(meta).encode()
        (self.gen / 'ple-gds-metadata.json').write_bytes(metadata)
        current = (self.name + '\n').encode()
        (self.src / 'CURRENT').write_bytes(current)
        self.identity = self.root / 'identity.json'
        self.identity.write_text(json.dumps(dict(generation=self.name, current_sha256=sha(current),
                                                metadata_sha256=sha(metadata), data_sha256=sha(self.data))))
        self.dst = self.root / 'copied'


class RamArtifactTests(RamFixture):
    def test_copy_verifies_bytes_and_only_active_generation(self):
        (self.src / 'generations' / 'unused').mkdir()
        (self.src / 'generations' / 'unused' / 'unrelated').write_text('keep on SSD')
        ram.copy(self.src, self.identity, self.dst)
        self.assertEqual(ram.verify(self.dst, self.identity), len(self.data))
        self.assertEqual([p.name for p in (self.dst / 'generations').iterdir()], [self.name])
        self.assertEqual((self.dst / 'generations' / self.name / self.payload.name).stat().st_mode & 0o777, 0o444)

    def test_same_size_corruption_rejected_and_new_destination_removed(self):
        self.payload.write_bytes(b'x' * len(self.data))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ram.copy(self.src, self.identity, self.dst)
        self.assertFalse(self.dst.exists())
        self.assertTrue(self.src.exists())

    def test_existing_destination_preserved(self):
        self.dst.mkdir()
        (self.dst / 'sentinel').write_text('important')
        with self.assertRaises(FileExistsError):
            ram.copy(self.src, self.identity, self.dst)
        self.assertEqual((self.dst / 'sentinel').read_text(), 'important')

    def test_wrong_metadata_identity_rejected(self):
        with (self.gen / 'ple-gds-metadata.json').open('a') as f:
            f.write(' ')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            ram.copy(self.src, self.identity, self.dst)
        self.assertFalse(self.dst.exists())

    def test_payload_symlink_outside_source_rejected(self):
        outside = self.root / 'outside'
        self.payload.rename(outside)
        self.payload.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            ram.verify(self.src, self.identity)

    def test_truncated_payload_rejected(self):
        self.payload.write_bytes(self.data[:-1])
        with self.assertRaisesRegex(ValueError, 'size'):
            ram.verify(self.src, self.identity)

    def test_nested_destination_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            ram.copy(self.src, self.identity, self.src / 'copy')

    def test_copy_io_failure_removes_only_new_destination(self):
        from unittest.mock import patch
        with patch.object(ram.os, 'fdatasync', side_effect=OSError('write failed')):
            with self.assertRaisesRegex(OSError, 'write failed'):
                ram.copy(self.src, self.identity, self.dst)
        self.assertFalse(self.dst.exists())
        self.assertEqual(self.payload.read_bytes(), self.data)


class RamSetupGuards(RamFixture):
    def run_setup(self, *, existing=False, owned=False, options='ro', corrupt=False, nonempty=False):
        sandbox = self.root / 'sandbox'
        sandbox.mkdir()
        mount = sandbox / 'mount'
        mount.mkdir()
        dev = sandbox / 'ram0'
        module = sandbox / 'brd'
        owner = sandbox / 'owner'
        meminfo = sandbox / 'meminfo'
        meminfo.write_text('MemAvailable: 1024 kB\n')
        boot = sandbox / 'boot'
        boot.write_text('test-boot\n')
        script = (ROOT / 'scripts/ple-ram-setup.sh').read_text()
        script = '\n'.join(line for line in script.splitlines() if not line.startswith('[[ $EUID')) + '\n'
        for old, new in {
            '/sys/module/brd': module, '/dev/ram0': dev,
            '/run/qwen-ple-ram.owner': owner, '/run/qwen-ple-ram.lock': sandbox / 'lock',
            '/proc/meminfo': meminfo, '/proc/sys/kernel/random/boot_id': boot,
            '/etc/udev/rules.d/61-ple-ram.rules': sandbox / 'rule',
        }.items():
            script = script.replace(old, str(new))
        (sandbox / 'setup.sh').write_text(script)
        shutil.copyfile(ROOT / 'scripts/ple-ram-artifact.py', sandbox / 'ple-ram-artifact.py')
        if existing:
            module.mkdir()
            dev.touch()
        if owned:
            owner.write_text(f'test-boot\n{dev}\n{mount}\ntest-uuid\n{self.src}\n')
            ram.copy(self.src, self.identity, mount / self.src.name)
            if corrupt:
                payload = mount / self.src.name / 'generations' / self.name / self.payload.name
                payload.chmod(0o644)
                payload.write_bytes(b'x' * len(self.data))
        if nonempty:
            (mount / 'important').write_text('keep')
        bin_dir = sandbox / 'bin'
        bin_dir.mkdir()
        commands = {'blkid': "printf '%s\\n' test-uuid", 'mountpoint': 'exit 1',
                    'findmnt': f'case "${{*: -1}}" in SOURCE) echo "{dev}";; OPTIONS) echo "{options}";; esac'}
        # Every host-mutating command is replaced with a failing sentinel.
        for name in ('modprobe', 'mkfs.ext4', 'mount', 'udevadm', 'sync'):
            commands[name] = f'touch "{sandbox / "mutation"}"; exit 99'
        for name, body in commands.items():
            f = bin_dir / name
            f.write_text('#!/bin/bash\n' + body + '\n')
            f.chmod(0o755)
        env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], PLE_SOURCE=str(self.src),
                   PLE_IDENTITY=str(self.identity), PLE_RAM_MOUNT=str(mount))
        result = subprocess.run(['bash', str(sandbox / 'setup.sh')], env=env, capture_output=True, text=True)
        self.assertFalse((sandbox / 'mutation').exists(), result.stdout + result.stderr)
        return result

    def test_unowned_device_never_formatted(self):
        result = self.run_setup(existing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not owned', result.stderr)

    def test_low_memory_rejected_before_device_creation(self):
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Not enough free RAM', result.stderr)

    def test_nonempty_mountpoint_preserved(self):
        result = self.run_setup(nonempty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonempty', result.stderr)
        self.assertEqual((self.root / 'sandbox/mount/important').read_text(), 'keep')

    def test_valid_owned_readonly_setup_is_reused(self):
        result = self.run_setup(existing=True, owned=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Verified existing', result.stdout)

    def test_owned_readwrite_mount_rejected(self):
        result = self.run_setup(existing=True, owned=True, options='rw')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('read-only', result.stderr)

    def test_corrupt_owned_payload_rejected_without_format(self):
        result = self.run_setup(existing=True, owned=True, corrupt=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
