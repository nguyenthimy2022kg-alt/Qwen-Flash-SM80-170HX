"""启动配置与 Docker 前置检查；不调用真实 Docker、NVML 或 GPU。"""
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_config", ROOT / "scripts/serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class ServeTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "config/example.json").read_text())

    def test_invalid_config_fails_before_gpu_or_docker_access(self):
        cases = {
            "port": [None, "18420", 18420.5, True],
            "gpu_ids": ["01", None, [True, 1], [{}, 1], [0, "0"]],
            "container_memory_gib": ["21", True, 0, float("inf")],
            "container_memory_and_swap_gib": [None, 20],
            "min_host_available_gib": [float("nan"), float("inf"), False, -1],
            "draft_int8": [None, "false", 1],
            "validation_dir": [False, 4, ""],
            "core_offset_guard": [False, {}, {"gpu_uuid": "GPU-a", "expected": True}],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            for field, values in cases.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        config = self.config()
                        config[field] = value
                        path.write_text(json.dumps(config))
                        with patch.object(serve, "gpu_inventory") as inventory, patch.object(serve.sp, "run") as docker:
                            with patch.object(sys, "argv", ["serve.py", "start", "--config", str(path)]):
                                with self.assertRaisesRegex(ValueError, field):
                                    serve.main()
                            inventory.assert_not_called()
                            docker.assert_not_called()

    def test_missing_required_fields_are_named(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            for field in ("port", "mode", "gpu_ids", "container_memory_gib", "draft_int8"):
                with self.subTest(field=field):
                    config = self.config()
                    del config[field]
                    path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, field):
                        serve.read_config(path)

    def test_cli_missing_config_has_recovery_hint_without_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/serve.py"), "start", "--config", str(Path(td) / "missing.json")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("config/example.json", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_docker_preflight_stops_at_unavailable_dependency(self):
        config = self.config()
        with patch.object(serve.sp, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Docker 命令"):
                serve.check_docker(config, "test")
        failure = subprocess.CompletedProcess([], 1, "", "unavailable")
        success = subprocess.CompletedProcess([], 0, "ok", "")
        with patch.object(serve.sp, "run", return_value=failure) as run:
            with self.assertRaisesRegex(RuntimeError, "Docker 服务"):
                serve.check_docker(config, "test")
            self.assertEqual(run.call_count, 1)
        with patch.object(serve.sp, "run", side_effect=[success, failure]) as run:
            with self.assertRaisesRegex(RuntimeError, "本地镜像"):
                serve.check_docker(config, "test")
            self.assertEqual(run.call_count, 2)

    def test_preflight_checks_local_image_without_pulling(self):
        success = subprocess.CompletedProcess([], 0, "ok", "")
        missing = subprocess.CompletedProcess([], 1, "", "No such container")
        with patch.object(serve.sp, "run", side_effect=[success, success, missing]) as run:
            serve.check_docker(self.config(), "test")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[1][-1], self.config()["image"])
            self.assertTrue(all("pull" not in cmd and "run" not in cmd for cmd in commands))
        with patch.object(serve.sp, "run", return_value=success):
            with self.assertRaisesRegex(RuntimeError, "同名容器"):
                serve.check_docker(self.config(), "test")

    def test_failed_preflight_does_not_submit_background_start(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            devices = {"0": {"uuid": "GPU-a", "bdf": "0000:01:00.0"},
                       "1": {"uuid": "GPU-b", "bdf": "0000:02:00.0"}}
            with patch.object(serve, "ROOT", root), patch.object(serve, "read_config", return_value=self.config()), \
                    patch.object(serve, "check_paths"), patch.object(serve, "gpu_inventory", return_value=devices), \
                    patch.object(serve, "build_command", return_value=[]), patch.object(serve, "available_gib", return_value=32), \
                    patch.object(serve, "check_docker", side_effect=RuntimeError("missing image")), patch.object(serve.sp, "Popen") as start:
                with patch.object(sys, "argv", ["serve.py", "start", "--name", "test"]):
                    with self.assertRaisesRegex(RuntimeError, "missing image"):
                        serve.main()
                start.assert_not_called()
                self.assertFalse((root / "runs").exists())

    def test_dry_run_accepts_one_character_name_without_docker(self):
        config = serve.read_config(ROOT / "config/example.json")
        devices = {"0": {"uuid": "GPU-a", "bdf": "0000:01:00.0"},
                   "1": {"uuid": "GPU-b", "bdf": "0000:02:00.0"}}
        with patch.object(serve, "read_config", return_value=config), patch.object(serve, "gpu_inventory", return_value=devices), \
                patch.object(serve, "check_paths") as paths, patch.object(serve, "check_docker") as docker:
            with patch.object(sys, "argv", ["serve.py", "start", "--name", "a", "--dry-run"]), contextlib.redirect_stdout(io.StringIO()):
                serve.main()
            paths.assert_not_called()
            docker.assert_not_called()

    def test_stop_refuses_unlabeled_container_and_uses_verified_id(self):
        for labels in (None, {}, {serve.LABEL: "0"}):
            with self.subTest(labels=labels), patch.object(serve, "inspect", return_value={"Config": {"Labels": labels}}), \
                    patch.object(serve.sp, "run") as stop:
                with self.assertRaisesRegex(RuntimeError, "拒绝停止"):
                    serve.stop("test")
                stop.assert_not_called()
        info = {"Config": {"Labels": {serve.LABEL: "1"}}, "Id": "container-id"}
        with patch.object(serve, "inspect", return_value=info), patch.object(serve.sp, "run") as stop:
            serve.stop("test")
            self.assertEqual(stop.call_args.args[0][-1], "container-id")


if __name__ == "__main__":
    unittest.main()
