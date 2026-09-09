"""模型下载前置检查的 CPU 回归测试，使用合成 safetensors 文件。"""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_model", ROOT / "scripts/check-model.py")
check_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_model)


class ModelCheckTests(unittest.TestCase):
    def checkpoint(self, root, mtp_dtype="BF16"):
        header = {}
        offset = 0
        for part in range(128):
            header[f"model.ple.ngram_embedding.shard_{part}.weight"] = {
                "dtype": "F8_E4M3", "shape": [1, 160], "data_offsets": [offset, offset + 160]}
            offset += 160
        header["model.ple.ngram_embedding.weight_scale"] = {
            "dtype": "BF16", "shape": [], "data_offsets": [offset, offset + 2]}
        offset += 2
        size = {"BF16": 2, "F32": 4}[mtp_dtype]
        header["mtp.fc_hidden.weight"] = {
            "dtype": mtp_dtype, "shape": [1, 1], "data_offsets": [offset, offset + size]}
        encoded = json.dumps(header).encode()
        shard = root / "model.safetensors"
        shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset + size))
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: shard.name for key in header}}))
        for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (root / filename).write_text("{}")
        return {"revision": "synthetic-test", "metadata_sha256": {
            f: hashlib.sha256((root / f).read_bytes()).hexdigest()
            for f in ("config.json", "model.safetensors.index.json")},
            "indexed_safetensors_files": 1, "indexed_safetensors_bytes": shard.stat().st_size,
            "ple_rows": 128, "ple_columns": 160, "ple_dtype": "F8_E4M3", "mtp_tensors": 1}

    def test_valid_headers_do_not_claim_payload_hash_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = check_model.inspect_checkpoint(root, self.checkpoint(root))
            self.assertEqual(result["ple_rows"], 128)
            self.assertFalse(result["weight_payload_sha256_verified"])

    def test_truncated_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.checkpoint(root)
            shard = root / "model.safetensors"
            with shard.open("r+b") as f:
                f.truncate(shard.stat().st_size - 1)
            with self.assertRaisesRegex(ValueError, "文件未下载完整"):
                check_model.inspect_checkpoint(root, source)

    def test_wrong_metadata_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.checkpoint(root)
            (root / "config.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "固定模型版本不一致"):
                check_model.inspect_checkpoint(root, source)

    def test_wrong_mtp_dtype_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "MTP 张量必须为 BF16"):
                check_model.inspect_checkpoint(root, self.checkpoint(root, "F32"))

    def test_malformed_tensor_header_is_rejected_with_tensor_name(self):
        cases = [
            ("data_offsets", [0.5, 160.5]),
            ("data_offsets", [False, 160]),
            ("data_offsets", [0]),
            ("shape", {"rows": 1, "columns": 160}),
            ("shape", [True, 160]),
            ("dtype", "UNKNOWN"),
            ("dtype", None),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = self.checkpoint(root)
                shard = root / "model.safetensors"
                data = shard.read_bytes()
                header_size = struct.unpack("<Q", data[:8])[0]
                header = json.loads(data[8:8 + header_size])
                name = "model.ple.ngram_embedding.shard_0.weight"
                header[name][field] = value
                encoded = json.dumps(header).encode()
                shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + header_size:])
                source["indexed_safetensors_bytes"] = shard.stat().st_size
                with self.assertRaisesRegex(ValueError, f"shard_0.weight.*{field}"):
                    check_model.inspect_checkpoint(root, source)


if __name__ == "__main__":
    unittest.main()
