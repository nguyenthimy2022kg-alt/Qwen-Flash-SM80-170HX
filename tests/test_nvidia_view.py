import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('view',ROOT/'scripts/prepare-nvidia-view.py')
view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view)

class NvidiaViewTests(unittest.TestCase):
    def source(self,p):
        p.mkdir();h={};offset=0
        for i in range(128):
            h[f'model.ngram_embedding.shard_{i}.weight']={'dtype':'F8_E4M3','shape':[1,1],'data_offsets':[offset,offset+1]};offset+=1
        for name in ['mtp.weight','model.ngram_embedding.weight_scale']:
            h[name]={'dtype':'BF16','shape':[1],'data_offsets':[offset,offset+2]};offset+=2
        raw=json.dumps(h).encode();(p/'mixed.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+bytes(range(offset)))
        (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'mixed.safetensors' for k in h}}))
        (p/'config.json').write_text('{}')
    def test_ple_excluded_and_retained_payloads_identical(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'source';self.source(src);before=(src/'mixed.safetensors').read_bytes()
            result=view.create_view(src,p/'target');self.assertTrue(result['verified'])
            index=json.loads((p/'target/model.safetensors.index.json').read_text())['weight_map']
            self.assertEqual(set(index),{'mtp.weight','model.ngram_embedding.weight_scale'})
            self.assertFalse((p/'target/mixed.safetensors').exists())
            self.assertEqual((src/'mixed.safetensors').read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'overwrite'):view.create_view(src,p/'target')
    def test_failed_copy_leaves_no_partial_view(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'source';self.source(src);f=src/'mixed.safetensors';f.write_bytes(f.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError,'Truncated'):view.create_view(src,p/'target')
            self.assertFalse((p/'target').exists());self.assertEqual(list(p.glob('.target-*')),[])
