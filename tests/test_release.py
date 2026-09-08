"""发布接口的 CPU 检查，不加载模型、不调用 GPU。"""
import importlib.util,json,tempfile,unittest,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_serve',ROOT/'scripts/serve.py');serve=importlib.util.module_from_spec(spec);spec.loader.exec_module(serve)
spec=importlib.util.spec_from_file_location('prepare_ple',ROOT/'scripts/prepare-ple.py');prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
from ple_gds.compact import convert_compact,load_current_compact

class ReleaseTests(unittest.TestCase):
    def config(self):return serve.read_config(ROOT/'config/example.json')
    def devices(self):return {'0':{'uuid':'GPU-test-a','bdf':'0000:01:00.0'},'1':{'uuid':'GPU-test-b','bdf':'0000:02:00.0'}}
    def test_launch_keeps_pinned_model_settings(self):
        c=self.config();cmd=serve.build_command(c,'test',Path('/tmp/release test'),self.devices())
        self.assertNotIn('/:/host-root:ro',cmd);self.assertIn('GPU-test-a,GPU-test-b',next(x.split('=',1)[1] for x in cmd if x.startswith('CUDA_VISIBLE_DEVICES=')))
        self.assertIn('--enable-expert-parallel',cmd)
        self.assertEqual(cmd[cmd.index('--tensor-parallel-size')+1],'2')
        self.assertEqual(cmd[cmd.index('--pipeline-parallel-size')+1],'1')
        self.assertEqual(json.loads(cmd[cmd.index('--speculative-config')+1])['num_speculative_tokens'],6)
        self.assertNotIn('Q38_DRAFT_VALIDATION_DIR=/validation',cmd)
    def test_fallback_disables_both_draft_overrides(self):
        c=self.config();c.update(draft_int8=False,mode='tp2')
        cmd=serve.build_command(c,'test',Path('/tmp/run'),self.devices())
        self.assertIn('Q38_DRAFT_INT8=0',cmd);self.assertNotIn('--enable-expert-parallel',cmd)
        self.assertFalse(json.loads(cmd[cmd.index('--speculative-config')+1])['use_local_argmax_reduction'])
    def test_alias_cannot_select_same_gpu_twice(self):
        c=self.config();c['gpu_ids']=['0','GPU-test-a'];d=self.devices();d['GPU-test-a']=d['0']
        with self.assertRaises(ValueError):serve.build_command(c,'test',Path('/tmp/run'),d)
    def test_wrong_upstream_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);f=p/'vllm/v1/worker/gpu/model_runner.py';f.parent.mkdir(parents=True);f.write_text('other-version')
            r=subprocess.run([sys.executable,str(ROOT/'scripts/apply-overlay.py'),'--target',td],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertEqual(f.read_text(),'other-version');self.assertFalse((p/'draft_int8_runtime.py').exists())
    def test_ple_identity_detects_data_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);data=p/'input.bin';data.write_bytes(bytes(range(64)))
            source={'schema':'ple-gds-source-v1','row_count':4,'components':[{'name':'ngram_embedding','dtype':'U8','row_bytes':16,'shards':[{'path':str(data),'offset':0,'rows':4,'row_bytes':16,'row_start':0}]}]}
            artifact=p/'artifact';convert_compact(source,artifact,block_rows=2,alignment=4096)
            identity=prepare.enroll(artifact,True);self.assertTrue(identity['data_bytes_verified'])
            meta,g=load_current_compact(artifact,check_sources=False,check_data=False);payload=g/meta['data_file'];payload.chmod(0o644)
            with payload.open('r+b') as f:f.write(b'X')
            with self.assertRaises((ValueError,RuntimeError)):prepare.enroll(artifact,True)
    def test_preloaded_binaries_match_manifest(self):
        p=ROOT/'src/preload';rows=json.loads((p/'manifest.json').read_text())
        self.assertEqual(len(rows),377)
        for row in rows:self.assertEqual(hashlib.sha256((p/row['file']).read_bytes()).hexdigest(),row['sha256'])

if __name__=='__main__':unittest.main()
