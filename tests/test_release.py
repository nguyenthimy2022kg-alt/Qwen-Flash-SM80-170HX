"""发布接口的 CPU 检查，不加载模型、不调用 GPU。"""
import importlib.util,json,tempfile,unittest,subprocess,sys,hashlib
from pathlib import Path
from unittest.mock import patch
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
    def test_missing_identity_is_rejected_before_device_access(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json';c=self.config();c['ple_identity']=None
            path.write_text(json.dumps(c))
            with patch.object(serve,'gpu_inventory') as inventory:
                with patch.object(sys,'argv',['serve.py','start','--config',str(path)]):
                    with self.assertRaisesRegex(ValueError,'ple_identity'):serve.main()
                inventory.assert_not_called()
    def test_config_file_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=self.config()
            c.update(models_root=td,model_subdir='model',ple_artifact=str(root/'ple'),
                     cufile_config=str(root/'cufile.json'),ple_identity=str(root/'identity.json'))
            (root/'model').mkdir();(root/'ple').mkdir();(root/'ple/CURRENT').write_text('generation')
            (root/'cufile.json').mkdir();(root/'identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'cufile_config'):serve.check_paths(c)
            (root/'cufile.json').rmdir();(root/'cufile.json').write_text('{}')
            serve.check_paths(c)
    def test_existing_identity_does_not_trigger_full_data_read(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'identity.json';output.write_text('original')
            with patch.object(prepare,'enroll') as enroll:
                with patch.object(sys,'argv',['prepare-ple.py','enroll','--artifact',td,'--output',str(output)]):
                    with patch('sys.stderr'):
                        with self.assertRaises(SystemExit) as result:prepare.main()
                self.assertEqual(result.exception.code,2);enroll.assert_not_called()
            self.assertEqual(output.read_text(),'original')
    def test_mapping_rejects_wrong_global_scale_before_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            index=Path(td)/'index.json'
            headers={f'model.ngram_embedding.shard_{i}.weight':
                     {'dtype':'F8_E4M3','shape':[2500012,160]} for i in range(128)}
            scale='model.ngram_embedding.weight_scale'
            headers[scale]={'dtype':'BF16','shape':[]}
            index.write_text(json.dumps({'weight_map':{key:'part.safetensors' for key in headers}}))
            with patch.object(prepare,'_parse_safetensors_header',return_value=(headers,0)):
                self.assertEqual(prepare.mapping(index)['row_count'],320001536)
                for entry in ({'dtype':'F32','shape':[]},{'dtype':'BF16','shape':[2]}):
                    headers[scale]=entry
                    with self.assertRaisesRegex(ValueError,'单个 BF16'):prepare.mapping(index)
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
