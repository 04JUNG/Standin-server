import io, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from services.refine_release import service

class ServiceContracts(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(service.Unavailable): service.settings()
    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError): service.inside(Path('/tmp/assets'), '../secret')
    def test_withheld_cannot_download(self):
        with tempfile.TemporaryDirectory() as d:
            handle='a'*32;job=Path(d)/handle;job.mkdir();(job/'result.fbx').write_bytes(b'not a release')
            (job/'result.json').write_text(json.dumps({'status':'withheld'}))
            with patch.object(service,'settings',return_value={'REFINE_RELEASE_STATE':d}):
                with self.assertRaises(FileNotFoundError):service.artifact(handle)
    def test_modified_accepted_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            handle='b'*32;job=Path(d)/handle;job.mkdir();(job/'result.fbx').write_bytes(b'changed')
            (job/'result.json').write_text(json.dumps({'status':'accepted','artifact_sha256':'original'}))
            with patch.object(service,'settings',return_value={'REFINE_RELEASE_STATE':d}):
                with self.assertRaises(FileNotFoundError):service.artifact(handle)
    def test_invalid_handle_rejected(self):
        with patch.object(service,'settings',return_value={}):
            with self.assertRaises(FileNotFoundError):service.artifact('../anything')
    def test_oversized_image_rejected_before_worker(self):
        with patch.object(service,'settings',return_value={}):
            with self.assertRaises(ValueError):service.execute('p',b'x'*(12*1024*1024+1))
    def test_non_png_rejected(self):
        image=io.BytesIO();Image.new('RGB',(2,2)).save(image,'JPEG')
        with patch.object(service,'settings',return_value={}):
            with self.assertRaises(ValueError):service.execute('p',image.getvalue())

class RegisteredAssetContracts(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';(source/'input/p1-native').mkdir(parents=True);(source/'base-mesh').mkdir()
        (source/'input/original.bvh').write_bytes(b'original-bvh');(source/'input/original.fbx').write_bytes(b'original-fbx');(source/'base-mesh/asset.npz').write_bytes(b'mesh')
        rig={'source':'original.fbx','source_sha256':service.sha(source/'input/original.fbx'),'source_bvh_sha256':service.sha(source/'input/original.bvh')}
        (source/'input/p1-native/asset-source.json').write_text(json.dumps(rig))
        row={'case_id':'asset','status':'original_library_bvh_confirmed','target_identity':'single_depicted_target','rough':'old.png','rough_sha256':'old','selection':{'selected_pose_id':'pose','selected_bvh_sha256':rig['source_bvh_sha256'],'verified_selected_bvh':'original.bvh'}}
        (source/'input/manifest.json').write_text(json.dumps({'rows':[row]}))
        from services.refine_release.register_assets import register
        registry=root/'registry.json';registry.write_text(json.dumps(register(source,['asset'])))
        return source, registry
    def test_uploaded_rough_replaces_only_image_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,registry=self.fixture(root)
            service.prepare(registry,'pose',b'new rough',root/'job',source/'input/original.bvh')
            row=json.loads((root/'job/input/manifest.json').read_text())['rows'][0]
            self.assertEqual(row['rough_sha256'],service.sha(root/'job/input/rough/asset.png'))
            self.assertEqual((source/'input/original.bvh').read_bytes(),(root/'job/input/original.bvh').read_bytes())
            self.assertEqual(json.loads((source/'input/manifest.json').read_text())['rows'][0]['rough'],'old.png')
    def test_registry_asset_tampering_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,registry=self.fixture(root);(source/'base-mesh/asset.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):service.prepare(registry,'pose',b'image',root/'job')
    def test_database_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,registry=self.fixture(root);other=root/'other.bvh';other.write_bytes(b'other')
            with self.assertRaises(ValueError):service.prepare(registry,'pose',b'image',root/'job',other)
    def test_enabled_settings_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,registry=self.fixture(root)
            relative=['src/pose_humanart.py','experiments/refine_v5/develop_skeleton_seeds.py']
            for name in relative:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# extractor')
            pin=root/'source.json';pin.write_text(json.dumps({n:service.sha(root/n) for n in relative}))
            model=root/'model.json';model.write_text('{}')
            env=dict(REFINE_RELEASE_ENABLED='1',REFINE_RELEASE_REGISTRY=str(registry),REFINE_RELEASE_STATE=str(root/'state'),REFINE_EXTRACTOR_CHECKOUT=str(root),REFINE_EXTRACTOR_REVISION='rev',REFINE_MODEL_MANIFEST=str(model),REFINE_MODEL_MANIFEST_SHA256=service.sha(model),REFINE_EXTRACTOR_SOURCE_MANIFEST=str(pin),REFINE_EXTRACTOR_SOURCE_SHA256=service.sha(pin),OPENAI_API_KEY='not-sent')
            with patch.dict(os.environ,env,clear=True),patch.object(service.shutil,'which',return_value='/bin/blender'),patch.object(service.subprocess,'check_output',return_value='rev'):
                self.assertEqual(service.settings()['blender'],'/bin/blender')
                (root/relative[0]).write_text('# changed')
                with self.assertRaises(service.Unavailable):service.settings()


class DeliveryContracts(unittest.TestCase):
    def run_fake_worker(self, decision, root):
        image=io.BytesIO();Image.new('RGB',(2,2)).save(image,'PNG')
        def prepare(registry, pose_id, data, destination, expected_bvh):
            output=destination.parent/'run/gate/asset';(output/'release').mkdir(parents=True)
            (output/'decision.json').write_text(json.dumps(decision))
            (output/'release/result.fbx').write_bytes(b'verified-test-artifact')
            (output/'release/camera.json').write_text(json.dumps({'projection':'orthographic'}))
            return 'asset'
        from unittest.mock import MagicMock
        process=MagicMock();process.pid=123456;process.wait.return_value=0
        cfg={'REFINE_RELEASE_STATE':str(root),'REFINE_RELEASE_REGISTRY':'registry','REFINE_EXTRACTOR_CHECKOUT':'extractor','REFINE_MODEL_MANIFEST':'model','blender':'blender'}
        with patch.object(service,'settings',return_value=cfg),patch.object(service,'prepare',side_effect=prepare),patch.object(service.subprocess,'Popen',return_value=process),patch.object(service.os,'killpg') as kill:
            result=service.execute('pose',image.getvalue());kill.assert_called_once()
            return result
    def test_accepted_requires_roundtrip_and_returns_camera(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_fake_worker({'release_eligible':True,'status':'accepted_under_measured_policy','selected':'shadow','export':{'full_matrix_roundtrip_passed':True}},Path(d))
            self.assertEqual(result['status'],'accepted');self.assertEqual(result['camera']['projection'],'orthographic')
    def test_missing_roundtrip_never_delivers(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_fake_worker({'release_eligible':True,'status':'accepted_under_measured_policy','selected':'shadow'},Path(d))
            self.assertEqual(result['status'],'withheld');self.assertIsNone(result['download_url'])
            self.assertFalse((Path(d)/result['handle']/'result.fbx').exists())

if __name__=='__main__':unittest.main()

class RuntimeIntegrity(unittest.TestCase):
    def test_packaged_runtime_is_hash_bound(self):
        service.verify_runtime()
    def test_modified_runtime_hash_fails_closed(self):
        with patch.object(service, 'sha', return_value='changed'):
            with self.assertRaises(service.Unavailable): service.verify_runtime()
