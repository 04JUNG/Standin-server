import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from services.refine_release import assets
from services.refine_release.worker import app
from fastapi.testclient import TestClient

class OriginalAssets(unittest.TestCase):
    def test_converter_sources_pinned(self):
        self.assertEqual(len(assets.verify_builder()),64)
    def test_cache_identity_changes_with_every_source(self):
        base=assets.cache_key('bvh','body','builder','blender')
        for i in range(4):
            parts=['bvh','body','builder','blender'];parts[i]+='changed'
            self.assertNotEqual(base,assets.cache_key(*parts))
    def test_unregistered_pose_prepares_and_rough_is_never_cached(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);body=root/'body.fbx';body.write_bytes(b'body')
            def build(command,**kwargs):
                target=Path(command[-1]);(target/'source.fbx').write_bytes(b'fbx');(target/'mesh.npz').write_bytes(b'mesh');(target/'rig.json').write_text('{}')
            cfg={'REFINE_RELEASE_STATE':str(root),'blender':'blender'}
            with patch.object(assets,'configuration',return_value=(body,'builder')),patch.object(assets.subprocess,'check_output',return_value='Blender 5.2\n'),patch.object(assets.subprocess,'run',side_effect=build) as run:
                for n,image in [('first',b'rough one'),('second',b'rough two')]:
                    assets.prepare('never-in-any-registry',b'HIERARCHY\nMOTION\n',image,root/n,cfg)
                self.assertEqual(run.call_count,1)
                self.assertEqual((root/'second/input/rough/source.png').read_bytes(),b'rough two')
                row=json.loads((root/'second/input/manifest.json').read_text())['rows'][0]
                self.assertEqual(row['selection']['selected_pose_id'],'never-in-any-registry')
                self.assertTrue(json.loads((root/'second/original-provenance.json').read_text())['cache_hit'])
                cache=next((root/'original-cache').iterdir())
                self.assertFalse(any('rough' in str(p) for p in cache.rglob('*')))
                (cache/'source.fbx').write_bytes(b'tampered')
                assets.prepare('never-in-any-registry',b'HIERARCHY\nMOTION\n',b'third',root/'third',cfg)
                self.assertEqual(run.call_count,2)
    def test_mismatched_uploaded_original_never_runs_solver(self):
        with patch('services.refine_release.worker.service.settings',return_value={}),patch('services.refine_release.worker.service.execute') as solve:
            r=TestClient(app).post('/refine/release',data={'pose_id':'new','source_sha256':'0'*64},files={'image':('rough.png',b'png'),'original_bvh':('source.bvh',b'changed')})
            self.assertEqual(r.status_code,422);solve.assert_not_called()
    def test_invalid_bvh_never_starts_blender(self):
        with patch.object(assets.subprocess,'run') as run:
            with self.assertRaises(ValueError):assets.prepare('new',b'not bvh',b'image',Path('/unused'),{})
            run.assert_not_called()
