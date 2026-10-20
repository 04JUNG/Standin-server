"""Memory/filesystem mocks only. No Blender process, external HTTP or socket."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from converter.lab_timing import StageTimings
from converter import framing, worker


class LabTests(unittest.TestCase):
    def test_real_frozen_solver_lineage_remains_valid(self):
        lineage = worker._frozen_lineage()
        self.assertEqual(lineage['retarget_sha256'], worker.RETARGET_SHA256)
        self.assertEqual(lineage['solver_manifest_sha256'], worker.SOLVER_MANIFEST_SHA256)

    def test_stage_clock_and_failure_are_exclusive_and_transparent(self):
        values = iter([1., 1.010, 2., 2.020])
        timings = StageTimings(enabled=True, clock=lambda: next(values))
        with timings.measure('crop'):
            pass
        error = RuntimeError('original')
        with self.assertRaises(RuntimeError) as caught:
            with timings.measure('png_render_write'):
                raise error
        self.assertIs(caught.exception, error)
        self.assertAlmostEqual(timings.records['crop']['duration_ms'], 10.)
        self.assertFalse(timings.records['png_render_write']['ok'])
        disabled = StageTimings(enabled=False, clock=lambda: self.fail('clock called while disabled'))
        with disabled.measure('crop'):
            pass
        self.assertEqual(disabled.records, {})

    def execute_mock_worker(self, enabled):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            folder = Path(tmp)
            bvh = folder/'source.bvh'; bvh.write_bytes(b'fake BVH')
            character = folder/'character.fbx'; character.write_bytes(b'fake character')
            output = folder/'final.fbx'
            calls = []
            def convert(**kwargs):
                calls.append('retarget'); Path(kwargs['out_path']).write_bytes(b'FBX:body')
                return SimpleNamespace(ok=True, as_dict=lambda: {'ok':True})
            def hands(path, requested):
                calls.append('hands'); Path(path).write_bytes(Path(path).read_bytes()+b':hands')
                return {'left':requested['left'],'right':requested['right']}
            def import_character(path):
                calls.append(('import', Path(path).read_bytes()))
                return object(), [Path(path).read_bytes()]
            def export(path, **kwargs):
                calls.append('export'); Path(path).write_bytes(Path(path).read_bytes()+b':crop')
            def render(meshes, path, view, **kwargs):
                calls.append(('render', meshes[0])); Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+meshes[0])
            rt = SimpleNamespace(reset_scene=lambda:None,import_character=import_character,export_fbx=export)
            camera = SimpleNamespace(validate_rotation=lambda _:None,rotate_scene=lambda *_:None)
            modules = {'converter.convert':SimpleNamespace(convert=convert),'converter.hands':SimpleNamespace(bake_hands=hands),
                       'converter.retarget':rt,'converter.camera':camera,
                       'bpy':SimpleNamespace(app=SimpleNamespace(version_string=worker.EXPECTED_BLENDER_VERSION,
                                                              build_hash=worker.EXPECTED_BLENDER_BUILD_HASH.encode()))}
            job = dict(bvh_path=str(bvh),character_fbx=str(character),character_sha256=hashlib.sha256(character.read_bytes()).hexdigest(),
                       output_path=str(output),temp_dir=str(folder),mirror=False,force_exact_v324=False,
                       shape_recipe=None,hands={'left':'fist','right':'fist'},output_scope='half',preview_view='front',
                       camera_rotation=None,conversion_id='mock-conversion',character_id='mock-character',thumbnail=None)
            with patch.dict(sys.modules, modules),\
                 patch.object(framing,'crop_meshes',return_value={'scope':'half'}),patch.object(framing,'render_preview',side_effect=render),\
                 patch.dict(os.environ,{'STANDIN_LAB_PREVIEW_TRACE':'1' if enabled else '0'}):
                report = worker.run_job(job)
            final_bytes = output.read_bytes(); preview = (folder/'preview.png').read_bytes()
            self.assertEqual(final_bytes, b'FBX:body:hands:crop')
            self.assertEqual(preview[8:], final_bytes)
            self.assertEqual(report['artifact_sha256'], hashlib.sha256(final_bytes).hexdigest())
            self.assertEqual(report['preview_sha256'], hashlib.sha256(preview).hexdigest())
            self.assertEqual(calls[-2], ('import', final_bytes))
            self.assertEqual(calls[-1], ('render', final_bytes))
            return report, final_bytes, preview

    def test_worker_enabled_disabled_produce_identical_final_pair(self):
        enabled, fbx, png = self.execute_mock_worker(True)
        disabled, off_fbx, off_png = self.execute_mock_worker(False)
        self.assertEqual((fbx,png),(off_fbx,off_png))
        self.assertNotIn('lab_stage_timings', disabled)
        expected = ['worker_init','retarget_initial_export','hand_import_apply_export','framing_import','crop',
                    'framing_export','final_fbx_reimport','png_render_write']
        self.assertEqual(list(enabled['lab_stage_timings']), expected)
        for metric in enabled['lab_stage_timings'].values():
            self.assertTrue(metric['ok']); self.assertGreaterEqual(metric['duration_ms'],0.)
        print('LAB_WORKER_TRACE '+json.dumps({'mockOnly':True,'realBlender':False,'stages':enabled['lab_stage_timings'],
                                             'artifactSha256':enabled['artifact_sha256'],'previewSha256':enabled['preview_sha256']}))

    def test_framing_render_failure_is_not_a_successful_preview(self):
        rt=SimpleNamespace(reset_scene=lambda:None,import_character=lambda _:(object(),[]),export_fbx=lambda *a,**kw:None)
        camera=SimpleNamespace(validate_rotation=lambda _:None,rotate_scene=lambda *_:None)
        timings=StageTimings(enabled=True)
        original=RuntimeError('render failed')
        with patch.dict(sys.modules,{'converter.retarget':rt,'converter.camera':camera}),\
             patch.object(framing,'crop_meshes',return_value={}),patch.object(framing,'render_preview',side_effect=original):
            with self.assertRaises(RuntimeError) as caught:
                framing.process_output('mock.fbx','half',preview_path='mock.png',lab_timings=timings)
        self.assertIs(caught.exception,original)
        self.assertFalse(timings.records['png_render_write']['ok'])


if __name__ == '__main__':
    def forbidden(*args, **kwargs):
        raise AssertionError('LAB_OFFLINE: no sockets allowed')
    with patch.object(socket.socket,'connect',forbidden),patch.object(socket,'create_connection',forbidden):
        unittest.main(verbosity=2)
