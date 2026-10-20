"""Factory scene integrity and opt-in guards. No Blender or sockets in these unit tests."""
import hashlib,json,socket,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter import fast_scene_reset as fast,worker
from converter_api.runner import BlenderRunner,RunnerSettings,WorkerIntegrityError
class ResetTests(unittest.TestCase):
 def setUp(self):
  self.template=patch.object(fast,'PATH',ROOT/'tests/converter/fixtures/empty_factory.blend')
  self.template.start();self.addCleanup(self.template.stop)
 def test_template_identity_and_corruption_rejection(self):
  self.assertEqual(fast.identity()['template_sha256'],hashlib.sha256(fast.PATH.read_bytes()).hexdigest())
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   bad=Path(folder)/'corrupt.blend';bad.write_bytes(fast.PATH.read_bytes()+b'corrupt')
   with patch.object(fast,'PATH',bad),self.assertRaises(RuntimeError):fast.identity()
 def test_symlink_template_rejected(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   link=Path(folder)/'link.blend';link.symlink_to(fast.PATH)
   with patch.object(fast,'PATH',link),self.assertRaises(RuntimeError):fast.identity()
 def test_reset_cannot_load_ui_or_execute_scripts(self):
  calls=[]
  bpy=SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(open_mainfile=lambda **kw:(calls.append(kw) or {'FINISHED'}))))
  with patch.dict(sys.modules,{'bpy':bpy}),patch.object(fast,'fingerprint',return_value=fast.EXPECTED_SCENE),patch.object(fast,'CALLS',0):
   fast.reset_scene();self.assertEqual(fast.CALLS,1)
  self.assertIs(calls[0]['load_ui'],False);self.assertIs(calls[0]['use_scripts'],False)
 def test_failed_open_or_nonempty_scene_fails_closed(self):
  for result,state in [({'CANCELLED'},fast.EXPECTED_SCENE),({'FINISHED'},{**fast.EXPECTED_SCENE,'objects':1})]:
   bpy=SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(open_mainfile=lambda **kw:result)))
   with patch.dict(sys.modules,{'bpy':bpy}),patch.object(fast,'fingerprint',return_value=state),patch.object(fast,'CALLS',0):
    with self.assertRaises(RuntimeError):fast.reset_scene()
    self.assertEqual(fast.CALLS,0)
 def test_default_disabled_and_worker_options_guard(self):
  self.assertIs(RunnerSettings(blender_binary='not-executed',worker_path=ROOT/'converter/worker.py').fast_scene_reset,False)
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   d=Path(folder);(d/'source.bvh').write_bytes(b'HIERARCHY MOTION');(d/'character.fbx').write_bytes(b'FBX')
   job=dict(schema_version=3,conversion_id='id',temp_dir=str(d),bvh_path=str(d/'source.bvh'),character_fbx=str(d/'character.fbx'),
    output_path=str(d/'output.fbx'),report_path=str(d/'report.json'),character_id='character',character_sha256='b'*64,
    frame=0,mirror=False,output_mode='rigged_rest',apply_root_translation=False,embed_textures=False,
    force_exact_v324=False,thumbnail=None,output_scope='half',preview_view='front',hands={'left':'fist','right':'fist'},fast_scene_reset=True)
   p=d/'job.json'
   def load(value):p.write_text(json.dumps(value));return worker.load_job(p)
   self.assertIs(load(job)['fast_scene_reset'],True)
   for options in [{'fast_scene_reset':1},{'fast_scene_reset':'true'},{'fast_scene_reset':None},{'shape_recipe':{}},{'hands':None},{'preview_view':'back'}]:
    with self.assertRaises(worker.JobValidationError):load({**job,**options})
   disabled={**job};disabled.pop('fast_scene_reset');self.assertFalse(load(disabled).get('fast_scene_reset',False))
 def test_settings_read_opt_in_environment(self):
  with patch.dict('os.environ',{'CONVERTER_RETARGET_CACHE_BYTES':'67108864','CONVERTER_RETARGET_CACHE_TTL_SECONDS':'120','CONVERTER_FAST_SCENE_RESET':'1'}):
   settings=RunnerSettings.from_env(ROOT)
   self.assertEqual(settings.retarget_cache_bytes,67108864)
   self.assertEqual(settings.retarget_cache_ttl_seconds,120)
   self.assertIs(settings.fast_scene_reset,True)
 def test_parent_rejects_wrong_receipt_or_zero_resets(self):
  runner=BlenderRunner(RunnerSettings(blender_binary='not-executed',fast_scene_reset=True,worker_path=ROOT/'converter/worker.py'))
  kwargs=dict(output_path=ROOT/'missing.fbx',temp_dir=ROOT,conversion_id='id',character_id='character',character_sha256='b'*64,
   source_bvh_sha256='a'*64,mirror=False,returncode=0,combined_log='',hands={'left':'fist','right':'fist'},output_scope='half',preview_view='front')
  for receipt in [{},{**fast.identity(),'calls':0},{**fast.identity(),'calls':1,'template_sha256':'wrong'}]:
   with self.assertRaisesRegex(WorkerIntegrityError,'factory-empty scene receipt mismatch'):
    runner._validate_success(report={'ok':True,'scene_reset':receipt},**kwargs)
if __name__=='__main__':
 def forbidden(*a,**k):raise AssertionError('no network')
 with patch.object(socket.socket,'connect',forbidden):unittest.main(verbosity=2)
