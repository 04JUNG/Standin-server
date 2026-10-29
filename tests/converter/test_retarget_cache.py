import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from converter.retarget_cache import (binding,cache_key,encode,capture_base,restore_base,
 RetargetBaseCache,CacheIntegrityError,Entry)
from converter_api.runner import BlenderRunner,RunnerSettings,WorkerIntegrityError
from converter import worker

DATA=b'Kaydara FBX Binary base-immutable'
BODY={'ok':True,'frame':0,'output_mode':'rigged_rest','mirrored':False,'mapped_bones':22,'missing_required':[]}
IDENTITY=binding('a'*64,'b'*64,'character',False)

def packet():
 return dict(schema='retarget-base.lab.v1',binding=IDENTITY,artifact_sha256=hashlib.sha256(DATA).hexdigest(),
  body_report=BODY,body_report_sha256=hashlib.sha256(encode(BODY)).hexdigest())

class CacheTests(unittest.TestCase):
 def test_identity_changes_invalidate_but_hand_scope_camera_are_not_base_inputs(self):
  cache=RetargetBaseCache(100000);cache.put(IDENTITY,DATA,encode(packet()))
  for field in IDENTITY:
   other=dict(IDENTITY);other[field]=str(IDENTITY[field])+'changed'
   self.assertNotEqual(cache_key(IDENTITY),cache_key(other),field)
   self.assertIsNone(cache.get(other),field)
  self.assertNotIn('hands',IDENTITY);self.assertNotIn('scope',IDENTITY);self.assertNotIn('camera',IDENTITY)
 def test_ttl_eviction_and_disabled_or_oversized_entries(self):
  clock=[1.];meta=encode(packet());size=len(DATA)+len(meta)
  cache=RetargetBaseCache(size,10,lambda:clock[0]);self.assertTrue(cache.put(IDENTITY,DATA,meta))
  self.assertIsNotNone(cache.get(IDENTITY));clock[0]=11.;self.assertIsNone(cache.get(IDENTITY))
  self.assertFalse(RetargetBaseCache().put(IDENTITY,DATA,meta))
  self.assertFalse(RetargetBaseCache(1).put(IDENTITY,DATA,meta))
  other=dict(IDENTITY);other['character_id']='other';m=packet();m['binding']=other
  cache=RetargetBaseCache(size+len(DATA)+len(encode(m))-1)
  cache.put(IDENTITY,DATA,meta);cache.put(other,DATA,encode(m))
  self.assertIsNone(cache.get(IDENTITY));self.assertIsNotNone(cache.get(other))
 def test_corrupt_fbx_or_body_report_cannot_be_used(self):
  cache=RetargetBaseCache(100000)
  with self.assertRaises(CacheIntegrityError):cache.put(IDENTITY,DATA+b'corrupt',encode(packet()))
  m=packet();m['body_report']={**BODY,'frame':1}
  with self.assertRaises(CacheIntegrityError):cache.put(IDENTITY,DATA,encode(m))
  cache.put(IDENTITY,DATA,encode(packet()));key=cache_key(IDENTITY)
  cache._entries[key]=Entry(DATA+b'corrupt',encode(packet()),float('inf'))
  with self.assertRaises(CacheIntegrityError):cache.get(IDENTITY)
 def test_postprocessed_result_or_wrong_identity_is_rejected(self):
  for field in ['hands','source_fingers','framing','camera_rotation','preview_sha256','shape_bake']:
   m=packet();m['body_report']={**BODY,field:{}};m['body_report_sha256']=hashlib.sha256(encode(m['body_report'])).hexdigest()
   with self.assertRaises(CacheIntegrityError):RetargetBaseCache(100000).put(IDENTITY,DATA,encode(m))
  m=packet();m['binding']={**IDENTITY,'source_sha':'changed'}
  with self.assertRaises(CacheIntegrityError):RetargetBaseCache(100000).put(IDENTITY,DATA,encode(m))
 def test_private_restore_and_report_mutation_cannot_modify_base(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   d=Path(folder);source=d/'source.fbx';source.write_bytes(DATA)
   capture_base(source,BODY,IDENTITY,d/'base.fbx',d/'base.json')
   body,_=restore_base(d/'base.fbx',d/'base.json',d/'output.fbx',IDENTITY)
   (d/'output.fbx').write_bytes(b'hand-modified');body['hands']={'left':'open'}
   self.assertEqual((d/'base.fbx').read_bytes(),DATA)
   self.assertEqual(json.loads((d/'base.json').read_bytes())['body_report'],BODY)
 def test_invalid_cache_bounds(self):
  for args in [(-1,10),(10,0),(10,float('nan')),(10,float('inf')),(1.5,10),(True,10)]:
   with self.assertRaises(ValueError):RetargetBaseCache(*args)

class PublishTests(unittest.TestCase):
 def test_final_failure_cannot_publish_and_success_publishes_pristine_base(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   d=Path(folder);character=d/'character.fbx';character.write_bytes(b'character')
   charsha=hashlib.sha256(character.read_bytes()).hexdigest();bvh=b'HIERARCHY MOTION'
   identity=binding(hashlib.sha256(bvh).hexdigest(),charsha,'character',False)
   class FakeRunner(BlenderRunner):
    fail=True
    corrupt_receipt=False
    def inspect_blender(self):pass
    def _command(self,job_path):self.job=json.loads(job_path.read_text());return ['never-executed']
    def _communicate(self,process,timeout):
     job=self.job;base=job['retarget_cache']
     if base['mode']=='capture':
      Path(job['output_path']).write_bytes(DATA)
      metadata=capture_base(job['output_path'],BODY,identity,base['fbx_path'],base['meta_path'])
     else:_,metadata=restore_base(base['fbx_path'],base['meta_path'],job['output_path'],identity)
     Path(job['output_path']).write_bytes(b'postprocessed final output')
     report={'retarget_reuse':{'key':cache_key(identity),'base_sha256':metadata['artifact_sha256'],'body_report_sha256':metadata['body_report_sha256'],'mode':base['mode']}}
     if self.corrupt_receipt:report['retarget_reuse']['body_report_sha256']='wrong'
     Path(job['report_path']).write_text(json.dumps(report));return '',''
    def _validate_success(self,**kwargs):
     if self.fail:raise WorkerIntegrityError('final PNG failed')
     return 'verified-final'
   runner=FakeRunner(RunnerSettings(blender_binary='never-executed',worker_path=ROOT/'converter/worker.py',temp_root=d,retarget_cache_bytes=100000))
   kwargs=dict(bvh_bytes=bvh,character_path=character,character_id='character',character_sha256=charsha,
    conversion_id='conversion',mirror=False,output_scope='half',preview_view='front',hands={'left':'fist','right':'fist'})
   with patch('converter_api.runner.subprocess.Popen'):
    with self.assertRaises(WorkerIntegrityError):runner._convert_one(**kwargs)
    self.assertIsNone(runner.retarget_cache.get(identity))
    runner.fail=False;runner.corrupt_receipt=True
    with self.assertRaises(WorkerIntegrityError):runner._convert_one(**kwargs)
    self.assertIsNone(runner.retarget_cache.get(identity))
    runner.corrupt_receipt=False;self.assertEqual(runner._convert_one(**kwargs),'verified-final')
    self.assertEqual(runner.job['retarget_cache']['mode'],'capture')
    self.assertEqual(runner._convert_one(**kwargs),'verified-final')
    self.assertEqual(runner.job['retarget_cache']['mode'],'reuse')
   self.assertEqual(runner.retarget_cache.get(identity).artifact,DATA)

class JobTests(unittest.TestCase):
 def test_private_cache_job_paths_and_unsupported_modes(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
   d=Path(folder);(d/'source.bvh').write_bytes(b'HIERARCHY MOTION');(d/'character.fbx').write_bytes(DATA)
   job=dict(schema_version=3,conversion_id='id',temp_dir=str(d),bvh_path=str(d/'source.bvh'),
    character_fbx=str(d/'character.fbx'),output_path=str(d/'output.fbx'),report_path=str(d/'report.json'),
    character_id='character',character_sha256='b'*64,frame=0,mirror=False,output_mode='rigged_rest',
    apply_root_translation=False,embed_textures=False,force_exact_v324=False,thumbnail=None,
    output_scope='half',preview_view='front',hands={'left':'fist','right':'fist'},
    retarget_cache={'mode':'capture','fbx_path':str(d/'base.fbx'),'meta_path':str(d/'base.json')})
   path=d/'job.json'
   def load(value):path.write_text(json.dumps(value));return worker.load_job(path)
   self.assertEqual(load(job)['retarget_cache']['mode'],'capture')
   for change in [{'mode':'invalid'},{'fbx_path':str(d.parent/'escape.fbx')},{'fbx_path':str(d/'output.fbx')},
                  {'mode':'reuse'},{'extra':'field'}]:
    with self.assertRaises(worker.JobValidationError):load({**job,'retarget_cache':{**job['retarget_cache'],**change}})
   (d/'base.fbx').symlink_to(d/'character.fbx')
   with self.assertRaises(worker.JobValidationError):load({**job,'retarget_cache':{**job['retarget_cache'],'mode':'reuse'}})
   with self.assertRaises(worker.JobValidationError):load({**job,'shape_recipe':{}})

if __name__=='__main__':
 def forbidden(*a,**k):raise AssertionError('no sockets')
 with patch.object(socket.socket,'connect',forbidden):unittest.main(verbosity=2)
