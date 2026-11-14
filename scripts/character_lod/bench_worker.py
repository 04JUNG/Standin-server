"""Local-only timing delegates around the unchanged production worker."""
import functools,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter import shape_bake,final_artifact,worker
records=[]
def wrap(owner,name,label):
 original=getattr(owner,name)
 @functools.wraps(original)
 def measured(*args,**kw):
  start=time.monotonic();ok=False
  try:
   result=original(*args,**kw);ok=True;return result
  finally:records.append({'stage':label,'seconds':time.monotonic()-start,'ok':ok})
 setattr(owner,name,measured)
wrap(shape_bake,'_load','shape_load_controls_validate')
wrap(shape_bake,'transfer_solver_pose','pose_transfer')
wrap(final_artifact,'validate','final_validation_reload')
from converter import retarget,convert
wrap(retarget,'export_fbx','solver_fbx_export')
old_load=shape_bake._load
def load(*args,**kw):
 c=old_load(*args,**kw)
 for name,label in [('export_fixed','authoring_fbx_export_roundtrip'),('correct_pose','psd_pose_correction')]:wrap(c,name,label)
 return c
shape_bake._load=load
old_run=worker.run_job
def run(job):
 start=time.monotonic();r=old_run(job);r['local_lod_timing']={'worker_total_seconds':time.monotonic()-start,'calls':records,'nested':True};return r
worker.run_job=run
if __name__=='__main__':raise SystemExit(worker.main())
