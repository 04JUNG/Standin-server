"""Opt-in delegates used only in the dedicated profiling run; no solver edits."""
import functools,sys,time
from pathlib import Path
RECORDS=[]

def install():
 from converter import retarget as rt
 def wrap(owner,name,label):
  original=getattr(owner,name)
  @functools.wraps(original)
  def measured(*args,**kwargs):
   caller=sys._getframe(1).f_code
   start=time.monotonic();ok=False
   try:
    result=original(*args,**kwargs);ok=True;return result
   finally:
    RECORDS.append(dict(function=label,caller=Path(caller.co_filename).name+':'+caller.co_name,
     durationMs=(time.monotonic()-start)*1000,ok=ok,
     file=Path(str(args[0])).name if args and isinstance(args[0],(str,Path)) else None))
  setattr(owner,name,measured)
 for name in ['reset_scene','import_character','import_bvh','apply_output_mode','export_fbx']:
  wrap(rt,name,'retarget.'+name)
 for name in ['airborne_plantar','foot_plant','ankle_clearance','ankle_swing']:
  module=__import__('converter.'+name,fromlist=['convert_safe'])
  if hasattr(module,'convert_safe'):wrap(module,'convert_safe',name+'.convert_safe')
 try:
  from io_scene_fbx import parse_fbx
  wrap(parse_fbx,'parse','fbx.parse')
 except ImportError:pass
