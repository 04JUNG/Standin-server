from pathlib import Path
import subprocess,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
python='/private/tmp/standin-04j289-venv/bin/python';layout_python=str(WORK/'tmp/rtmpose_eval_venv/bin/python');blender='/private/tmp/standin-complement-blender-KDq2Ja/Blender.app/Contents/MacOS/Blender';node='/Users/dw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
stages=[('selection',[python,str(ROOT/'select_refine.py')]),('people',[blender,'--background','--python-exit-code','1','--python',str(ROOT/'build_people.py')]),('layout',[layout_python,str(ROOT/'fit_layout.py')]),('scenes',[blender,'--background','--python-exit-code','1','--python',str(ROOT/'build_scenes.py')]),('three',[node,str(ROOT/'verify_fbx.mjs')]),('review',[layout_python,str(ROOT/'build_review.py')])]
if '--layout-onward' in sys.argv:stages=stages[2:]
results=[]
for name,command in stages:
 start=time.monotonic();print('START',name,flush=True)
 with (ROOT/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT)
 row=dict(stage=name,command=command,exit_code=result.returncode,elapsed_s=time.monotonic()-start);results.append(row);(ROOT/'pipeline-result.json').write_text(json.dumps(results,indent=2));print('DONE',row,flush=True)
 if result.returncode:sys.exit(result.returncode)
