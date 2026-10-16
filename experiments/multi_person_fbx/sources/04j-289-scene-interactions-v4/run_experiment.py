from pathlib import Path
import subprocess,time,json,sys
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];python='/private/tmp/standin-04j289-venv/bin/python';lp=str(WORK/'tmp/rtmpose_eval_venv/bin/python');blender='/private/tmp/standin-complement-blender-KDq2Ja/Blender.app/Contents/MacOS/Blender';node='/Users/dw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
def bc(script):return [blender,'--background','--python-exit-code','1','--python',str(ROOT/script)]
stages=[('plan',[python,str(ROOT/'plan.py')]),('people',bc('build_people.py')),('layout-before-arms',[lp,str(ROOT/'fit_layout.py')]),('arms',bc('refine_arms.py')),('layout',[lp,str(ROOT/'fit_layout.py')]),('scenes',bc('build_scenes.py')),('three',[node,str(ROOT/'verify_fbx.mjs')])];results=[]
for name,command in stages:
 print('START',name,flush=True);start=time.monotonic()
 with (ROOT/(name+'.log')).open('w') as log:r=subprocess.run(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT)
 row=dict(stage=name,exit_code=r.returncode,elapsed_s=time.monotonic()-start,command=command);results.append(row);(ROOT/'pipeline-result.json').write_text(json.dumps(results,indent=2));print('DONE',row,flush=True)
 if r.returncode:sys.exit(r.returncode)
