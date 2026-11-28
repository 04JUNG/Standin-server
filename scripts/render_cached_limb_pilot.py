"""Render frozen reference/before/final at shared native cameras, offline."""
import sys,json,math
from pathlib import Path
repo,root,run=map(lambda p:Path(p).resolve(),sys.argv[sys.argv.index('--')+1:]);sys.path.insert(0,str(repo))
from converter import retarget as rt
from src.arm_refine.score_views import points,camera_spec,render
from src.arm_refine.blender_adapter import body_frame
for case in json.loads((run/'cases.json').read_text()):
    folder=run/case['id'];sources={'before':folder/'before.fbx','after':folder/'after.fbx'}
    bounds=[];front=None
    for name,path in sources.items():
        rt.reset_scene();arm,meshes=rt.import_character(str(path));bounds+=points(meshes,True)
        if front is None:
            forward=-body_frame(arm)[:,2];front=math.degrees(math.atan2(forward[0],-forward[1]))
    cameras={str(angle):camera_spec(bounds,front+angle) for angle in (0,45,90)}
    for name,path in sources.items():
        rt.reset_scene();rt.import_character(str(path))
        for angle,spec in cameras.items():render(folder/(name+'_'+angle+'.png'),spec)
    (folder/'cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
    print('rendered',case['id'],flush=True)
