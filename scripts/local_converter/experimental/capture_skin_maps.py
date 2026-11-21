"""Capture linear skin maps for registered FBX/pose pairs in the pinned runtime."""
import hashlib
import sys
from pathlib import Path
import bpy
from mathutils import Matrix
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.local_converter.experimental.trace_pose_onset import linear_skin_maps


def capture(runtime,model,pose,stage,out,fraction):
    sys.path.insert(0,str(runtime))
    from converter import retarget
    from converter.local import inspection
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=inspection.import_fbx(model)
    assert len(meshes)==1
    obj=meshes[0]
    profile,_=inspection.inspect(arm,meshes)
    rest=np.asarray(inspection.surface(meshes)[obj.name])
    raw=np.load(stage)
    np.testing.assert_allclose(rest,raw["rest"],atol=2e-5,rtol=0)
    retarget._apply_object_transform(arm)
    initial={bone.name:bone.matrix_basis.copy() for bone in arm.pose.bones}
    src=retarget.import_bvh(str(pose),frame=0)
    assert retarget.retarget(src,arm,dst_profile=profile["profile"],apply_root_translation=False).ok
    if fraction != 1.:
        for bone in arm.pose.bones:
            l0,r0,s0=initial[bone.name].decompose()
            l1,r1,s1=bone.matrix_basis.decompose()
            bone.matrix_basis=Matrix.LocRotScale(l0.lerp(l1,fraction),r0.slerp(r1,fraction),s0.lerp(s1,fraction))
        bpy.context.view_layer.update()
    maps=linear_skin_maps(arm,obj,np.arange(len(rest)))
    predicted=np.einsum("nij,nj->ni",maps[:,:3,:3],rest)+maps[:,:3,3]
    error=float(np.linalg.norm(predicted-raw["solver" if "solver" in raw else "corrected"],axis=1).max())
    assert error<2e-5, error
    np.savez_compressed(out,maps=maps,sourceSHA=hashlib.sha256(stage.read_bytes()).hexdigest(),
                        poseSHA=hashlib.sha256(pose.read_bytes()).hexdigest(), reconstructionErrorM=error)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("runtime",type=Path)
    parser.add_argument("model",type=Path)
    parser.add_argument("cases",nargs="+",type=Path)
    parser.add_argument("--fraction",type=float,default=1.)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    runtime,model,cases=args.runtime,args.model,args.cases
    if len(cases)%3: raise ValueError("Expected pose, stage, output triples")
    for i in range(0,len(cases),3): capture(runtime,model,*cases[i:i+3],args.fraction)
