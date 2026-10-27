"""Blender planning input: bind a realized rig, never certify its old mesh.

The complete pose is audited against the original converter rig. Mesh vertices
are not accepted from this artifact. Final QA skins fresh corrected rest vertices
and independently checks them, while preserving this realized rig during export.
"""
from pathlib import Path
import argparse,json,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.refine_v3_safe_snapshot_qa import snapshot,sha
    from converter.pose_safety import audit_pose
    from converter.hands import bake_hands
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--realized-body',type=Path)
    source.add_argument('--rest-fbx',type=Path,help='Realize the proposal directly from the derived rest rig')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--hand-left',required=True);p.add_argument('--hand-right',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.output_dir.mkdir(parents=True,exist_ok=False)
    with np.load(a.proposal,allow_pickle=False) as z:x={k:z[k] for k in z.files}
    rt.import_character=import_full_character
    rig=a.output_dir/'unapproved-body-rig.fbx'
    if a.rest_fbx:
        import bpy
        from mathutils import Matrix
        rt.reset_scene();arm,meshes=rt.import_character(str(a.rest_fbx))
        names=list(x['names'])
        if sorted(b.name for b in arm.data.bones)!=names:raise ValueError('proposal_rig_mismatch')
        for bone in arm.data.bones:
            arm.pose.bones[bone.name].matrix=arm.matrix_world.inverted()@Matrix(x['body_bones'][names.index(bone.name)])
            bpy.context.view_layer.update()
        rt.apply_output_mode(arm,meshes,'rigged_rest');rt.export_fbx(str(rig),embed_textures=False)
    else:
        shutil.copyfile(a.realized_body,rig)
    body=snapshot(rig)
    # Bind the actual one-export adapter, including FBX serialization. Some
    # near-pi bone rolls need one canonicalizing roundtrip before their next
    # export stays within the unchanged 1e-5 matrix contract. Never increase it.
    probe=a.output_dir/'unapproved-body-export-probe.fbx'
    for iteration in range(3):
        rt.reset_scene();rt.import_character(str(rig));rt.export_fbx(str(probe),embed_textures=False)
        exported=snapshot(probe)
        error=float(np.max(abs(exported['bones']-body['bones'])))
        if error<=1e-5:
            body=exported
            break
        if iteration==2:raise ValueError('realized_rig_export_not_stable')
        shutil.copyfile(probe,rig);body=exported
    if body['names']!=list(x['names']) or body['parents']!=json.loads(str(x['parents'])):
        raise ValueError('realized_rig_identity')
    audit=audit_pose(x['base_bones'],body['bones'],body['names'],body['parents'],json.loads(str(x['camera_json'])),depth_reference=x['depth_reference'])
    if audit['status']!='pose_validated':raise ValueError('realized_pose_rejected:'+str(audit['violations']))
    # Keep realization close to the proposed physical pose. This is an input
    # generation bound, not the strict final export equality check (1e-5).
    if np.max(np.linalg.norm(body['bones'][:,:3,3]-x['body_bones'][:,:3,3],axis=1))>1e-5:
        raise ValueError('realization_changed_joint_positions')
    hand=a.output_dir/'unapproved-hand-planning.fbx';shutil.copyfile(probe,hand)
    bake_hands(hand,{'left':a.hand_left,'right':a.hand_right})
    x.update(body_bones=body['bones'],hand_bones=snapshot(hand)['bones'],
        realized_body_sha256=sha(rig),proposal_sha256=sha(a.proposal),hand_left=a.hand_left,hand_right=a.hand_right)
    np.savez(a.output_dir/'realized-pose.npz',**x)
    (a.output_dir/'realization.json').write_text(json.dumps(dict(pose_audit=audit,
        realized_body_sha256=sha(rig),proposal_sha256=sha(a.proposal),old_mesh_adopted=False,final_fbx_certified=False),indent=2))
if __name__=='__main__':main()
