"""Separate Blender process: inspect final saved native bytes and export preview meshes."""
import bpy,json,sys
from pathlib import Path

def main():
    run=Path(sys.argv[sys.argv.index('--')+1]);repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
    from converter.mesh_validation import extract_evaluated_skin,audit_mesh
    from experiments.refine_v5.release_gate import sha
    rows=json.loads((run/'native-audit/records.json').read_text());rigs={r['case_id']:r for r in json.loads((run/'native-data/records.json').read_text()) if r['status']=='ready'}
    records=[];preview={};baseline_audits={}
    for row in rows:
        case=row['case_id']
        if sha(row['artifact'])!=row['sha256']:raise ValueError('saved artifact changed')
        for kind,path in [('base',rigs[case]['base_fbx']),('candidate',row['artifact'])]:
            if kind=='base' and case in baseline_audits:continue
            bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(path),ignore_leaf_bones=False)
            arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];v,f=extract_evaluated_skin(meshes)
            bones=[dict(name=b.name.split(':')[-1],parent=b.parent.name.split(':')[-1] if b.parent else None,matrix=[list(x) for x in arm.matrix_world@b.matrix],head=list(arm.matrix_world@b.head),tail=list(arm.matrix_world@b.tail)) for b in arm.pose.bones]
            key=case+'.'+kind+('.r'+str(row['rounds']) if kind=='candidate' else '')
            # Small enough for local review, all evaluated mesh triangles retained.
            preview[key]=dict(vertices=v,faces=f,bones=bones)
            if kind=='base':baseline_audits[case]=audit_mesh(v,f,seconds=2.)
            else:
                if len(bones)!=row['preserved_bones']:raise ValueError('saved bone inventory changed')
                records.append(dict(case_id=case,rounds=row['rounds'],bones=bones,sha256=sha(path),safe_fallback_sha256=sha(row['release']['path']),source_sha256=sha(rigs[case]['base_fbx']),safe_is_exact_source=sha(row['release']['path'])==sha(rigs[case]['base_fbx'])))
    (run/'native-audit/independent-readback.json').write_text(json.dumps(dict(records=records,baseline_full_skin=baseline_audits),indent=2))
    (run/'native-audit/preview-meshes.json').write_text(json.dumps(preview))
    print('V5_INDEPENDENT_READBACK',len(records))
if __name__=='__main__':main()
