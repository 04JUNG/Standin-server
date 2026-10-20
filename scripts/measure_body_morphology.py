#!/usr/bin/env python3
"""Read confirmed Blender sources and validate dimensionless measures against FBX.

Run: blender -b --python scripts/measure_body_morphology.py --
  --confirmation docs/BODY_SELECTION/CONFIRMED_MODELS.json
  --package <confirmed-model-assets package directory> --out <new measurements.json>
This offline command imports trusted, hash-pinned local authoring controllers. It
never saves the .blend, publishes assets, or grants runtime recommendation eligibility.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.morphology import VERSION,measure_neutral

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import bpy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.out.exists():raise FileExistsError(args.out)
    record=json.loads(args.confirmation.read_text());rows=[];surfaces={}
    # Reject all source hash mismatches before executing any controller.
    for model in record['models'].values():
        for key in ('model','controller'):
            if sha(model[key+'_path'])!=model[key+'_sha256']:raise ValueError('source_hash_mismatch')
    for family,model in record['models'].items():
        bpy.ops.wm.open_mainfile(filepath=model['model_path'])
        spec=importlib.util.spec_from_file_location('measure_'+family,model['controller_path'])
        c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.register()
        try:
            m,a=c.objects();s=c.source();base_lm=json.loads(m['landmarks'])
            for preset in c.PRESETS:
                ctrl=c.preset_values(preset);co=c.compute(ctrl)[0]
                lm={k:float(c.warp(np.array([[0,base_lm.get('head_y',0),base_lm[k]]]),ctrl,m)[0,2]) for k in ('chin','hip','shoulder','knee','ankle')}
                joints={b.name.split(':')[-1]:c.warp(np.array([s['rest'][b.name]['head']]),ctrl,m)[0].tolist() for b in a.data.bones}
                asset=next(r for r in model['neutral_fbx'] if r['preset']==preset)
                fbx=(args.package/asset['path']).resolve()
                if not fbx.is_relative_to(args.package.resolve()) or sha(fbx)!=asset['sha256']:raise ValueError('fbx_hash_mismatch')
                body_id=family+'-'+preset.lower().replace('_','-')+'-v'+model['version'].replace('.','-')
                row=dict(body_id=body_id,family=family,preset=preset,version=model['version'],display_name=c.PRESETS[preset][0],
                    model_sha256=model['model_sha256'],controller_sha256=model['controller_sha256'],fbx_sha256=asset['sha256'],
                    landmarks=lm,joints=joints,authored=ctrl,measurements=measure_neutral(co,lm,joints),_fbx=str(fbx))
                rows.append(row);surfaces[body_id]=co.copy()
        finally:c.unregister()
    for row in rows:
        bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=row.pop('_fbx'))
        meshes=[m for m in bpy.context.scene.objects if m.type=='MESH']
        if len(meshes)!=1:raise ValueError('expected_single_neutral_mesh')
        m=meshes[0];co=np.array([m.matrix_world@v.co for v in m.data.vertices]);base=surfaces[row['body_id']]
        if co.shape!=base.shape:raise ValueError('fbx_vertex_count_mismatch')
        error=float(np.linalg.norm(co-base,axis=1).max())
        measured=measure_neutral(co,row['landmarks'],row['joints'])
        ratio_error=max(abs(measured[k]-v) for k,v in row['measurements'].items())
        if error>=.00001 or ratio_error>=.0001:raise ValueError('fbx_morphology_mismatch')
        row['validation']=dict(fbx_roundtrip='passed',max_surface_error_mm=error*1000,max_ratio_error=ratio_error)
    output=dict(schema_version=VERSION,status='diagnostic_not_runtime_eligible',confirmation_sha256=sha(args.confirmation),assets=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(assets=len(rows),measurement_version=VERSION,status=output['status'])))

if __name__=='__main__': main()
