"""Neutral identity, body-control and posed FBX round-trip checks in Blender."""
import argparse,importlib.util,json,math,sys,time
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Quaternion,Vector

def main(args):
    root=Path(args.root);folder=root/'authoring'/args.gender
    bpy.ops.wm.open_mainfile(filepath=str(folder/'model.blend'))
    spec=importlib.util.spec_from_file_location('verify_controller',folder/'standin_body.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.register()
    m,a=c.objects();base=c.source()['co'];values=c.normalize(c.defaults(m));floor=base[:,2].min()
    out=c.compute(values)[0];expected=base.copy();expected[:,2]-=floor
    error=float(np.max(np.linalg.norm(out-expected,axis=1)))
    if error>1e-5:raise ValueError(f'default changed approved geometry by {error*1000} mm')
    cases={'base':{},'muscle':{'muscle':1.5},'slim':{'physique':-.7},'soft':{'physique':1.0},
           'proportions':{'height_cm':180.,'heads':7.,'shoulders':.35,'arm_length':.3,'leg_length':.3},
           'details':{'neck_length':-.2,'waist':-.25,'pelvis':.3,'glute':.3,'calf':.2}}
    report={'gender':args.gender,'default_error_mm':error*1000,'controls':values,'cases':{}}
    for name,edits in cases.items():
        t=time.time();v={**values,**edits};metric=c.validate_geometry(v);report['cases'][name]=metric
        print('CASE',args.gender,name,round(time.time()-t,2),json.dumps(metric),flush=True)
    c.set_controls(values,correct=False)
    neutral=c.export_fixed(str(folder/'qa-exports'))
    report['neutral']=json.loads((neutral/'manifest.json').read_text(encoding='utf-8'))
    for name,raise_angle,thigh,knee in [('raise',-68,0,0),('sit',60,90,90)]:
        for b in a.pose.bones:b.matrix_basis=Matrix.Identity(4);b.rotation_mode='QUATERNION'
        for side,sign in [('Left',1),('Right',-1)]:
            c.rotate(a,side+'Arm',(0,1,0),sign*raise_angle)
            if thigh:c.rotate(a,side+'UpLeg',(-1,0,0),thigh)
            if knee:c.rotate(a,side+'Leg',(1,0,0),knee)
        c.correct_pose();posed=c.export_fixed(str(folder/'qa-exports'),posed=True)
        record=json.loads((posed/'manifest.json').read_text(encoding='utf-8'))
        record['helper_lift']={s:float(a.pose.bones['Standin_'+s+'ScapularLift'].location.length) for s in ('Left','Right')}
        report[name]=record;print('POSE',args.gender,name,record['roundtrip_max_error_mm'],record['helper_lift'],flush=True)
    (folder/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS',args.gender,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gender',required=True)
    main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
