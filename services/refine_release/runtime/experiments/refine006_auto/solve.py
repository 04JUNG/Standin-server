"""Generic original-rig solve with image-derived gesture/depth and develop XY. No oracle imports."""
import argparse,json,hashlib,sys,copy
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageOps,ImageDraw
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from validity import shadow, check_world

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_new'))
from skin_guard import refine as refine_skin
from core import prepare,frame,fk,targets,skin,deformation,render_projected,shoulder_stage

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refine_unified'))
from solve_budget import solve as budget_solve, eligible
def least_squares(fun,x0,*args,**kwargs):
 return budget_solve(fun,x0,*args,stage='depth',**kwargs)

def chest_yaw_prior(torso_facts,head_yaw):
    return float(np.clip(head_yaw*.25,-20,20)) if torso_facts.get('view')=='front' else 0.

def solve(assets,fixture,obs,semantic,out,lateral_bias=0.):
    if semantic['image_sha256']!=fixture['files']['rough']['sha256'] or obs['image_sha256']!=semantic['image_sha256']:raise ValueError('Image mismatch')
    if not obs.get('producer','').startswith('develop HumanArt'):raise ValueError('Develop extractor required')
    for item in fixture['files'].values():
        if hashlib.sha256((assets/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Asset mismatch')
    facts=semantic['raw']
    if facts.get('person_count')!=1:raise ValueError('Single person required')
    rig=json.loads((assets/fixture['files']['rig']['path']).read_text());names,ix,parents,local,base=prepare(rig);_,origin,torso,basis=frame(rig['bones']);goal,kp,valid=targets(obs,fixture['image_size'])
    if not all(n in goal for n in ['LeftArm','RightArm','LeftUpLeg','RightUpLeg']):raise ValueError('Insufficient torso anchors')
    side_order=sorted(['Left','Right'],key=lambda s:goal[s+'Arm'][0]);arm_map=dict(zip(['screen_left','screen_right'],side_order));sweeps={}
    for arm in facts.get('arms',[]):
        side=arm_map.get(arm.get('shoulder_position'))
        if side and arm.get('sweep') in ['lateral','away','toward'] and arm.get('confidence')=='clear':sweeps[side]=arm['sweep']
    raised=None;leg=facts.get('raised_leg',{})
    if all(s+'Foot' in goal for s in ['Left','Right']):
        foot_order=sorted(['Left','Right'],key=lambda s:goal[s+'Foot'][0]);side=dict(zip(['screen_left','screen_right'],foot_order)).get(leg.get('ankle_position'))
        if side and goal[side+'Foot'][1]<goal[('Right' if side=='Left' else 'Left')+'Foot'][1]-20:raised=side
    head=facts.get('head',{});head_yaw={'profile':75,'oblique':30,'front':0}.get(head.get('turn'),0)*(-1 if head.get('direction')=='screen_left' else 1)
    if head.get('confidence')!='clear':head_yaw=0
    active=['Spine','Spine1','Spine2']+[s+n for s in ['Right','Left'] for n in ['Arm','ForeArm','UpLeg','Leg','Foot']]+['Neck']
    limits=np.repeat(np.radians([15,15,15]+[90,100,90,110,85]*2+[35]),3);n=len(limits);ids=[ix[k] for k in goal];tar=np.array(list(goal.values()))
    def calc(x):
        w=fk(local,parents,[ix[k] for k in active],x[:n]);cam=basis@Rotation.from_rotvec(x[n:n+3]).as_matrix();scale=np.exp(x[n+3]);off=x[n+4:];z=(w[:,:3,3]-origin)@cam;return w,cam,scale,off,z,z[:,:2]*[scale,-scale]+off
    def depth_loss(w,z):
        res=[]
        for side,sweep in sweeps.items():
            a,e,h=[ix[side+x] for x in ['Arm','ForeArm','Hand']];length=np.linalg.norm(w[e,:3,3]-w[a,:3,3])+np.linalg.norm(w[h,:3,3]-w[e,:3,3]);de=(z[e,2]-z[a,2])/length;dh=(z[h,2]-z[a,2])/length
            # Lateral gesture keeps the arm near the shoulder plane; never interpret palm orientation as forward depth.
            center={'lateral':lateral_bias,'away':-.25,'toward':.25}[sweep]
            res.extend([(de-center*.5)*4,(dh-center)*6])
        for side in ['Left','Right']:
            hip,knee,foot=[ix[side+x] for x in ['UpLeg','Leg','Foot']];shin=np.linalg.norm(w[foot,:3,3]-w[knee,:3,3])
            if side==raised and leg.get('bend')=='backward':res.extend([((z[foot,2]-z[knee,2])/shin+.8)*8,((z[knee,2]-z[hip,2])/torso)*2])
            elif side!=raised and side+'Foot' in goal:res.append((z[foot,2]-z[hip,2])/torso*2)
        return res
    def residual(x):
        w,c,s,o,z,xy=calc(x);return np.r_[(xy[ids]-tar).ravel()/15,x[:n]*1.2,x[n:n+3]*2,depth_loss(w,z)]
    shoulder_span=np.linalg.norm(kp[5]-kp[6]);world_span=np.linalg.norm(base[ix['LeftArm'],:3,3]-base[ix['RightArm'],:3,3]);scale=np.clip(shoulder_span/world_span,200,800);hip=(goal['LeftUpLeg']+goal['RightUpLeg'])/2
    x=np.r_[np.zeros(n+3),np.log(scale),hip];lo=np.r_[-limits,[-.7,-1.2,-.7],np.log(150),[-150,-100]];hi=np.r_[limits,[.7,1.2,.7],np.log(1500),[700,1000]]
    r=least_squares(residual,np.clip(x,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=400)
    if not eligible(r) and not shadow():raise RuntimeError(r.message)
    w,cam,scale,off,z,xy=calc(r.x);ed={k:np.degrees(r.x[3*i:3*i+3]).tolist() for i,k in enumerate(active)}
    def world(e):return fk(local,parents,[ix[k] for k in e],np.radians(list(e.values())).ravel())
    def orient(e,name,desired):
        ww=world(e);old=Rotation.from_rotvec(np.radians(e.get(name,[0,0,0]))).as_matrix();e[name]=np.degrees(Rotation.from_matrix(old@np.linalg.solve(ww[ix[name],:3,:3],desired)).as_rotvec()).tolist()
    # Frontal is an angular category, not exactly zero yaw. Bounded torso co-rotation must not jump when confidence changes.
    torso_facts=facts.get('torso',{});chest_yaw=chest_yaw_prior(torso_facts,head_yaw)
    before=w.copy();orient(ed,'Hips',cam@Rotation.from_euler('y',chest_yaw*.3,degrees=True).as_matrix())
    for name in ['Spine','RightUpLeg','LeftUpLeg']:orient(ed,name,before[ix[name],:3,:3])
    for name,fraction in [('Spine',.2),('Spine1',.6),('Spine2',1)]:orient(ed,name,cam@Rotation.from_euler('y',chest_yaw*fraction,degrees=True).as_matrix())
    for side in ['Left','Right']:
        for joint in ['Shoulder','Arm','ForeArm']:orient(ed,side+joint,before[ix[side+joint],:3,:3])
    orient(ed,'Neck',cam@Rotation.from_euler('y',head_yaw*.65,degrees=True).as_matrix());orient(ed,'Head',cam@Rotation.from_euler('y',head_yaw,degrees=True).as_matrix())
    # Reset clavicles BEFORE projected/depth fitting; never chase unreachable rolled-shoulder 3D targets.
    _,ed,initial_shoulder=shoulder_stage(local,parents,ix,world(ed),ed)
    # Refit limbs after parent alignment, holding camera, pelvis, chest and gaze fixed.
    moving=[s+n for s in ['Right','Left'] for n in ['Shoulder','Arm','ForeArm','UpLeg','Leg']];start=np.radians([ed.get(k,[0,0,0]) for k in moving]).ravel()
    def candidate(v):
        e=copy.deepcopy(ed);e.update({k:np.degrees(v[i*3:i*3+3]).tolist() for i,k in enumerate(moving)});return e,world(e)
    def loss(v):
        e,ww=candidate(v);zz=(ww[:,:3,3]-origin)@cam;uv=zz[:,:2]*[scale,-scale]+off
        return np.r_[(uv[ids]-tar).ravel()/15,(v-start)*.6,depth_loss(ww,zz)]
    lim=np.repeat(np.radians([12,180,180,120,150]*2),3);r2=least_squares(loss,np.clip(start,-lim+1e-8,lim-1e-8),bounds=(-lim,lim),max_nfev=350)
    if not eligible(r2) and not shadow():raise RuntimeError(r2.message)
    ed,w=candidate(r2.x)
    for side in ['Left','Right']:
        if side==raised and leg.get('toe_direction')=='down':
            yy=np.array([0.,-.8,-.6]) if leg.get('bend')=='backward' else np.array([0.,-1.,0.]);xx=np.array([-1.,0.,0.]);orient(ed,side+'Foot',cam@np.column_stack([xx,yy,np.cross(xx,yy)]))
    w=world(ed);shoulderdiag=dict(initial_shoulder,reset_before_depth_fit=True)
    pr=dict(edits=ed,camera=cam.tolist(),origin=origin.tolist(),pixel_scale=float(scale),offset=off.tolist(),source=str((assets/fixture['files']['source']['path']).resolve()),targets={k:v.tolist() for k,v in goal.items()})
    with np.load(assets/fixture['files']['mesh']['path']) as m:v,f,weights=m['vertices'],m['faces'],m['weights']
    ww,ee,skin_diag=refine_skin(local,parents,base,ix,names,w,ed,v,f,weights,torso)
    skin_diag.update(shadow_only=shadow(),before_deformation=deformation(skin(v,weights,w,base),v,f),proposal_deformation=deformation(skin(v,weights,ww,base),v,f))
    if not shadow():w,ed=ww,ee
    check_world(w,base,parents);pr['edits']=ed
    vv=skin(v,weights,w,base);cv=(vv-origin)@cam;xy=((w[:,:3,3]-origin)@cam)[:,:2]*[scale,-scale]+off
    rough=ImageOps.pad(Image.open(assets/fixture['files']['rough']['path']).convert('RGB'),(535,687),color='white');im=render_projected(cv,f,cv[:,:2]*[scale,-scale]+off,(535,687));out.mkdir(parents=True,exist_ok=False);im.save(out/'adjusted.png')
    sheet=Image.new('RGB',(1070,722),'white');draw=ImageDraw.Draw(sheet)
    for i,(img,title) in enumerate([(rough,'ROUGH'),(im,'AUTOMATIC BODY + DEPTH')]):sheet.paste(img,(535*i,35));draw.text((535*i+8,10),title,fill='black')
    sheet.save(out/'comparison.png');(out/'adjustment.json').write_text(json.dumps(pr,indent=2));np.savez_compressed(out/'expected-world.npz',names=names,world=w)
    metrics=dict(case_id=fixture['case_id'],files=fixture['files'],manual_targets=False,manual_conditions_used=False,reference_used_for_fit=False,production_enabled=False,solver_success=bool(r.success and r2.success),convergence=dict(initial=bool(r.success),refit=bool(r2.success),initial_evaluations=r.nfev,refit_evaluations=r2.nfev),rmse_px=float(np.sqrt(np.mean((xy[ids]-tar)**2))),bound_hit_bones=[],gesture=facts.get('gesture'),arm_lengths={s:float(np.linalg.norm(w[ix[s+'ForeArm'],:3,3]-w[ix[s+'Arm'],:3,3])+np.linalg.norm(w[ix[s+'Hand'],:3,3]-w[ix[s+'ForeArm'],:3,3])) for s in ['Left','Right']},lateral_depth_hypothesis=lateral_bias,sweeps=sweeps,raised_leg=raised,head_yaw_prior=float(head_yaw),chest_yaw_prior=float(chest_yaw),skin_refinement=skin_diag,shoulder=shoulderdiag,deformation=deformation(vv,v,f),wrist_depth={s:float(((w[ix[s+'Hand'],:3,3]-w[ix[s+'Arm'],:3,3])@cam)[2]) for s in ['Left','Right']},max_bone_length_error=float(max(abs(np.linalg.norm(w[i,:3,3]-w[j,:3,3])-np.linalg.norm(base[i,:3,3]-base[j,:3,3])) for i,j in enumerate(parents) if j>=0)))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'camera.json').write_text(json.dumps(dict(camera=pr['camera'],origin=pr['origin'],projection='orthographic',inverse_distance=0,pixel_scale=pr['pixel_scale'],offset=pr['offset'],anchor_camera_xyz=((w[ix['Hips'],:3,3]-origin)@cam).tolist()),indent=2));return metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--semantics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--lateral-bias',type=float,default=0.);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for cid in a.cases:
  fixture=json.loads((a.baseline/cid/'fixture.json').read_text());fixture={k:v for k,v in fixture.items() if k in ['case_id','files','image_size']}
  m=solve(a.assets,fixture,json.loads((a.observations/f'{cid}.json').read_text()),json.loads((a.semantics/f'{cid}.json').read_text()),a.out/cid/'fit',a.lateral_bias);rows.append(dict(case_id=cid,status='fit'));print(cid,m,flush=True)
 (a.out/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
