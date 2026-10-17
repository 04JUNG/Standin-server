"""Independent least-squares replay for partial-joint diagnostic, not GT scoring."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from scripts.review_perspective_roughs import preview
from standin_eval.perspective_search import Camera,project
from standin_eval.dual_label_search import LEFT_RIGHT


def read(p):return json.loads(p.read_text())


def audit(a):
    policy=read(a.study/'protocol.json')
    for p,h in policy['inputs'].items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    with np.load(a.gallery,allow_pickle=False) as d:ids=d['pose_ids'].tolist();joints=d['joints'].copy()
    pred=read(a.crop/'results.json')['confidence_selected'];q=np.asarray(pred['keypoints2d'],float)
    records=read(a.study/'results.json')['arms'];maximum=0;checks=0
    for name,r in records.items():
        path=a.study/(name+'-costs.npz');assert sha256(path)==r['costs_sha256']
        with np.load(path,allow_pickle=False) as d:cost,params,order,mask,swap=[d[k] for k in ('cost','parameters','order','mask','global_swap')]
        selected=np.flatnonzero(mask[5:])+5
        assert order.tolist()==sorted(range(len(ids)),key=lambda j:float(cost[j]))
        assert [ids[j] for j in order[:20]]==[c['pose_id'] for c in r['top20']]
        for j,p in enumerate(joints):
            yaw,pitch,inv=map(float,params[j]);cam=Camera(yaw,pitch,None if inv==0 else 1/inv)
            candidate,valid=project(p,cam)
            if not valid:assert np.isinf(cost[j]);continue
            query=(q[LEFT_RIGHT] if swap[j] else q)[selected]
            lib=candidate[selected].astype(float)
            design=np.zeros((len(selected)*2,3));design[0::2,0]=lib[:,0];design[1::2,0]=lib[:,1]
            design[0::2,1]=1;design[1::2,2]=1
            coefficients,_,rank,_=np.linalg.lstsq(design,query.reshape(-1),rcond=None)
            if rank<3 or coefficients[0]<=1e-8:assert np.isinf(cost[j]);continue
            estimate=(design@coefficients).reshape(-1,2)
            expected=np.linalg.norm(estimate-query)/np.linalg.norm(query-query.mean(0))
            difference=abs(float(expected)-float(cost[j]));assert difference<3e-5,(name,ids[j],difference)
            maximum=max(maximum,difference);checks+=1
        print('partial audit',name,'all1981 fitted cameras/ranks',flush=True)
    font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
    sheet=Image.new('RGB',(1280,730),'#f5f5f5');draw=ImageDraw.Draw(sheet)
    for method,y in (('upper6',10),('forearm4',370)):
        draw.text((12,y),method+' · 상체 관측 진단 · 정확도/전체 자세 미검증',font=font,fill='#202124')
        for k,c in enumerate(records[method]['top20'][:5]):
            im,kind,error=preview(joints[c['pose_index']],Camera(**c['camera']),size=230)
            x=8+k*254;sheet.paste(im,(x,y+28));draw.text((x,y+265),str(k+1)+' · '+c['pose_id'][:25],font=font,fill='#202124')
            if error:draw.text((x,y+290),'표면 실패 · 관절 투영',font=font,fill='#202124')
    sheet.save(a.study/'top5-contact-sheet.png')
    save_json(a.study/'audit.json',{'independent_fitted_camera_cost_replays':checks,'max_normalized_cost_difference':maximum,
        'top20_ids_and_orders_verified':True,'original_source_hashes_unchanged':True,
        'result_sha256':sha256(a.study/'results.json'),'diagnostic_only':True,'ground_truth_accuracy':None,
        'cross_subset_cost_comparison':'not valid: selected-joint count and query RMS differ',
        'current_user_review_changed':False})
    print('PASS',checks,'independent SVD fits, max difference',maximum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('study','gallery','crop'):p.add_argument('--'+n,type=Path,required=True)
    audit(p.parse_args())
