"""T013 frozen real-Gemini description: upper/lower search + offline BVH probe.

Preserves all originals and does not call VLM, write DB/index, or change runtime.
Compares naive Euler copy with body-frame arm-direction transport on seated bases.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.bvh import parse_bvh, fk, coco17_from_fk, channel_starts, write_single_frame_bvh, hierarchy_text
from src.experimental.part_composition import compose_arms, canonical, arm_mask, torso_frame
from src.search_routing import PoseDescription
from src.semantic_search import TagSemanticSearcher, SemanticQuery
from src.semantic_body_index import load_body_index, describe_library_body
from src.semantic_shapes import load_shape_index, relational_geometry, rotate_for_camera
from src.thumbnail_renderer import render_thumbnail


def seated_geometry(p,s):
    body=describe_library_body(p,s);m=body['metrics']
    angles=[m.get('left_knee_deg'),m.get('right_knee_deg')]
    slopes=[];shins=[]
    for hip,knee,ankle in ((11,13,15),(12,14,16)):
        vec=p[knee]-p[hip];leg=p[ankle]-p[knee]
        slopes.append(float(np.degrees(np.arcsin(np.clip(abs(vec[1])/max(np.linalg.norm(vec),1e-8),0,1)))))
        shins.append(float(-leg[1]/max(np.linalg.norm(leg),1e-8)))
    passed=all(a is not None and 70<=a<=115 for a in angles) and max(slopes)<=30 and min(shins)>=.65
    return {'chair_like_geometry':passed,'knees_deg':angles,'thigh_slope_deg':slopes,'shin_down_fraction':shins,
            'meaning':'chair-like geometry, not verified chair contact'}


def naive_copy(bj,bf,dj,df):
    source={canonical(j[0]):i for i,j in enumerate(dj)};bs=channel_starts(bj);ds=channel_starts(dj)
    out=bf.copy()
    for i in arm_mask(bj):
        si=source.get(canonical(bj[i][0]))
        if si is None:continue
        values={ch:df[ds[si]+k] for k,ch in enumerate(dj[si][3]) if ch.endswith('rotation')}
        for k,ch in enumerate(bj[i][3]):
            if ch in values:out[bs[i]+k]=values[ch]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','tags','body-index','shape-index','bvh-root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int,default=12)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not 1<=args.limit<=30:raise ValueError('limit must be 1..30')
    original=json.loads(args.results.read_text());case=next(c for c in original['cases'] if c['test_id']=='T013')
    desc=PoseDescription(**case['people'][1]['routing']['semantic_evidence']['description'])
    if 'fold_arms' not in desc.actions or 'sitting' not in desc.postures:raise ValueError('expected saved T013 seated fold description')
    raw=json.loads(args.tags.read_text());records=raw['poses'] if isinstance(raw,dict) else raw
    body=load_body_index(args.body_index,records);shapes=load_shape_index(args.shape_index,records)
    upper_query=SemanticQuery(actions=['fold_arms'],text='가슴 앞에서 양팔을 교차해 팔짱을 낀 자세')
    lower_text=desc.body_details[desc.body_details.index('허벅지'):] if '허벅지' in desc.body_details else '무릎을 굽혀 앉은 자세'
    lower_query=SemanticQuery(postures=['sitting'],text=lower_text,
        body_observations={k:v for k,v in desc.body_observations.items() if k in ('knees','legs','torso')},min_body_matches=1)
    upper=TagSemanticSearcher(records,body_metadata=body,shape_metadata=shapes).search(upper_query,3)
    seated=[r for r in records if any(t.get('field')=='posture' and t.get('value')=='sitting' for ts in r.get('groups',{}).values() for t in ts)]
    lower=TagSemanticSearcher(seated,body_metadata=body).search(lower_query,len(seated))
    if not upper.hits:raise RuntimeError('no verified fold donor')
    donor_id=upper.hits[0].pose_id;root=args.bvh_root.resolve()
    def source(pose_id):
        path=(root/(pose_id+'.bvh')).resolve()
        if path.parent!=root:raise ValueError('unsafe source ID')
        return path
    donor_path=source(donor_id);dj,df=parse_bvh(str(donor_path))
    hashes={str(donor_path):hashlib.sha256(donor_path.read_bytes()).hexdigest()}
    trials=[];skipped=[]
    for hit in lower.hits:
        path=source(hit.pose_id);bj,bf=parse_bvh(str(path));bp,bs=coco17_from_fk(bj,fk(bj,bf[0]))
        lower_check=seated_geometry(bp,bs)
        if not lower_check['chair_like_geometry']:
            skipped.append({'base':hit.pose_id,'reason':'not_chair_like_geometry','measurements':lower_check});continue
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        row={'base':hit.pose_id,'lower_search_score':hit.score,'lower_check':lower_check}
        try:
            naive=naive_copy(bj,bf[0],dj,df[0]);np_,ns=coco17_from_fk(bj,fk(bj,naive))
            row['naive_fold_geometry']=relational_geometry(np_,ns)['fold_geometry']
            result=compose_arms(bj,bf[0],dj,df[0]);row['diagnostics']=result.diagnostics
            cp,cs=coco17_from_fk(bj,fk(bj,result.frame));row['composed_lower_check']=seated_geometry(cp,cs)
            if not row['composed_lower_check']['chair_like_geometry']:row['diagnostics']['reasons'].append('seated_shape_lost');row['diagnostics']['kinematic_probe_pass']=False
            row['synthetic_id']='experiment_T013_fold_sitting_'+str(len(trials)+1).zfill(2)
            artifact=args.out/(row['synthetic_id']+'.bvh')
            if result.diagnostics['kinematic_probe_pass']:
                write_single_frame_bvh(str(path),result.frame,str(artifact));row['bvh']=str(artifact)
                rj,rf=parse_bvh(str(artifact));rp,rs=coco17_from_fk(rj,fk(rj,rf[0]))
                row['roundtrip']={'hierarchy_preserved':hierarchy_text(str(path))==hierarchy_text(str(artifact)),
                    'max_position_delta':float(np.max(np.linalg.norm(rp-cp,axis=1))),
                    'fold_geometry':relational_geometry(rp,rs)['fold_geometry']}
                original_positions=fk(bj,bf[0]); reloaded_positions=fk(rj,rf[0]);mask=arm_mask(bj)
                scale=float(np.linalg.norm((bp[5]+bp[6]-bp[11]-bp[12])*.5))
                row['roundtrip']['protected_position_delta_torso']=max(float(np.linalg.norm(reloaded_positions[i]-original_positions[i])) for i in original_positions if i not in mask)/scale
                row['roundtrip']['all_position_delta_torso']=row['roundtrip']['max_position_delta']/scale
                row['roundtrip']['export_check_pass']=(row['roundtrip']['hierarchy_preserved'] and row['roundtrip']['fold_geometry'] is True and row['roundtrip']['protected_position_delta_torso']<=1e-6 and row['roundtrip']['all_position_delta_torso']<=1e-6)
                if not row['roundtrip']['export_check_pass']:
                    row['diagnostics']['reasons'].append('serialization_check_failed');row['diagnostics']['kinematic_probe_pass']=False
            axes=torso_frame(bj,bf[0]);row['renders']={}
            for label,p,s in (('base',bp,bs),('naive',np_,ns),('composed',cp,cs)):
                row['renders'][label]={}
                for view,yaw in (('front',0),('three_quarter',45),('opposite_three_quarter',-45),('side',90),('back',180)):
                    camera={'yaw_deg':yaw,'pitch_deg':0,'roll_deg':0}
                    file=args.out/(row['synthetic_id']+'-'+label+'-'+view+'.png')
                    render_thumbnail(rotate_for_camera(p@axes,camera),s,0,320).save(file)
                    row['renders'][label][view]=str(file)
        except ValueError as error:row['error']=str(error)
        trials.append(row)
        if len(trials)>=args.limit:break
    dp,ds=coco17_from_fk(dj,fk(dj,df[0]));donor_axes=torso_frame(dj,df[0])
    donor_renders={}
    for view,yaw in (('front',0),('three_quarter',45),('side',90),('back',180)):
        file=args.out/('donor-'+view+'.png');render_thumbnail(rotate_for_camera(dp@donor_axes,{'yaw_deg':yaw,'pitch_deg':0,'roll_deg':0}),ds,0,320).save(file);donor_renders[view]=str(file)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    summary={'attempted':len(trials),'kinematic_probe_pass':sum(r.get('diagnostics',{}).get('kinematic_probe_pass',False) for r in trials),
        'naive_fold_geometry_pass':sum(r.get('naive_fold_geometry') is True for r in trials),
        'rejections':dict(Counter(reason for r in trials for reason in r.get('diagnostics',{}).get('reasons',[]))),
        'source_bvhs_unchanged':unchanged,'extra_vlm_calls':0,'release_allowed':False}
    summary['max_export_protected_delta_torso']=max((r.get('roundtrip',{}).get('protected_position_delta_torso',0.) for r in trials),default=0.)
    summary['max_bone_length_delta_torso']=max((r.get('diagnostics',{}).get('bone_length_delta_torso',0.) for r in trials),default=0.)
    summary['reloaded_export_pass']=sum(r.get('roundtrip',{}).get('export_check_pass',False) for r in trials)
    data={'evaluation_kind':'offline_single_rough_part_composition','description_source':str(args.results),
        'rough':case['image'],'person_index':1,'description':desc.__dict__,'upper_query':upper_query.to_dict(),
        'lower_query':lower_query.to_dict(),'split_kind':'analyst-scoped fold/sitting query split; not general learned parsing',
        'upper_hits':[h.to_dict() for h in upper.hits],'donor':donor_id,'donor_renders':donor_renders,
        'lower_pool':len(seated),'skipped_lower_candidates':skipped,'trials':trials,'summary':summary,
        'source_sha256':hashes,'limitations':['single T013 only','COCO17 mannequin, not actual skin mesh',
            'hand roll/joint limits/arm-arm skin contact not certified','no rough 3D GT','no API integration']}
    (args.out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if unchanged and summary['kinematic_probe_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
