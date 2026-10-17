"""Replay saved actual Gemini descriptions: critical search -> empty -> generate.

No new VLM calls, no query joints, no library mutation. Export research previews
only, and round-trip each BVH before treating it as a valid preview.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.bvh import parse_bvh, coco17_from_fk, fk, hierarchy_text
from src.experimental.part_composition import torso_frame
from src.experimental.text_pose_generation import TextConstraintPoseGenerator, after_no_result, compile_plan, validate_generated
from src.search_routing import PoseDescription
from src.semantic_search import SemanticQuery, TagSemanticSearcher
from src.semantic_body_index import load_body_index
from src.semantic_shapes import load_shape_index, rotate_for_camera
from src.thumbnail_renderer import render_thumbnail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','tags','body-index','shape-index','rig','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--attribution',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    original=json.loads(args.reference.read_text());raw=json.loads(args.tags.read_text())
    records=raw['poses'] if isinstance(raw,dict) else raw
    searcher=TagSemanticSearcher(records,body_metadata=load_body_index(args.body_index,records),
                                shape_metadata=load_shape_index(args.shape_index,records))
    generator=TextConstraintPoseGenerator(args.rig)
    sha=hashlib.sha256(args.rig.read_bytes()).hexdigest();cases=[];statuses=Counter();roundtrips=[]
    for case in original['cases']:
        row={'test_id':case['test_id'],'image':case['image'],'people':[]}
        for p in case['people']:
            routing=p.get('routing') or {};d=(routing.get('semantic_evidence') or {}).get('description')
            person={'index':p['index'],'previous_route':routing.get('route'),'description':d}
            if routing.get('route')!='semantic' or d is None:
                person['generation']={'status':'not_requested','reason':'not_a_valid_semantic_description'}
            else:
                description=PoseDescription(**d);search=searcher.search(SemanticQuery.from_description(description),5)
                person['search']={'status':search.status,'reason':search.reason,'hits':[h.to_dict() for h in search.hits],'backend':search.backend}
                result=after_no_result(generator,description,search_status=search.status,hit_count=len(search.hits),
                                       policy_block=bool((routing.get('joint_evidence') or {}).get('policy_block')))
                person['generation']=result.to_dict()
                for candidate in person['generation']['candidates']:
                    artifact=args.out/(candidate['pose_id']+'.bvh');artifact.write_text(candidate['bvh'])
                    j,f=parse_bvh(str(artifact));p3,s=coco17_from_fk(j,fk(j,f[0]));plan=compile_plan(description)
                    check=validate_generated(j,f[0],plan)
                    scale=float(np.linalg.norm((p3[5]+p3[6]-p3[11]-p3[12])*.5))
                    delta=float(np.max(np.linalg.norm(p3-np.asarray(candidate['keypoints_3d']),axis=1)))/scale
                    passed=check['kinematic_probe_pass'] and delta<=1e-6 and hierarchy_text(str(artifact))==hierarchy_text(str(args.rig))
                    candidate['roundtrip']={'pass':passed,'max_joint_delta_torso':delta,'diagnostics':check}
                    roundtrips.append(passed);candidate['bvh_path']=str(artifact)
                    if not passed:
                        person['generation']['status']='rejected';person['generation']['reason']='serialization_validation_failed'
                    axes=torso_frame(j,f[0]);candidate['renders']={}
                    for view,yaw in (('front',0),('three_quarter',45),('opposite_three_quarter',-45),('side',90),('back',180)):
                        file=args.out/(candidate['pose_id']+'-'+view+'.png')
                        render_thumbnail(rotate_for_camera(p3@axes,{'yaw_deg':yaw,'pitch_deg':0,'roll_deg':0}),s,0,320).save(file)
                        candidate['renders'][view]=str(file)
            statuses[person['generation']['status']]+=1;row['people'].append(person)
        cases.append(row)
    summary={'cases':len(cases),'people':sum(len(c['people']) for c in cases),'generation_statuses':dict(statuses),
             'generated_candidates':len(roundtrips),'roundtrip_pass':sum(roundtrips),'extra_vlm_calls':0,
             'source_rig_unchanged':sha==hashlib.sha256(args.rig.read_bytes()).hexdigest(),'library_modified':False,
             'actual_mesh_validated':False,'release_allowed':False}
    data={'evaluation_kind':'saved_real_Gemini_description_critical_search_to_text_IK_generation',
          'description_source':str(args.reference),'rig_source':str(args.rig),'rig_sha256':sha,
          'motion_source':'neutral_zero_channels_not_library_motion','summary':summary,'cases':cases,
          'limitations':['not_new_Gemini_E2E','no_new_skeleton_extraction','synthetic_variants_not_accuracy_samples','not_actual_skin_mesh']}
    (args.out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    if args.attribution:shutil.copy2(args.attribution,args.out/'ATTRIBUTION.md')
    report=['# 자연어 조건 → 새 BVH 생성 실험 (2026-10-14)','',
            '저장된 실제 Gemini 설명을 고정해 검색→후보 없음→생성을 비교. 새 VLM 호출 없음.',
            '기존 포즈 motion/팔 donor를 쓰지 않고 명시한 rig의 HIERARCHY/OFFSET만 재사용.',
            '구조 검사 통과는 피부·손목 자연스러움·3D 정확도를 보증하지 않음. 자동 적용/refine/release 금지.','',
            '| 러프 | 인물 | 검색 상태 | 생성 | 후보 수 |','|---|---|---|---|---|']
    for c in cases:
        for p in c['people']:report.append(f"| {c['test_id']} | {p['index']+1} | {p.get('search',{}).get('status',p['previous_route'])} | {p['generation']['status']} | {len(p['generation'].get('candidates',[]))} |")
    report+=['','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
             '저장된 입력: '+str(args.reference),'생성물은 같은 조건의 변형이며 독립적인 성공 표본이 아니다.']
    (args.out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary['source_rig_unchanged'] and all(roundtrips) else 1


if __name__=='__main__':raise SystemExit(main())
