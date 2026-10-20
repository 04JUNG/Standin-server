"""Paired full-search vs full-first part references using frozen real RTM coords.

No VLM calls or generated query coordinates. Does not create a synthetic BVH.
Snapshot metadata provenance must be retained; archived final boxes are not a
replacement for original VLM boxes in a new end-to-end accuracy evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.repo import load_entries
from src.pose_quarantine import load_pose_quarantine
from src.config import CFG
from src.experimental.geometric_parts import PartGeometryIndex,full_first_supplement


def run(snapshot,index):
    cases=[];statuses=Counter();part_counts=Counter();calls=0
    for c in snapshot['cases']:
        case={**c,'people':[]}
        for p in c['people']:
            full=p['baseline']
            supplemented=full_first_supplement(index,full,p['keypoints'],p['joint_mask'],p['bone_mask'],
                full_trusted=p['confidence']=='high',owner_box=p['owner_box'],peer_boxes=p['peer_boxes'],
                image_size=c['size'],policy_block=p['policy_block'],suspect_limbs=p['suspect_limbs'],
                quarantined=load_pose_quarantine(CFG))
            case['people'].append({**p,'supplement':supplemented})
            statuses[supplemented['status']]+=1;calls+=supplemented['part_search_calls']
            for part,row in supplemented['parts'].items():
                if row['hits']:part_counts[part]+=1
        cases.append(case)
    people=[p for c in cases for p in c['people']]
    summary={'cases':len(cases),'people':len(people),'statuses':dict(statuses),'part_reference_people':dict(part_counts),
        'part_search_calls':calls,'full_candidates_preserved':all(p['baseline']==p['supplement']['full_candidates'] for p in people),
        'full_candidate_people_before':sum(bool(p['baseline']) for p in people),
        'any_reference_people_after':sum(bool(p['baseline']) or any(row['hits'] for row in p['supplement']['parts'].values()) for p in people),
        'synthetic_bvh_created':0,'extra_vlm_calls':0,'release_allowed':False}
    return {**snapshot,'evaluation_kind':'paired-frozen-real-RTM-full-first-parts','cases':cases,'summary':summary}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','db','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();snapshot=json.loads(args.snapshot.read_text())
    data=run(snapshot,PartGeometryIndex(load_entries(str(args.db))))
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps(data['summary'],ensure_ascii=False,indent=2))
    for c in data['cases']:
        for p in c['people']:
            print(c['test_id'],p['index'],p['supplement']['status'],
                {part:{'decision':r['decision'],'reason':r['evidence']['reasons'],
                       'top':r['hits'][0]['pose_id'] if r['hits'] else None} for part,r in p['supplement']['parts'].items()})
