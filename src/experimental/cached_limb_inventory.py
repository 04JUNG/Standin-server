"""Match frozen people to cached observations without person-index guesses.

Only image hashes, source boxes, frozen person IDs and selected asset hashes
are used. No grades, expected poses or correction answers enter diagnosis.
"""
import math


def box_iou(a,b):
    if not isinstance(a,(list,tuple)) or not isinstance(b,(list,tuple)) or len(a)!=4 or len(b)!=4:return 0.
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*a,*b)):return 0.
    if a[2]<=a[0] or a[3]<=a[1] or b[2]<=b[0] or b[3]<=b[1]:return 0.
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def match_cached_people(case,people,*,min_iou=.8,min_margin=.15):
    """Require unique box evidence in multi-person cuts, independent of order."""
    ids=case['expected_person_ids'];boxes={r['reference_person_id']:r['box'] for r in case['source_person_boxes']}
    result={};proposed={}
    for pid in ids:
        if len(ids)==1 and len(people)==1:
            proposed[pid]={'cache_person_index':0,'source':'unique_single_person_in_same_hashed_image',
                'box_iou':box_iou(boxes.get(pid),people[0].get('box'))};continue
        scores=sorted(((box_iou(boxes.get(pid),p.get('box')),i) for i,p in enumerate(people)),reverse=True)
        best,index=scores[0] if scores else (0.,None);second=scores[1][0] if len(scores)>1 else 0.
        if best<min_iou or best-second<min_margin:
            result[pid]={'status':'held','reason':'cached_person_mapping_missing_or_ambiguous',
                'best_iou':best,'runner_up_iou':second};continue
        proposed[pid]={'cache_person_index':index,'source':'unique_source_box_iou_not_person_index',
                       'box_iou':best,'runner_up_iou':second}
    for pid,mapping in proposed.items():
        if sum(m['cache_person_index']==mapping['cache_person_index'] for m in proposed.values())>1:
            result[pid]={'status':'held','reason':'one_cached_person_matches_multiple_frozen_people'}
        else:result[pid]={'status':'matched',**mapping}
    return result
