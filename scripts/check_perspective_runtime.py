"""Read-only real-rough replay and runtime/frozen-policy parity, not accuracy.

Uses cached measured XY, never calls paid APIs/models or reads answer IDs.
All outputs go to a new ignored directory. Original inputs remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import normalize_skeleton
from src.perspective_search import PerspectiveSearchIndex
from src.schema import LibraryEntry, View
from src.search import PositionSearchIndex, pose_family_id


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_reference(index, xy, mask, image):
    from standin_eval.perspective_search import normalize_batch, build_camera_index, camera_grid, Camera, score_camera_index
    from standin_eval.camera_refinement import normalized_joints, refine_costs
    from standin_eval.dual_label_search import alternate_labels
    from standin_eval.contour_letterbox import prepare_letterboxed_query
    from standin_eval.contour_rerank import candidate_evidence
    grid = camera_grid()
    cameras = grid['yaw_pitch_perspective_117'] + [Camera(c.yaw,c.pitch,d) for c in grid['yaw_pitch_39'] for d in (2.,3.)]
    full = build_camera_index(index.poses, cameras)
    coarse = tuple(a[:117] for a in full)
    q = normalize_batch(xy); alt, am = alternate_labels(q, mask)
    base = np.minimum(score_camera_index(coarse,q,mask), score_camera_index(coarse,alt,am))
    fit, params = refine_costs(normalized_joints(np.stack([p.joints for p in index.poses])),full,cameras,q,mask)
    scores = .25*base.astype(float)+.75*fit.astype(float)
    first = int(np.argmin(base)); order = np.argsort(scores,kind='stable')
    order = np.concatenate(([first],order[order!=first])); short = order[:20]
    penalties = np.zeros(len(short)); fallback = False
    try:
        query, _ = prepare_letterboxed_query(image, xy)
        for k,j in enumerate(short):
            penalties[k] = candidate_evidence(query,index.poses[j].joints,params[j])[0]
    except ValueError:
        penalties[:] = 0; fallback = True
    head = short[np.argsort(scores[short]+penalties,kind='stable')]
    order = np.concatenate((head,order[len(short):])); scores[short] += penalties
    seen, result = set(), []
    for i in order:
        entry = index.poses[i].entry; family = pose_family_id(entry.pose_id,entry.meta)
        if family not in seen:
            seen.add(family); result.append((entry.pose_id,float(scores[i])))
        if len(result)==5: break
    return result, fallback


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    db = a.library_root/'poses.db'
    before = {str(db):digest(db), str(a.review_root/'protocol.json'):digest(a.review_root/'protocol.json')}
    # Do not call src.repo.load_entries: its connector creates schema/indexes.
    con = sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute('SELECT p.*,pr.view,pr.feature_blob,pr.feature_version FROM poses p JOIN pose_projections pr USING(pose_id)').fetchall()
    finally: con.close()
    entries=[]
    for r in rows:
        assert r['feature_version']==1
        stored=Path(r['bvh_path']); path=a.library_root/'bvh'/stored.name
        if not path.is_file(): path=stored
        before[str(path)]=digest(path)
        entries.append(LibraryEntry(r['pose_id'],View(r['view']),np.frombuffer(r['feature_blob'],np.float32).copy(),
                                   {k:r[k] for k in ('shot','action','relationship')},str(path),json.loads(r['meta_json'] or '{}')))
    start=perf_counter(); index=PerspectiveSearchIndex.build(entries); build_ms=(perf_counter()-start)*1000
    legacy=PositionSearchIndex.build(entries)
    policy=json.loads((a.review_root/'protocol.json').read_text())
    records=[]
    for case in policy['cases']:
        path=a.review_root/'predictions'/(case['id']+'.json'); before[str(path)]=digest(path)
        pred=json.loads(path.read_text()); record={'id':case['id']}
        if pred.get('error'):
            record['error']=pred['error']; records.append(record); continue
        xy=np.asarray(pred['keypoints2d'],np.float32); mask=np.asarray(pred['scores'])>=.3
        if mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
            record['error']='original observed-joint gate failed'; records.append(record); continue
        path=Path(case['path']); before[str(path)]=digest(path)
        assert before[str(path)]==case['sha256']
        with Image.open(path) as im: image=np.asarray(im.convert('RGB'))
        started=perf_counter()
        old=legacy.search(normalize_skeleton(xy,valid_mask=mask),top_k=5,query_valid_mask=mask)
        record['legacy_ms']=(perf_counter()-started)*1000
        new, trace=index.search(xy,mask,image)
        reference, fallback=frozen_reference(index,xy,mask,image)
        assert [c.pose_id for c in new]==[p for p,s in reference]
        assert np.allclose([c.distance for c in new],[s for p,s in reference],atol=1e-7,rtol=0)
        assert (trace['contour']=='camera_only_fallback')==fallback
        record.update(before=[c.pose_id for c in old],after=[c.pose_id for c in new],trace=trace,frozen_policy_parity=True)
        records.append(record)
        print(case['id'],'runtime/frozen parity OK',flush=True)
    assert all(digest(Path(p))==h for p,h in before.items())
    searchable=[r for r in records if not r.get('error')]
    result={'stage':'cached measured XY component replay, not fresh accuracy or live /analyze benchmark',
            'gallery':len(index.poses),'inputs':len(records),'searchable':len(searchable),
            'original_failures':len(records)-len(searchable),'accuracy':None,
            'build_ms':build_ms,'runtime_search_median_ms':float(np.median([r['trace']['elapsed_ms'] for r in searchable])),
            'legacy_median_ms':float(np.median([r['legacy_ms'] for r in searchable])),
            'parity_queries':sum(r.get('frozen_policy_parity',False) for r in records),
            'input_hashes':before,'records':records}
    (a.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print({k:v for k,v in result.items() if k not in ('records','input_hashes')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('library-root','review-root','out'): p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
