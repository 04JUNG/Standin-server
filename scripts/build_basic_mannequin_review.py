"""Presentation-only reuse of original library JPEGs; never rerun or rescore."""
import argparse
import base64
import copy
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json
from scripts.build_perspective_review import STANDALONE_STYLE
from src.bvh import load_coco17

VIEWS=('front','three_quarter','side','back')
INLINE_IDS=('rough_288','rough_315','rough_301','rough_291')


def strip_images(source,keep_ids=None):
    data=copy.deepcopy(source)
    if keep_ids is not None:
        index={q['id']:q for q in data['queries']};data['queries']=[index[pid] for pid in keep_ids]
        # Full10 screen retains extraction overlays; compact inline uses raw art.
        data['disable_overlay']=True
        for q in data['queries']:q.pop('overlay',None)
    for q in data['queries']:
        for arm in ('before','after'):
            for c in q[arm]:
                for field in ('matched','canonical','canonical_kind'):c.pop(field,None)
    data['source_study_id']=source['study_id'];data['study_id']='basic-mannequin-'+source['study_id']+('-focus4' if keep_ids else '-all10')
    data['default_query']='rough_288';data.pop('summary_label',None)
    return data


def assemble(source,assets,keep_ids=None):
    data=strip_images(source,keep_ids)
    selected={c['pose_id'] for q in data['queries'] for arm in ('before','after') for c in q[arm]}
    data['assets']={pid:assets[pid] for pid in sorted(selected)}
    return data


def run(a):
    template=ROOT/'scripts/templates/basic_mannequin_review.html'
    inputs=(a.source/'review-data-private.json',a.source/'rankings.json',a.source/'summary.json',a.gallery,a.database,a.thumbs/'thumbnail_manifest.json')
    hashes={str(p):sha256(p) for p in inputs}
    source=json.loads(inputs[0].read_text());summary=json.loads(inputs[2].read_text())
    assert hashes[str(inputs[1])]==summary['rankings_sha256']
    ids=sorted({c['pose_id'] for q in source['queries'] for arm in ('before','after') for c in q[arm]})
    manifest=json.loads((a.thumbs/'thumbnail_manifest.json').read_text());entries={(r['pose_id'],r['view']):r for r in manifest['results']}
    conn=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
    try:paths=dict(conn.execute('SELECT pose_id,bvh_path FROM poses').fetchall())
    finally:conn.close()
    with np.load(a.gallery,allow_pickle=False) as d:gallery=dict(zip(d['pose_ids'].tolist(),d['joints']))
    assert not a.inline.exists(),'new inline destination required; preserve existing reviews'
    a.out.mkdir(parents=True,exist_ok=False)
    code=('scripts/build_basic_mannequin_review.py','scripts/templates/basic_mannequin_review.html')
    policy={'stage':'display-only; all candidate IDs, order, fitted camera, features, masks and source images unchanged',
            'assets':'verbatim JPEG files with exact pose_id/view match and thumbnail manifest SHA verification; deduplicated by pose_id',
            'view':'native library front/three_quarter/side/back shared by both columns; NEVER labelled matched search camera',
            'inline_scope':list(INLINE_IDS),'inline_overlay':'omitted for1MB budget; full10 retains unchanged overlays','standalone_scope':'all10 existing queries',
            'limits':['library thumbnails retargeted to character; body/head details can differ from raw BVH geometry',
                      'elbow labels are original BVH, not image measurements','no new inference, scoring, pose editing or accuracy claim',
                      'matched camera still exists in old archived screen; not supported by these fixed-view JPEGs'],
            'inputs':hashes,'code':{n:sha256(ROOT/n) for n in code}}
    save_json(a.out/'protocol.json',policy)
    assets={};asset_hashes={};bvh_hashes={}
    for pid in ids:
        stored=Path(paths[pid]);choices=(stored,a.library/stored,a.library/'bvh'/stored.name)
        path=next(p for p in choices if p.is_file());bvh_hashes[str(path)]=sha256(path)
        kp,_=load_coco17(str(path));assert np.array_equal(kp,gallery[pid]),'gallery geometry changed: '+pid
        assets[pid]={}
        for view in VIEWS:
            record=entries[(pid,view)];p=a.thumbs/record['filename']
            assert record['status']=='ok' and p.name==pid+'__'+view+'.jpg'
            digest=sha256(p);assert digest==record['sha256'],'thumbnail checksum mismatch: '+str(p)
            with Image.open(p) as image:assert image.format=='JPEG' and image.size==(record['width'],record['height']);image.verify()
            asset_hashes[str(p)]=digest;assets[pid][view]='data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()
    full=assemble(source,assets);focus=assemble(source,assets,INLINE_IDS)
    make=lambda data:template.read_text().replace('__REVIEW_DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    fragment=make(focus);assert len(fragment.encode())<1_000_000,'focus inline exceeds budget'
    a.inline.parent.mkdir(parents=True,exist_ok=True);a.inline.write_text(fragment,encoding='utf-8')
    full_fragment=make(full)
    document='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>기본 인형 러프 검수</title><style>'+STANDALONE_STYLE+'</style></head><body>'+full_fragment+'</body></html>'
    (a.out/'review.html').write_text(document,encoding='utf-8')
    save_json(a.out/'review-data-private.json',full)
    save_json(a.out/'asset-provenance.json',{'thumbs':asset_hashes,'bvh':bvh_hashes})
    # Independent structural check ensures candidate payload unchanged except cached render bytes.
    before={q['id']:q for q in source['queries']}
    for q in full['queries']:
        original=before[q['id']]
        assert q['input']==original['input'] and q['overlay']==original['overlay']
        for arm in ('before','after'):
            assert q[arm]==[{k:v for k,v in c.items() if k not in ('matched','canonical','canonical_kind')} for c in original[arm]]
    result={'queries':len(full['queries']),'focus_queries':len(focus['queries']),'unique_candidates':len(ids),'verified_original_jpegs':len(asset_hashes),
            'candidate_occurrences':sum(len(q[arm]) for q in full['queries'] for arm in ('before','after')),
            'gallery_geometry_verified':len(ids),'source_rankings_sha256':hashes[str(inputs[1])],
            'inline_bytes':len(fragment.encode()),'inline_sha256':sha256(a.inline),'rankings_changed':False,'accuracy':None}
    save_json(a.out/'verification.json',result)
    for p,h in {**hashes,**asset_hashes,**bvh_hashes}.items():assert sha256(Path(p))==h
    for n,h in policy['code'].items():assert sha256(ROOT/n)==h
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('source','gallery','database','library','thumbs','out','inline'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args()
    with (ROOT/'out/.perspective-rough-review.lock').open('a+') as rough,(ROOT/'out/.joint-shape.lock').open('a+') as shape,(ROOT/'out/.first-rank-confirm.lock').open('a+') as fresh:
        for lock in (rough,shape,fresh):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)
