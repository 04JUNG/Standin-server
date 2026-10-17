"""Static input, image and blinding checks, not browser rendering QA."""
import argparse,hashlib,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    public=json.loads((a.out/'review-data.json').read_text())
    key=json.loads((a.out/'blinding-key-private.json').read_text())
    private={r['image']:r for r in json.loads((a.out/'rankings-private.json').read_text())}
    policy=json.loads((a.out/'protocol.json').read_text())
    assert len(public['queries'])==10 and len(public['candidate_ids'])==len(set(public['candidate_ids']))==120
    assert sum(q['failed'] for q in public['queries'])==2
    copy=dict(public);digest=copy.pop('public_data_sha256')
    assert hashlib.sha256(json.dumps(copy,sort_keys=True).encode()).hexdigest()==digest
    assert key['public_data_sha256']==digest
    assert sha256(ROOT/'out/humanart-plus-z-20261012/results.json')==policy['cache_sha256']
    for f,h in policy['frozen_sources'].items():assert sha256(ROOT/f)==h
    count=0
    for q in public['queries']:
        private_row=private[key['queries'][q['id']]['image']]
        assert sha256(a.out/q['input'])==private_row['input_sha256']
        assert q['failed']==bool(private_row.get('error'))
        assert set(key['queries'][q['id']]['slots'].values())=={'anchor_4','anchor_117','pooled_117'}
        assert len(q['groups'])==(0 if q['failed'] else 3)
        for group in q['groups']:
            assert len(group['candidates'])==5
            for c in group['candidates']:
                assert c['id'] in public['candidate_ids']
                with Image.open(a.out/c['image']) as im:assert im.size==(448,448);im.verify()
                count+=1
    html=(a.out/'review.html').read_text()
    assert '__STUDY_DATA__' not in html
    assert all(x not in html for x in ('anchor_4','anchor_117','pooled_117','pose_id','rankings-private','blinding-key-private'))
    assert not list(a.out.glob('*ratings.json'))
    audit={'queries':10,'input_bytes_verified':10,'candidate_pngs_verified':count,
           'failed_queries_retained':2,'source_cache_and_code_hashes_verified':True,
           'blinded_public_payload_verified':True,'human_ratings_received':0,
           'browser_visual_qa':'not performed: local-file browser access blocked by policy',
           'review_html_sha256':sha256(a.out/'review.html')}
    save_json(a.out/'audit.json',audit);print(json.dumps(audit,indent=2))
