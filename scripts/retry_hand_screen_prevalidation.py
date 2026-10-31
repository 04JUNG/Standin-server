"""Reinspect fixed offline checkpoint failures, never live/user requests.

The isolated retry worker uses one Blender process and a longer offline budget.
Only completed, lineage-validated clear/intersection receipts become certificates.
"""
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.hand_screen_catalog import digest, make_catalog
from scripts.prevalidate_hand_screen_batch import fixtures, inspect_one, put_checkpoint


def prepare(plan, s3):
    from converter_api.app import app
    profile = next(r.endpoint for r in app.routes if r.path=='/hand-screen-profile')
    endpoint = next(r.endpoint for r in app.routes if r.path=='/screen-hand-contact')
    before = {c['id']:profile(character_id=c['id']) for c in plan['characters']}
    fingerprint = plan['profile_sha256']
    if any(p['profile_sha256']!=fingerprint or p['character_sha256']!=c['sha256']
           or p.get('shape_sha256','missing') is not None
           for c in plan['characters'] for p in [before[c['id']]]):
        raise ValueError('retry target distribution mismatch')
    started = time.monotonic()
    pending, parts, seen = [], [], set()
    counts = dict(attempted=0,clear_screen=0,intersection=0,unavailable=0)
    cursors = {j['shard']:0 for j in plan['source_jobs']}
    closed = set()
    print(json.dumps(dict(event='prevalidation_retry_started',profile_sha256=fingerprint)),flush=True)
    def read_optional(key):
        try:
            response=s3.get_object(Bucket=plan['output_bucket'],Key=key)
            with response['Body'] as body:return json.loads(body.read(4*1024*1024))
        except s3.exceptions.NoSuchKey:
            return None
    def checkpoint():
        nonlocal pending
        if not pending:return
        if {c['id']:profile(character_id=c['id']) for c in plan['characters']}!=before:
            raise ValueError('retry profile changed')
        pairs=[(r['binding'],r['receipt']) for r in pending if 'receipt' in r]
        value=dict(schema='hand-screen-preparation-part.v1',shard=0,part=len(parts),
            profile_sha256=fingerprint,library_sha256=plan['library']['sha256'],
            catalog=make_catalog(pairs) if pairs else None,
            unavailable=[r['unavailable'] for r in pending if 'unavailable' in r])
        parts.append(put_checkpoint(s3,plan['output_bucket'],plan['parts'][len(parts)],value))
        pending=[]
        print(json.dumps(dict(event='prevalidation_retry_progress',counts=counts,
            elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
    with tempfile.TemporaryDirectory(prefix='hand-retry-') as temp:
        root=Path(temp);archive=root/'library.tar.gz';lib=plan['library']
        s3.download_file(lib['bucket'],lib['key'],str(archive))
        rows=fixtures(archive,lib['sha256'],lib['pose_count'],0,1,root)
        by_id={r['id']:r for r in rows}
        while len(closed)<len(plan['source_jobs']):
            if time.monotonic()-started>plan.get('max_seconds',10800):
                raise ValueError('retry preparation deadline')
            found=False
            for job in plan['source_jobs']:
                if job['shard'] in closed:continue
                prefix=job['summary_key'].rsplit('/',1)[0]
                while True:
                    value=read_optional(prefix+'/part-'+str(cursors[job['shard']]).zfill(3)+'.json')
                    if value is None:break
                    if (value.get('schema')!='hand-screen-preparation-part.v1'
                            or value['shard']!=job['shard']
                            or value['part']!=cursors[job['shard']]
                            or value['profile_sha256']!=fingerprint
                            or value['library_sha256']!=lib['sha256']):
                        raise ValueError('unexpected retry source checkpoint')
                    for failure in value['unavailable']:
                        identity=failure['binding'];key=digest(identity)
                        if key in seen:raise ValueError('duplicate retry source')
                        seen.add(key)
                        row=by_id[identity['pose_id']]
                        if identity['source_bvh_sha256']!=row['sha256']:
                            raise ValueError('retry source hash mismatch')
                        result=inspect_one(endpoint,identity,row['path'],attempts=3,budget_seconds=60.)
                        counts['attempted']+=1
                        counts[result.get('receipt',{}).get('status','unavailable')]+=1
                        pending.append(result)
                        if len(pending)>=32:checkpoint()
                    cursors[job['shard']]+=1
                    found=True
                summary=read_optional(job['summary_key'])
                if summary is not None:
                    if cursors[job['shard']]!=len(summary['parts']):
                        raise ValueError('missing completed source checkpoint')
                    closed.add(job['shard'])
            # Save completed reinspections promptly, including sparse tails.
            checkpoint()
            if not found and len(closed)<len(plan['source_jobs']):time.sleep(10)
    summary=dict(schema='hand-screen-preparation-retry.v1',profile_sha256=fingerprint,
        library_sha256=plan['library']['sha256'],counts=counts,parts=parts,
        target_profiles=before,hand_settings=plan['hand_settings'],
        source_summary_keys=[j['summary_key'] for j in plan['source_jobs']],
        elapsed_seconds=round(time.monotonic()-started,2))
    put_checkpoint(s3,plan['output_bucket'],plan['summary'],summary)
    print(json.dumps(dict(event='prevalidation_retry_finished',counts=counts,
                         elapsed_seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    import argparse
    import boto3
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    prepare(json.loads(args.plan.read_text()),boto3.client('s3'))
