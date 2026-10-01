#!/usr/bin/env python3
"""Replay frozen real VLM responses through individual routing and actual E5 search."""
from pathlib import Path
import argparse,json,sys,time
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scripts.eval_rough_slots import replay
from src.experimental.rough_semantic import RoughSemanticSearch,execute_semantic_plan,sha
from src.experimental.rough_semantic_v2 import build_rough_semantic_runtime,execute_structured_plan
from src.semantic_search import parse_semantic_query
from src.semantic_index import validate_semantic_index


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--responses',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('artifacts/camera_real_rough_20260921_v2/protocol.json'))
    ap.add_argument('--facts',type=Path,default=Path('artifacts/rough_router_phase2_20260922/semantic-refresh/current-member-facts.jsonl'))
    ap.add_argument('--implementation',choices=['structured_v2','dense_v1','multiview_v3'],default='structured_v2')
    ap.add_argument('--captions',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text());by_cut={}
    for c in protocol['cases']:
        if sha(c['image'])!=c['image_sha256'] or sha(c['response'])!=c['response_sha256']:raise ValueError('protocol changed')
        by_cut.setdefault(c['cut_id'],[]).append(c)
    source_manifest=json.loads((args.responses/'manifest.json').read_text())
    if sha(args.protocol)!=source_manifest['protocol_sha256']:raise ValueError('source protocol mismatch')
    rows=[]
    for c in source_manifest['inputs']:
        f=sorted((args.responses/'responses').glob(c['cut_id']+'.attempt-*.json'))[-1]
        r=json.loads(f.read_text())
        if r['status']!='ok':raise ValueError('VLM response missing')
        if sha(c['image'])!=c['image_sha256'] or r['image_sha256']!=c['image_sha256']:raise ValueError('image changed')
        with Image.open(c['image']) as im:w,h=im.size
        fresh=replay(r['raw'],w,h,by_cut[c['cut_id']])
        rows.append(dict(cut_id=c['cut_id'],image=c['image'],source_response=str(f),source_sha256=sha(f),
                         original_routes=[p['plan']['route'] for p in r['people']],**fresh))
    (args.out/'plans.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    audits=[]
    for manifest in sorted(Path('data/semantic/builds').glob('*/semantic-build.json')):
        try:
            validate_semantic_index(manifest.parent/'pose_semantics.db',manifest,
                geometry_db_path=Path('data/poses.db'),profile_path=Path('config/semantic_embedding.e5-small.v1.json'))
            state='valid'
        except Exception as e:state=str(e)
        audits.append(dict(build=str(manifest.parent),status=state))
    queries=[q for r in rows for p in r['people'] for c in p['plan']['channels'] if c['kind']=='S' for q in c['queries']]
    audit=dict(existing_builds=audits,legacy_parser_total=len(queries),legacy_parser_clarify=sum(parse_semantic_query(q).intent=='clarify' for q in queries))
    (args.out/'existing-runtime-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print('plans',len(rows),'queries',len(queries),'building current E5 index',flush=True)
    started=time.perf_counter()
    if args.implementation=='multiview_v3':
        from src.experimental.rough_multiview import MultiviewRoughSemanticSearch
        runtime=MultiviewRoughSemanticSearch(captions_path=args.captions,facts_path=args.facts,geometry_db=Path('data/poses.db'),build_dir=args.build)
    else:
        runtime=build_rough_semantic_runtime(implementation=args.implementation,facts_path=args.facts,geometry_db=Path('data/poses.db'),build_dir=args.build)
    build_seconds=time.perf_counter()-started
    counts=Counter();latencies=[];candidate_ids=set();regions=Counter()
    with (args.out/'results.jsonl').open('x') as stream:
        for row in rows:
            for p in row['people']:
                started=time.perf_counter();execution=(execute_structured_plan(p['plan'],runtime,slots=p['slots']) if args.implementation!='dense_v1' else execute_semantic_plan(p['plan'],runtime))
                elapsed=time.perf_counter()-started
                result=dict(cut_id=row['cut_id'],person_index=p['person_index'],slots=p['slots'],
                    observation_mapping=p['observation_mapping'],plan=p['plan'],execution=execution,elapsed_seconds=elapsed)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                counts['people']+=1;counts[p['plan']['route']]+=1
                if execution['executed_query_count']:counts['semantic_people']+=1;latencies.append(elapsed)
                for request in execution['semantic_requests']:
                    counts['search_requests']+=1;counts['nonempty_requests']+=bool(request['results']);regions[request['region']]+=1
                    for c in request['results']:
                        assert c['refine_allowed'] is False and c['match_source']=='semantic_rough'
                        assert sha(c['bvh_path'])==c['bvh_sha256'].removeprefix('sha256:')
                        candidate_ids.add(c['pose_id'])
                print(row['cut_id'],p['person_index'],p['plan']['route'],execution['executed_query_count'],flush=True)
    summary=dict(counts=dict(counts),query_regions=dict(regions),unique_candidate_members=len(candidate_ids),
        build_seconds=round(build_seconds,3),search_seconds=round(sum(latencies),3),
        build_id=runtime.build_id,index_documents=len(runtime.documents),index_members=len(runtime.paths),
        source_model=source_manifest['requested_model'],vlm_calls=0,composition_executed=False,
        production_api_semantic_enabled=False,accuracy_evaluated=False,production_ready=False)
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False),flush=True)
    from scripts.build_rough_semantic_review import build_review
    review=build_review(args.out,args.out/'review',build_manifest=args.build/'manifest.json')
    print(json.dumps({'visual_review':review},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
