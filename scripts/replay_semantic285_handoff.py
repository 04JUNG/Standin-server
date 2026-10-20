"""Replay imported 30-case inputs and saved real VLM descriptions, no new calls.

Human corrections/preferences are kept as evidence, NOT used as VLM output or
training labels. This is an integration regression, not a new accuracy benchmark.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.semantic_adapter_285 import build_semantic285_searcher, Semantic285Adapter
from src.semantic_search import SemanticQuery
from src.semantic_search_285 import SemanticSettings


def query_for(person):
    vlm=person['vlm'];mapping={'posture':'postures','action':'actions','domain':'domains','activity':'activities',
                            'object':'objects','intent':'intents','emotion':'emotions'}
    fields={name:[] for name in mapping.values()}
    for tag in vlm.get('tags',[]):
        key,sep,value=tag.partition(':')
        if sep and key in mapping:fields[mapping[key]].append(value)
    return SemanticQuery(text=vlm['sentence'],body_details=vlm.get('observation',''),**fields)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('library','inputs','baseline','shape-index','body-index','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dense-model-dir',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('choose new output; do not overwrite evaluation')
    args.out.mkdir(parents=True)
    settings=SemanticSettings()
    if args.dense_model_dir:settings=replace(settings,weight_dense=.5,dense_lo=.72,dense_hi=.88)
    lib=args.library
    adapter=build_semantic285_searcher(tags_path=lib/'dist/data/search-v2.json',taxonomy_path=lib/'dist/data/search-taxonomy-v2.json',
        db_path=lib/'source_bundle/poses.db',shape_index_path=args.shape_index,body_index_path=args.body_index,
        dense_model_dir=args.dense_model_dir,dense_cache_dir=args.out/'dense-cache',settings=settings)
    if not isinstance(adapter,Semantic285Adapter):raise RuntimeError(adapter.reason)
    assert adapter.service.index.review_preferences=={}
    inputs=json.loads(args.inputs.read_text());saved=json.loads(args.baseline.read_text());by_id={r['id']:r for r in saved['rows']}
    output={'evaluation_kind':'imported_30_saved_descriptions_integration_no_preferences',
            'inputs_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),'description_source':str(args.baseline),
            'index':adapter.service.index.identity,'settings':vars(settings),'extra_vlm_calls':0,
            'exact_query_preferences_enabled':False,'per_query_rejections_enabled':False,'rows':[]}
    statuses=Counter();id_ok=True
    def search(query,source):
        nonlocal id_ok
        r=adapter.search(query,5,source=source)
        id_ok &= all(h.pose_id==h.pose['pose_id'] and not h.pose['refine_allowed'] for h in r.hits)
        statuses[r.status]+=1
        return {'status':r.status,'reason':r.reason,'backend':r.backend,'elapsed_ms':r.elapsed_ms,
                'candidates':[h.to_dict() for h in r.hits]}
    for c in inputs['cases']:
        row={'id':c['id'],'input_kind':c['input_kind'],'people':[]}
        if c['input_kind']=='text':row['search']=search(SemanticQuery(text=c['text']),'user')
        else:
            original=by_id[c['id']]
            for p in original.get('people',[]):
                v=p.get('vlm')
                person={'person_index':p.get('person_index'),'vlm':v,'original_path':p.get('path')}
                if not v or not v.get('readable') or not v.get('sentence'):
                    person['search']={'status':'not_requested','reason':'saved_vlm_unreadable'}
                else:person['search']=search(query_for(p),'vlm')
                row['people'].append(person)
            row['selected_person_index']=original.get('selected_person_index')
        output['rows'].append(row)
    output['summary']={'inputs':len(output['rows']),'search_requests':sum(statuses.values()),'statuses':dict(statuses),
                       'candidate_id_json_consistent_and_refine_blocked':id_ok,'not_independent_accuracy':True}
    (args.out/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output['summary'],ensure_ascii=False,indent=2))
    if not id_ok:raise RuntimeError('candidate contract failed')


if __name__=='__main__':main()
