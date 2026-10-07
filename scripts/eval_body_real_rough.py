#!/usr/bin/env python3
"""Fresh Flash-Lite body observations on hash-verified real roughs and nine real FBX renders.
Reuses audited cached person/pose extraction; never promotes assets to production QA.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import CFG
from src.schema import BBox,CutResult,PersonDescriptor,PoseCandidate,Skeleton,Shot,Action,View,Relationship
from src.experimental.body_matching.observation import GeminiBodyAttributeClient,PROMPT_VERSION,parse_response
from src.experimental.body_matching.schema import OBSERVATION_VERSION
from src.experimental.body_matching.catalog import file_sha256
from src.experimental.body_matching.service import BodyMatchingService
from src.experimental.body_matching.selection import compare_body_shapes,SCORER_VERSION


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class RecordingClient(GeminiBodyAttributeClient):
    def __init__(self,model,path,timeout=30):
        super().__init__(model,timeout)
        self.path=path
    def analyze(self,crops):
        if self.path.exists():
            saved=json.loads(self.path.read_text())
            return parse_response(json.loads(saved['text']),[k for k,_ in crops])
        try:
            return super().analyze(crops)
        except Exception as exc:
            write(self.path.with_suffix('.error.json'),dict(error_type=type(exc).__name__,
                http_status=getattr(exc,'code',None),provider_status=getattr(exc,'status',None)))
            raise
    def record_response(self,response):
        write(self.path,dict(model_version=getattr(response,'model_version',None),
            response_id=getattr(response,'response_id',None),text=response.text,
            usage=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/body-matching/2026-10-06-flash-lite-9')
    parser.add_argument('--model',default='gemini-flash-lite-latest')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args();out=args.out.resolve();source=args.source_root.resolve()
    from dotenv import load_dotenv
    load_dotenv(source/'.env',override=False)
    model_manifest=json.loads((out/'models.json').read_text())
    protocol_path=source/'artifacts/camera_real_rough_20260921_v2/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert len(model_manifest['models'])==9
    if (out/'shape-catalog.json').exists():
        catalog=json.loads((out/'shape-catalog.json').read_text())
        assert catalog['requested_model']==args.model
        for a in catalog['assets']:assert file_sha256(source/a['fbx_path'])==a['asset_sha256']
    else:
        assets=[]
        for offset in range(0,9,3):
            group=model_manifest['models'][offset:offset+3]
            crops=[]
            for a in group:
                assert file_sha256(source/a['fbx_path'])==a['asset_sha256']
                crops.append((a['body_id'],Image.open(out/a['preview']).convert('RGB')))
            response_path=out/f'provider/model-{offset//3}.json'
            started=perf_counter()
            if response_path.exists():
                observed=parse_response(json.loads(json.loads(response_path.read_text())['text']),[k for k,_ in crops])
            else:
                client=RecordingClient(args.model,response_path,timeout=60)
                try:observed=client.analyze(crops)
                finally:client.client.close()
            for a in group:
                attrs={key:item['value'] for key,item in observed[a['body_id']]['attributes'].items()}
                assets.append(dict(**a,body_version=a['asset_sha256'][:12],rig_version='existing-fbx',
                    measurement_version='vlm-semantic-render.v1',attributes=attrs,observation=observed[a['body_id']],
                    projections=[],quality_priority=0,preview_sha256=file_sha256(out/a['preview'])))
            print('MODELS',offset+len(group),'/9',round(perf_counter()-started,2),flush=True)
        catalog=dict(scope='offline_shape_only_not_production_eligible',requested_model=args.model,
                     default_body_id='male-base',assets=assets)
        write(out/'shape-catalog.json',catalog)
    by_cut=defaultdict(list)
    for case in protocol['cases']:by_cut[case['cut_id']].append(case)
    frozen=dict(requested_model=args.model,prompt_version=PROMPT_VERSION,scorer_version=SCORER_VERSION,
                input_protocol_sha256=file_sha256(protocol_path),shape_catalog_sha256=file_sha256(out/'shape-catalog.json'),
                people=len(protocol['cases']),cuts=len(by_cut),excluded_models=model_manifest['excluded'],
                person_pose_source='hash-verified real cached RTMPose/cascade + existing Top-5',
                body_source='fresh real Gemini requests; resumable saved results',timeout_seconds=30,
                code_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in (ROOT/'src/experimental/body_matching').glob('*.py')})
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==frozen,'protocol changed; use new output folder'
    else:write(out/'protocol.json',frozen)
    def execute(cut_id,cases):
        dest=out/'cuts'/(cut_id+'.json')
        if dest.exists():return json.loads(dest.read_text())
        cases=sorted(cases,key=lambda c:int(c['id'].split(':p')[1]))
        first=cases[0];image_path=Path(first['image']);assert file_sha256(image_path)==first['image_sha256']
        assert file_sha256(first['response'])==first['response_sha256']
        cached=json.loads(Path(first['response']).read_text())
        result=CutResult(cached['route'],cached['count_confidence'],cached['detector_count'],cached['vlm_count'])
        for c in cases:
            index=int(c['id'].split(':p')[1]);p=cached['people'][index]
            result.descriptors.append(PersonDescriptor(Shot.FULL_HALF,Action.OTHER,View.FRONT,Relationship.SOLO,
                Skeleton(np.array(c['keypoints']),np.array(c['scores'])),None,box=BBox(*c['box']),
                valid_joint_mask=np.array(c['mask']),skeleton_state=p.get('skeleton_state','valid')))
            result.person_candidates.append([PoseCandidate(x['pose_id'],x['view'],x['distance'],x['tags']) for x in p['candidates']])
        image=Image.open(image_path).convert('RGB')
        client=RecordingClient(args.model,out/'provider'/('rough-'+cut_id+'.json'))
        service=BodyMatchingService(ROOT/'config/body_catalog.v1.json',client=client,provider='gemini')
        try:production=service.analyze(image,result,mode='shadow')
        finally:client.client.close()
        evaluated=[]
        for c,p in zip(cases,production['people']):
            obs=p['observations'];decision=compare_body_shapes(catalog['assets'],obs,default_body_id=catalog.get('default_body_id'), presentation_defaults=catalog.get('presentation_defaults',{}))
            cropfile='roughs/'+c['id'].replace(':','-')+'.png';(out/'roughs').mkdir(exist_ok=True)
            if obs['crop_box']:image.crop(obs['crop_box']).save(out/cropfile)
            evaluated.append(dict(case_id=c['id'],crop=cropfile,observation=obs,decision=decision,
                                  production_diagnostic=p['diagnostic'],pose_bindings=p['pose_bindings']))
        payload=dict(cut_id=cut_id,body_elapsed_ms=production.get('elapsed_ms'),is_mock=production['is_mock'],
                     provider_actual=production['provider_actual'],people=evaluated)
        write(dest,payload);return payload
    cuts=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(execute,k,v):k for k,v in by_cut.items()}
        for future in as_completed(tasks):
            key=tasks[future]
            try:
                item=future.result();cuts.append(item)
                print('CUT',key,[(p['case_id'],p['decision']['auto_body_id'],p['decision']['selection_source']) for p in item['people']],flush=True)
            except Exception as exc:
                errors.append(dict(cut_id=key,error_type=type(exc).__name__));print('ERROR',key,type(exc).__name__,flush=True)
    people=[p for c in cuts for p in c['people']]
    summary=dict(cuts=len(cuts),people=len(people),expected_people=len(protocol['cases']),errors=errors,
                 selection_counts=dict(Counter(p['decision']['auto_body_id'] for p in people)),
                 source_counts=dict(Counter(p['decision']['selection_source'] for p in people)),
                 provider_error_people=sum(bool(p['observation'].get('provider_error')) for p in people),
                 ownership_blocked_people=sum(p['observation']['ownership_ambiguous'] for p in people),
                 tied_top1_people=sum(len(p['decision'].get('tied_body_ids',[]))>1 for p in people),
                 actual_model_versions=sorted({json.loads(p.read_text())['model_version'] for p in (out/'provider').glob('*.json') if json.loads(p.read_text()).get('model_version')}),
                 accuracy=None,accuracy_reason='no artist-labeled acceptable-body ground truth',
                 rendering_executed=False,geometry_scored=False,pose_extraction_reused=True)
    write(out/'results.json',dict(protocol=frozen,summary=summary,cuts=sorted(cuts,key=lambda x:x['cut_id'])))
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
