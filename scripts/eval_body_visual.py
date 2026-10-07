#!/usr/bin/env python3
"""Compare the same real rough people with nine actual rest-view candidate cards."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experimental.body_matching.visual_selection import GeminiVisualBodySelector,VISUAL_VERSION,VISUAL_PROMPT,parse_comparison
from src.experimental.body_matching.catalog import file_sha256
from scripts.eval_body_real_rough import write
from PIL import Image


class RecordingSelector(GeminiVisualBodySelector):
    def __init__(self,model,cards,directory):
        self.directory=directory;self.call_index=0;self.current=None
        super().__init__(model,cards,record=self.save)
    def save(self,response,order):
        write(self.current,dict(text=response.text,model_version=getattr(response,'model_version',None),
            response_id=getattr(response,'response_id',None),candidate_order=order,
            usage=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None))
    def _compare(self,crop,observation,ordered,aliases):
        self.current=self.directory/(str(self.call_index)+'.json');self.call_index+=1
        if self.current.exists():
            try:return parse_comparison(json.loads(json.loads(self.current.read_text())['text']),list(aliases),observation)
            except ValueError as exc:
                write(self.current.with_suffix('.error.json'),dict(error_type='ValueError',validation_error=str(exc),replayed=True))
                raise
        try:return super()._compare(crop,observation,ordered,aliases)
        except Exception as exc:
            write(self.current.with_suffix('.error.json'),dict(error_type=type(exc).__name__,
                http_status=getattr(exc,'code',None),provider_status=getattr(exc,'status',None),
                validation_error=str(exc) if isinstance(exc,ValueError) else None,
                quota_details=[d for d in getattr(exc,'details',[]) if isinstance(d,dict) and str(d.get('@type','')).endswith(('QuotaFailure','RetryInfo'))] if isinstance(getattr(exc,'details',None),list) else []))
            raise
        finally:
            time.sleep(5)  # pacing for the shared provider RPM/TPM budget


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--stage',default='visual-v2')
    parser.add_argument('--model',default='gemini-flash-lite-latest')
    args=parser.parse_args();out=args.out.resolve();source=args.source_root.resolve()
    from dotenv import load_dotenv
    load_dotenv(source/'.env',override=False)
    catalog=json.loads((out/'shape-catalog.json').read_text());assets=catalog['assets']
    measurements={p['body_id']:p for p in json.loads((out/'measurements.json').read_text())['profiles']}
    cards={}
    for a in assets:
        assert file_sha256(source/a['fbx_path'])==a['asset_sha256']
        m=measurements[a['body_id']];assert m['asset_sha256']==a['asset_sha256']
        front=out/a['preview'];side=front.with_name(front.stem+'-side.png')
        cards[a['body_id']]=dict(asset_sha256=a['asset_sha256'],metrics=m['metrics'],
            front=dict(path=str(front),sha256=file_sha256(front)),side=dict(path=str(side),sha256=file_sha256(side)))
    data=json.loads((out/'results.json').read_text());people=[p for c in data['cuts'] for p in c['people']]
    import hashlib
    frozen=dict(version=VISUAL_VERSION,requested_model=args.model,source_results_sha256=file_sha256(out/'results.json'),
        measurement_sha256=file_sha256(out/'measurements.json'),catalog_sha256=file_sha256(out/'shape-catalog.json'),
        prompt_sha256=hashlib.sha256(VISUAL_PROMPT.encode()).hexdigest(),cards=cards,
        code_sha256=file_sha256(ROOT/'src/experimental/body_matching/visual_selection.py'),
        crop_sha256={p['case_id']:file_sha256(out/p['crop']) for p in people},
        mode='offline shape only; same real people and cached pose IDs; two orders + alias rotation')
    dest=out/args.stage;dest.mkdir(exist_ok=True)
    if (dest/'protocol.json').exists():assert json.loads((dest/'protocol.json').read_text())==frozen,'Use new output folder for changed inputs/code'
    else:write(dest/'protocol.json',frozen)
    def run(person):
        key=person['case_id'].replace(':','-');result_file=dest/'people'/(key+'.json')
        if result_file.exists():return json.loads(result_file.read_text())
        client=RecordingSelector(args.model,cards,dest/'provider'/key);started=time.perf_counter()
        try:
            with Image.open(out/person['crop']) as crop:
                decision=client.select(crop,person['observation'],assets,person['decision'])
        except Exception as exc:
            decision=dict(person['decision'],visual_comparison=dict(accepted=False,
                reason='visual_validation_rejected' if isinstance(exc,ValueError) else 'visual_provider_'+type(exc).__name__,
                validation_error=str(exc) if isinstance(exc,ValueError) else None))
        finally:client.close()
        result=dict(person,decision=decision,attribute_decision=person['decision'],elapsed_ms=round((time.perf_counter()-started)*1000,2))
        write(result_file,result);print(key,decision['auto_body_id'],decision['visual_comparison']['reason'],flush=True);return result
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run,people):results.append(result)
    summary=dict(people=len(results),accepted_visual=sum(p['decision']['visual_comparison']['accepted'] for p in results),
        selection_counts=dict(Counter(p['decision']['auto_body_id'] for p in results)),
        source_counts=dict(Counter(p['decision']['selection_source'] for p in results)),
        comparison_reasons=dict(Counter(p['decision']['visual_comparison']['reason'] for p in results)),
        changed_from_attributes=sum(p['decision']['auto_body_id']!=p['attribute_decision']['auto_body_id'] for p in results),
        actual_model_versions=sorted({json.loads(p.read_text())['model_version'] for p in (dest/'provider').rglob('*.json') if json.loads(p.read_text()).get('model_version')}),
        accuracy=None,rendering_executed=False,pose_search_reused=True)
    write(dest/'results.json',dict(summary=summary,people=results));print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
