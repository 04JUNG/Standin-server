"""Prepare a strictly offline replay of existing VLM records and frozen FBX."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from src.arm_refine.cached_targets import target_from_cached

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();root=args.workspace.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frozen=root/'outputs/standin-rescore-v3.2';manifest=json.loads((frozen/'dataset-manifest.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    snapshot=out/'source-snapshot';snapshot.mkdir();hashes={}
    for name in ('src/arm_refine/diagnosis.py','src/arm_refine/cached_targets.py','src/arm_refine/native_state.py',
                 'scripts/prepare_cached_limb_pilot.py','scripts/run_cached_limb_pilot_native.py',
                 'src/arm_refine/canonical_surface.py','src/arm_refine/explicit_limb_text.py'):
        source=REPO/name;shutil.copyfile(source,snapshot/source.name);hashes[name]=sha(source)
    (out/'implementation-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cases=[]
    for tid in ('T001','T002','T018','T024','T026','T027'):
        case=next(c for c in manifest['cases'] if c['test_id']==tid)
        cache=root/'outputs/closed-beta-develop-20261019/full50'/(tid+'.json')
        raw=json.loads(cache.read_text());assert raw['input_sha256']==case['input']['sha256']
        person=raw['response']['people'][0]
        description=person['dynamic_search']['description_evidence']['description']
        hands=person.get('hands') or {}
        target=target_from_cached(description,hands)
        folder=out/tid;folder.mkdir()
        (folder/'cached-vlm-evidence.json').write_text(json.dumps({'description':description,'hands':hands,
            'source_path':str(cache),'source_sha256':sha(cache),'input_sha256':raw['input_sha256']},ensure_ascii=False,indent=2)+'\n')
        (folder/'target.json').write_text(json.dumps(target,ensure_ascii=False,indent=2)+'\n')
        image=frozen/case['input']['path'];assert sha(image)==case['input']['sha256']
        shutil.copyfile(image,folder/'input.png')
        fbx=frozen/'outputs'/case['case_id']/'P1/final.fbx'
        cases.append({'id':tid,'case_id':case['case_id'],'source_fbx':str(fbx.relative_to(root)),
            'source_sha256':sha(fbx),'input_sha256':sha(image),'reference_image':str(image.relative_to(root)),
            'scope':case['evaluation_scope'],'target_source':'replayed_existing_service_vlm',
            'cached_response_sha256':sha(cache),'new_api_calls':0,'new_api_tokens':0})
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps({'mode':'strictly_offline_cached_vlm_vs_native_selected_fbx',
        'cases':len(cases),'new_api_calls':0,'new_api_tokens':0,'network_required':False,
        'manual_corrections_used_as_inputs':False,'frozen_reviews_used_as_inputs':False,
        'fresh_image_recognition_measured':False,'automatic_limb_selection':True,
        'production_or_staging_modified':False,'scope':'development pilot, not 50-cut accuracy or 90-percent claim'},indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'new_api_calls':0,'output':str(out)}))
if __name__=='__main__':main()
