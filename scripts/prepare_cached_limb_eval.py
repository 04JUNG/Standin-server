"""Offline full frozen-set inventory/targets. Native execution is separate opt-in."""
import argparse
import hashlib
import json
import shutil
import sys
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from src.arm_refine.cached_targets import target_from_cached
from src.experimental.cached_leg_diagnosis import targets_from_cached
from src.experimental.cached_limb_inventory import match_cached_people


def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();root=args.workspace.resolve();out=args.output.resolve()
    frozen=root/'outputs/standin-rescore-v3.2';manifest=json.loads((frozen/'dataset-manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    snapshot=out/'source-snapshot';snapshot.mkdir();hashes={}
    sources=list((REPO/'src/arm_refine').glob('*.py'))+list((REPO/'src/experimental').glob('*diagnosis.py'))+[REPO/'src/experimental/cached_limb_inventory.py']
    sources += [Path(__file__),REPO/'scripts/run_cached_limb_pilot_native.py',REPO/'scripts/verify_arm_language_batch.py']
    for source in sources:
        relative=source.relative_to(REPO);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);hashes[str(relative)]=sha(source)
    save(out/'implementation-hashes.json',hashes)
    inventory=[];cases=[]
    for case in manifest['cases']:
        cache=root/'outputs/closed-beta-develop-20261019/full50'/(case['test_id']+'.json')
        raw=json.loads(cache.read_text()) if cache.is_file() else None
        image=frozen/case['input']['path']
        image_valid=image.is_file() and sha(image)==case['input']['sha256']
        cache_valid=raw is not None and raw.get('input_sha256')==case['input']['sha256']
        people=(raw.get('response') or {}).get('people',[]) if cache_valid else []
        mapping=match_cached_people(case,people) if image_valid and cache_valid else {}
        selections={s['reference_person_id']:s for s in case['final_selections']}
        for pid in case['expected_person_ids']:
            ident=case['test_id']+'-'+pid.rsplit('-',1)[-1];folder=out/ident;folder.mkdir()
            row={'id':ident,'test_id':case['test_id'],'reference_person_id':pid,'case_id':case['case_id'],
                 'status':'held','reasons':[],'new_api_calls':0,'person_mapping':mapping.get(pid)}
            if not image_valid:row['reasons'].append('frozen_image_hash_missing_or_changed')
            if not cache_valid:row['reasons'].append('original_vlm_cache_missing_or_image_hash_changed')
            match=mapping.get(pid) or {}
            if match.get('status')!='matched':row['reasons'].append(match.get('reason','cached_person_mapping_unavailable'))
            else:
                person=people[match['cache_person_index']]
                description=((person.get('dynamic_search') or {}).get('description_evidence') or {}).get('description')
                if not isinstance(description,dict):row['reasons'].append('cached_description_unavailable')
                else:
                    evidence={'description':description,'hands':person.get('hands') or {},'source_sha256':sha(cache),
                        'input_sha256':raw['input_sha256'],'person_mapping':match}
                    save(folder/'cached-vlm-evidence.json',evidence)
                    save(folder/'target.json',target_from_cached(description,evidence['hands']))
                    save(folder/'leg-target.json',targets_from_cached(description))
                    row['target_extracted']=True
            selection=selections.get(pid);asset=(selection or {}).get('fbx')
            source=frozen/asset['path'] if asset else None
            if not asset:row['reasons'].append('no_frozen_selected_fbx_record')
            elif not source.is_file():row['reasons'].append('frozen_selected_fbx_file_missing')
            elif sha(source)!=asset['sha256']:row['reasons'].append('frozen_selected_fbx_hash_changed')
            if not row['reasons']:
                row['status']='native_ready';shutil.copyfile(image,folder/'input.png')
                cases.append({'id':ident,'test_id':case['test_id'],'reference_person_id':pid,'case_id':case['case_id'],
                    'source_fbx':str(source.relative_to(root)),'source_sha256':asset['sha256'],
                    'input_sha256':case['input']['sha256'],'reference_image':str(image.relative_to(root)),
                    'scope':case['evaluation_scope'],'person_mapping':match,'new_api_calls':0})
            inventory.append(row);save(folder/'inventory.json',row)
    save(out/'cases.json',cases);save(out/'inventory.json',inventory)
    protocol={'mode':'strictly_offline_frozen_full_set_cached_diagnosis','dataset_id':manifest['dataset_id'],
        'baseline_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'cuts':len(manifest['cases']),'expected_people':len(inventory),'native_ready_people':len(cases),
        'targets_extracted_people':sum(r.get('target_extracted',False) for r in inventory),
        'new_api_calls':0,'new_api_tokens':0,'network_required':False,'fresh_recognition_measured':False,
        'grades_used_as_inputs':False,'correction_answers_used_as_inputs':False,
        'native_execution_enabled_by_default':False,'production_or_staging_modified':False,
        'multi_person_policy':'unique box IoU>=0.8 and margin>=0.15; reject collisions; never index fallback',
        'accuracy_estimated':False,'release_allowed':False}
    save(out/'protocol.json',protocol);print(json.dumps(protocol))


if __name__=='__main__':main()
