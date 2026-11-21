"""Frozen paired minimum-gate comparison; no network requests from this entrypoint."""
import argparse, concurrent.futures, copy, os, runpy, subprocess, time
from common import HERE, Path, read, sha, shutil, write
EXCLUSIONS = {}

run_case=runpy.run_path(str(HERE/'pipeline.py'))['run_case']

def main():
    p=argparse.ArgumentParser()
    for k in ['assets','prepared','out','extractor-checkout','model-manifest']:
        p.add_argument('--'+k,type=Path,required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prior-runs',type=Path)
    modes.add_argument('--fresh-prepared',type=Path)
    modes.add_argument('--replay-run',type=Path,help='Replay the original shared response and skeleton from the frozen baseline')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--blender',default=__import__('os').environ.get('REFINE_BLENDER','blender'))
    p.add_argument('--profile-directions',type=Path)
    p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args()
    selected=list(dict.fromkeys(a.cases))
    a.out.mkdir(parents=True,exist_ok=False)
    a.semantic_cache=a.out/'response-replay'
    a.semantic_cache.mkdir()
    paths={}
    from semantics import prompt
    from hashlib import sha256
    ph=sha256(prompt().encode()).hexdigest()
    rows={r['case_id']:r for r in read(a.assets/'input/manifest.json')['rows']}
    if not set(selected)<=set(rows):raise ValueError('Cut missing from bound manifest')
    fresh={}
    prior=None
    if a.replay_run:
        prior=read(a.replay_run/'config.json')
        if (prior['version']!='minimum-gate-fresh-v3' or prior['prompt_sha256']!=ph
                or prior['manifest_sha256']!=sha(a.assets/'input/manifest.json')
                or prior['model_manifest_sha256']!=sha(a.model_manifest)
                or not set(selected)<=set(prior['cases'])):
            raise ValueError('Original baseline does not match the replay inputs')
    if a.fresh_prepared:
        fresh={row['case']:row for row in read(a.fresh_prepared/'luna-summary.json')}
        assert set(selected)<=set(fresh)
    for c in selected:
        if prior:
            response=a.replay_run/'response-replay'/f'{c}.json'
            saved_observation=a.replay_run/'preserve'/c/'observations'/f'{c}.json'
            prepared_observation=a.prepared/c/'observations'/f'{c}.json'
            if (sha(response)!=prior['response_cache'][c]['sha256']
                    or sha(saved_observation)!=sha(prepared_observation)
                    or sha(a.assets/'input'/rows[c]['rough'])!=rows[c]['rough_sha256']):
                raise ValueError(f'{c}: baseline response, skeleton or original image hash mismatch')
            sources=[('minimum_gate_fresh_v3',response)]
        elif a.fresh_prepared:
            sources=[('fresh_luna_cohort',a.fresh_prepared/c/'semantics/response.json')] if fresh[c]['status']=='complete' else []
        else:
            # Frozen chronology priority, independent of outcomes.
            sources=[(cohort,a.prior_runs/cohort/c/'semantics/response.json') for cohort in ['preserve-five-original-v1','unified-004-fresh-v1','unified-pilot-v1']]
        for cohort,src in sources:
            if not src.exists():continue
            d=read(src)
            if d.get('model')=='gpt-6-luna' and d.get('reasoning_effort')=='low' and d.get('manual_conditions_used') is False and d.get('image_sha256')==rows[c]['rough_sha256'] and d.get('prompt_sha256')==ph:
                shutil.copy2(src,a.semantic_cache/f'{c}.json')
                paths[c]=dict(source_run=cohort,sha256=sha(src),usage=d.get('usage'),provider_seconds=d.get('seconds'))
                break
        if (prior or (a.fresh_prepared and fresh[c]['status']=='complete')) and c not in paths:
            raise ValueError(f'{c}: Luna output failed exact provenance verification')
    profile_directions={}
    if a.profile_directions:
        from profile_direction import select
        statuses={r['case']:r['status'] for r in read(a.profile_directions/'status.json')}
        for c in selected:
            response=read(a.semantic_cache/f'{c}.json')
            camera=response['raw']['camera']
            if camera.get('torso_view')!='profile':
                if c in statuses and statuses[c]!='not_profile_no_request':
                    raise ValueError(f'{c}: unexpected profile request')
                continue
            if c not in statuses:
                raise ValueError(f'{c}: missing profile request status')
            file=a.profile_directions/f'{c}.json'
            direction=read(file) if file.is_file() else None
            _,decision=select({'raw':camera,'image_sha256':response['image_sha256']},direction)
            profile_directions[c]={'status':statuses[c],'decision':decision}
            if direction:
                profile_directions[c].update(
                    sha256=sha(file),prompt_sha256=direction['prompt_sha256'],
                    usage=direction.get('usage'),
                    chest_facing=direction['raw'].get('chest_facing'),
                    confidence=direction['raw'].get('confidence'),
                )
    cfg=dict(version='New Refine Minute View v2',cases=selected,exclusions=EXCLUSIONS,
             source='manifest_bound_original_rig_and_actual_bvh',search_rerun=False,
             prepared_skeleton=str(a.prepared),model='gpt-6-luna',reasoning_effort='low',
             fresh_provider_calls=sum((a.fresh_prepared/c/'semantics/request-metadata.json').exists() for c in selected) if a.fresh_prepared else 0,
             fresh_provider_observations=fresh, provider_completed=len(paths) if a.fresh_prepared else 0,
             provider_mode='fresh_once_shared_by_both_branches' if a.fresh_prepared else 'historical_replay_or_geometry_only',
             replay_run=str(a.replay_run.resolve()) if prior else None,
             replay_source_config_sha256=sha(a.replay_run/'config.json') if prior else None,
             profile_direction_provider=profile_directions,
             response_cache=paths,prompt_sha256=ph,manifest_sha256=sha(a.assets/'input/manifest.json'),
             extractor_revision=subprocess.check_output(['git','-C',a.extractor_checkout,'rev-parse','HEAD'],text=True).strip(),
             model_manifest_sha256=sha(a.model_manifest),
             code_hashes={str(f.relative_to(HERE.parent)):sha(f) for f in sorted(HERE.parent.glob('refine*/*.py'))})
    write(a.out/'config.json',cfg)
    runs={}
    for mode in ('preserve','shadow'):
        b=copy.copy(a);b.out=a.out/mode;b.out.mkdir();b.shadow=mode=='shadow'
        if b.shadow:b.gated_root=a.out/'preserve'
        runs[mode]=b
    start=time.monotonic()
    def task(c):
        out={}
        for mode,b in runs.items():
            t=time.monotonic()
            out[mode]=run_case(c,b)
            out[mode]['case_wall_seconds']=time.monotonic()-t
            print(c,mode,out[mode]['status'],out[mode].get('error',''),flush=True)
        write(a.out/'pairs'/f'{c}.json',dict(case=c,**out))
        return dict(case=c,**out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(task,c) for c in selected];results={}
        for f in concurrent.futures.as_completed(futures):
            row=f.result()
            results[row['case']]=row
            write(a.out/'results.json',[results[c] for c in selected if c in results])
    write(a.out/'execution.json',dict(seconds=time.monotonic()-start,workers=a.workers,count=len(results)))

if __name__=='__main__':
    main()
