"""Preserve every quarter-turn prediction, then diagnose agreement against GT."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import save_json,sha256
from scripts.experiment_humanart_z import read_rgb
from scripts.diagnose_dual_camera_coverage import valid_query
from scripts.diagnose_body_observations import diagnose
from standin_eval.z_diagnostics import infer_pose
from standin_eval.rotation_consistency import agreement

CODE = ('scripts/diagnose_rotation_consistency.py','standin_eval/rotation_consistency.py',
        'standin_eval/z_diagnostics.py','scripts/experiment_humanart_z.py',
        'scripts/diagnose_body_observations.py','standin_eval/dual_label_search.py')


def read(path):
    return json.loads(path.read_text())


def verify(args):
    p=read(args.out/'protocol.json')
    assert all(sha256(ROOT/n)==h for n,h in p['code_hashes'].items())
    assert all(sha256(Path(n))==h for n,h in p['input_hashes'].items())


def infer(args):
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'hypotheses').mkdir()
    paths=[args.source/n for n in ('answers.json','manifest.json','audit.json')]
    paths+=sorted((args.source/'predictions').glob('*.json'))+[args.model]
    assert sha256(args.model)==read(args.source/'protocol.json')['hashes']['model']
    save_json(args.out/'protocol.json',{
        'stage':'diagnosis on 120 previously examined families / 360 images; no retrieval changes',
        'inference':'same local HumanArt M RGB192x256, pixel crop, turns0/1/2/3 independently through existing infer_pose; preserve all outputs',
        'reference':'original max median body score hypothesis; first tie; require exact XY/scores/turn reproduction, no replacement',
        'alignment':'each non-reference turn: direct or global LR permutation, mean pixel distance / observed reference torso on common score>=.3 body joints, >=6 required; minimum mean, direct tie; no GT',
        'agreement':'median per-joint distance to reference over available aligned alternatives; >=2 other score>=.3 predictions required',
        'bands':'[0,.1),[.1,.25),[.25,infinity) observed torso units; insufficient support separate',
        'diagnostic_error':'original selected XY vs GT pixels / GT torso with one global oracle mapping from prior body diagnosis; error>.25; GT only after all hypotheses saved',
        'stratify':'group, original score [.3,.5),[.5,.7),>=.7; all failures retained, searchable and all observations separate',
        'limits':'not independent model votes, upside-down inputs can be out of distribution; consistency not correctness; known synthetic development only',
        'code_hashes':{n:sha256(ROOT/n) for n in CODE},'input_hashes':{str(p):sha256(p) for p in paths}})
    from rtmlib import RTMPose
    model=RTMPose(str(args.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    manifest=read(args.source/'manifest.json');assert len(manifest)==360
    rows=[]
    for item in manifest:
        name=item['image'];path=args.source/'images'/name;assert sha256(path)==item['sha256']
        rgb=read_rgb(path);hypotheses=[infer_pose(rgb,model,crop=True,turns=(turn,),is_3d=False) for turn in range(4)]
        selected=max(hypotheses,key=lambda h:h['selection_score'])
        old=read(args.source/'predictions'/(Path(name).stem+'.json'))
        assert selected['turn']==old['turn'] and selected['selection_score']==old['selection_score'],name
        assert np.array_equal(selected['keypoints2d'],old['keypoints2d']),name
        assert np.array_equal(selected['scores'],old['scores']),name
        signal=agreement(old,hypotheses)
        dest=args.out/'hypotheses'/(Path(name).stem+'.json')
        save_json(dest,{'image':name,'hypotheses':hypotheses,'selected_turn':selected['turn'],'agreement':signal})
        rows.append({'image':name,'sha256':sha256(dest),'searchable':valid_query(old) is not None})
        if len(rows)%30==0:print('inferred',len(rows),'/360',flush=True)
    verify(args)
    save_json(args.out/'inferred.json',{'records':rows,'original_predictions_exactly_reproduced':360,'answers_parsed':False})


def band(j):
    d=j['disagreement']
    return 'insufficient' if d is None else 'lt01' if d<.1 else '01_to_025' if d<.25 else 'ge025'


def cells(joints):
    result={}
    for b in ('lt01','01_to_025','ge025','insufficient'):
        group=[j for j in joints if band(j)==b]
        result[b]={'n':len(group),'error_over_025':sum(j['gt_error']>.25 for j in group),
                   'median_gt_error':float(np.median([j['gt_error'] for j in group])) if group else None}
    return result


def summary(rows):
    joints=[j for r in rows for j in r['joints'] if j['valid']]
    return {'images':len(rows),'searchable':sum(r['searchable'] for r in rows),
            'alignment_eligible':sum(a['eligible'] for r in rows for a in r['alignments']),
            'alignment_swapped':sum(a.get('swapped',False) for r in rows for a in r['alignments']),
            'valid_joints':len(joints),'by_agreement':cells(joints),
            'searchable_only':cells([j for r in rows if r['searchable'] for j in r['joints'] if j['valid']]),
            'by_score':{name:cells([j for j in joints if lo<=j['score']<hi])
                        for name,lo,hi in [('03_to_05',.3,.5),('05_to_07',.5,.7),('ge07',.7,np.inf)]}}


def analyze(args):
    verify(args);assert not (args.out/'results.json').exists()
    inferred=read(args.out/'inferred.json');truth={r['image']:r for r in read(args.source/'answers.json')}
    assert len(truth)==len(inferred['records'])==360
    rows=[]
    for item in inferred['records']:
        name=item['image'];path=args.out/'hypotheses'/(Path(name).stem+'.json');assert sha256(path)==item['sha256']
        h=read(path);old=read(args.source/'predictions'/(Path(name).stem+'.json'));t=truth[name]
        gt=diagnose(old['keypoints2d'],old['scores'],t['ground_truth_xy'])
        joints=[]
        for signal,error in zip(h['agreement']['joints'],gt['joints']):
            assert signal['index']==error['index']
            joints.append({**signal,'valid':error['valid'],'score':error['score'],'gt_error':error['global_error']})
        rows.append({'image':name,'group':t['group'],'family_id':t['family_id'],'searchable':item['searchable'],
                     'selected_turn':h['selected_turn'],'alignments':h['agreement']['alignments'],'joints':joints})
    groups={g:summary([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    groups['all']=summary(rows)
    save_json(args.out/'results.json',{'groups':groups,'records':rows,'accuracy':None,'no_search_changes':True})
    print(json.dumps(groups['all'],indent=2),flush=True)


def parse():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('infer','analyze'))
    for name in ('source','out','model'):p.add_argument('--'+name,type=Path,required=True)
    return p.parse_args()


if __name__=='__main__':
    args=parse();args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.rotation-consistency.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.stage](args)
