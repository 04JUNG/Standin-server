"""Scalar agreement/GT checks plus inference replay on fixed first three images."""
import math
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_rotation_consistency import parse,read,verify
from scripts.experiment_z_rendered import sha256,save_json
from scripts.experiment_humanart_z import read_rgb
from standin_eval.z_diagnostics import infer_pose


def audit(args):
    verify(args);assert not (args.out/'audit.json').exists()
    inferred=read(args.out/'inferred.json');result=read(args.out/'results.json')
    answers={r['image']:r for r in read(args.source/'answers.json')}
    manifest={r['image']:r for r in read(args.source/'manifest.json')}
    rows={r['image']:r for r in result['records']}
    assert len(inferred['records'])==len(rows)==len(answers)==360
    from rtmlib import RTMPose
    model=RTMPose(str(args.model),model_input_size=(192,256),backend='onnxruntime',device='cpu')
    joints_checked=alignments_checked=replays=0
    for index,item in enumerate(inferred['records']):
        name=item['image'];ip=args.source/'images'/name;assert sha256(ip)==manifest[name]['sha256']
        path=args.out/'hypotheses'/(Path(name).stem+'.json');assert sha256(path)==item['sha256']
        data=read(path);hs=data['hypotheses'];assert [h['turn'] for h in hs]==[0,1,2,3]
        reference=read(args.source/'predictions'/(Path(name).stem+'.json'))
        for h in hs:assert h['selection_score']==float(np.median(h['scores'][5:]))
        winner=max(hs,key=lambda h:h['selection_score'])
        assert winner['keypoints2d']==reference['keypoints2d'] and winner['scores']==reference['scores']
        assert winner['turn']==reference['turn']==data['selected_turn']
        if index<3:
            rgb=read_rgb(ip)
            for h in hs:
                replay=infer_pose(rgb,model,crop=True,turns=(h['turn'],),is_3d=False)
                assert replay==h;replays+=1
        xy=reference['keypoints2d'];s=reference['scores'];row=rows[name]
        valid=[j for j in range(5,17) if s[j]>=.3]
        searchable=len(valid)>=6 and all(j in valid for j in (5,6,11,12))
        assert row['searchable']==item['searchable']==searchable
        torso=math.hypot((xy[5][0]+xy[6][0]-xy[11][0]-xy[12][0])/2,
                         (xy[5][1]+xy[6][1]-xy[11][1]-xy[12][1])/2)
        swap_index=lambda j:j+1 if j%2 else j-1
        observed={j:[] for j in range(5,17)}
        for alignment in data['agreement']['alignments']:
            h=hs[alignment['turn']];assert h['turn']!=reference['turn']
            options=[]
            for swap in (False,True):
                common=[j for j in valid if h['scores'][swap_index(j) if swap else j]>=.3]
                if len(common)<6:continue
                distances={j:math.dist(xy[j],h['keypoints2d'][swap_index(j) if swap else j])/torso for j in common}
                options.append((sum(distances.values())/len(common),swap,common,distances))
            assert alignment['eligible']==bool(options)
            if options:
                cost,swap,common,distances=min(options,key=lambda o:(o[0],o[1]))
                assert swap==alignment['swapped']
                assert alignment['common']==[j in common for j in range(17)]
                assert math.isclose(cost,alignment['mean_disagreement'],abs_tol=1e-12)
                for j in common:observed[j].append(distances[j])
            alignments_checked+=1
        assert row['alignments']==data['agreement']['alignments']
        gt=answers[name]['ground_truth_xy']
        length=math.hypot((gt[5][0]+gt[6][0]-gt[11][0]-gt[12][0])/2,
                          (gt[5][1]+gt[6][1]-gt[11][1]-gt[12][1])/2)
        direct=sum(math.dist(xy[j],gt[j]) for j in valid)
        alt=sum(math.dist(xy[j],gt[swap_index(j)]) for j in valid)
        swap=alt<direct
        for j in row['joints']:
            k=j['index'];values=observed[k]
            assert j['valid']==(k in valid) and j['score']==s[k] and j['support']==len(values)
            if len(values)<2:assert j['disagreement'] is None
            else:assert math.isclose(j['disagreement'],float(np.median(values)),abs_tol=1e-12)
            error=math.dist(xy[k],gt[swap_index(k) if swap else k])/length
            assert math.isclose(j['gt_error'],error,abs_tol=1e-12)
            joints_checked+=1
    def check_cells(saved,joints):
        for band,cell in saved.items():
            def included(j):
                d=j['disagreement']
                if band=='insufficient':return d is None
                if d is None:return False
                return d<.1 if band=='lt01' else .1<=d<.25 if band=='01_to_025' else d>=.25
            subset=[j for j in joints if included(j)]
            assert cell['n']==len(subset) and cell['error_over_025']==sum(j['gt_error']>.25 for j in subset)
            if subset:assert np.isclose(cell['median_gt_error'],np.median([j['gt_error'] for j in subset]),atol=1e-12)
            else:assert cell['median_gt_error'] is None
    for name,group in result['groups'].items():
        rr=list(rows.values()) if name=='all' else [r for r in rows.values() if r['group']==name]
        joints=[j for r in rr for j in r['joints'] if j['valid']]
        assert group['images']==len(rr) and group['searchable']==sum(r['searchable'] for r in rr)
        assert group['valid_joints']==len(joints)
        assert group['alignment_eligible']==sum(a['eligible'] for r in rr for a in r['alignments'])
        assert group['alignment_swapped']==sum(a.get('swapped',False) for r in rr for a in r['alignments'])
        check_cells(group['by_agreement'],joints)
        check_cells(group['searchable_only'],[j for r in rr if r['searchable'] for j in r['joints'] if j['valid']])
        for label,lo,hi in [('03_to_05',.3,.5),('05_to_07',.5,.7),('ge07',.7,math.inf)]:
            check_cells(group['by_score'][label],[j for j in joints if lo<=j['score']<hi])
    verify(args)
    report={'images_checked':360,'all_four_hypotheses_preserved':360,'selected_prediction_exactly_reproduced':360,
            'scalar_alignments_checked':alignments_checked,'scalar_joint_signals_and_gt_errors_checked':joints_checked,
            'local_inference_replays':replays,'all_summary_cells_recomputed':True,
            'results_sha256':sha256(args.out/'results.json'),'auditor_sha256':sha256(Path(__file__))}
    save_json(args.out/'audit.json',report);print(report,flush=True)


if __name__=='__main__':audit(parse())
