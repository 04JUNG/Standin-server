"""Independent post-run GT evaluator. Never imported by inference/solver."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.refine_v5.learning_data import BONES,POINTS,axes,frame,visibility,point_visibility

def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))
def mean(a,m):return float(np.asarray(a)[np.asarray(m,bool)].mean()) if np.asarray(m,bool).any() else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--previous',required=True);p.add_argument('--repo',required=True);p.add_argument('--fusion',choices=['observation','learned_balanced','geometric'],default='observation');a=p.parse_args();run=Path(a.run);previous=Path(a.previous);repo=Path(a.repo)
    rigs={r['case_id']:r for r in json.loads((run/'native-data/records.json').read_text()) if r['status']=='ready'}
    gt={r['case_id']:r for r in json.loads((previous/'data/native_audit.json').read_text())};reviews={r['case_id']:r for r in json.loads((repo/'evaluation/refine_v5/visual_review_v2.json').read_text())['records']}
    directory={'observation':'iterative','learned_balanced':'iterative-balanced','geometric':'iterative-geometric'}[a.fusion]
    results=json.loads((run/directory/'results.json').read_text());records=[]
    for row in results:
        case=row['case_id'];rig=rigs[case];mask=visibility(reviews[case]);base=axes(rig['bones']);goal=axes(gt[case]['bones']);finalbones=[{**b,'matrix':row.get('planned_matrices',{}).get(b['name'],b['matrix'])} for b in rig['bones']];final=axes(finalbones)
        before=angle(base[:,1],goal[:,1]);after=angle(final[:,1],goal[:,1])
        b0,c,s,f=frame(rig['bones']);bg,cg,sg,fg=frame(gt[case]['bones'])
        gf=(np.array([bg[n]['head'] for n in POINTS])-cg)@fg/sg
        bp=(np.array([np.array(b0[n]['matrix'])[:3,3] for n in POINTS])-c)@f/s
        # Fixed original source body frame, rather than re-aligning the candidate.
        bf={b['name']:b for b in finalbones};fp=(np.array([np.array(bf[n]['matrix'])[:3,3] for n in POINTS])-c)@f/s
        pointmask=point_visibility(mask)
        eb=np.linalg.norm(bp-gf,axis=-1);ef=np.linalg.norm(fp-gf,axis=-1)
        records.append(dict(case_id=case,split=rig['split'],profile=row.get('profile','unobservable'),stage=row['stage'],rounds=row['rounds'],status=row['status'],observed_limbs=int(mask[:8].sum()),observed_arms=int(mask[:4].sum()),before_angles=before.tolist(),after_angles=after.tolist(),mask=mask.tolist(),limb_before=mean(before[:8],mask[:8]),limb_after=mean(after[:8],mask[:8]),arm_before=mean(before[:4],mask[:4]),arm_after=mean(after[:4],mask[:4]),joint_before=mean(eb,pointmask),joint_after=mean(ef,pointmask),final_objective=row.get('final_objective'),target_applied=row['target_applied'],trace=row['trace']))
    summary=[]
    for split in ['development','validation']:
        for profile in ['conservative12','experimental30']:
            for stage in ['source','selected_target']:
                for count in [1,3]:
                    rows=[r for r in records if r['split']==split and r['profile']==profile and r['stage']==stage and r['rounds']==count and r['observed_limbs']]
                    if not rows:continue
                    def weighted(key,weight):
                        return sum(r[key]*r[weight] for r in rows if r[key] is not None)/sum(r[weight] for r in rows if r[key] is not None)
                    summary.append(dict(split=split,profile=profile,stage=stage,rounds=count,cases=len(rows),total_source_ready_cases=sum(r['split']==split for r in rigs.values()),unobserved_cases=sum(r['split']==split for r in rigs.values())-len(rows),limb_before=weighted('limb_before','observed_limbs'),limb_after=weighted('limb_after','observed_limbs'),arm_before=weighted('arm_before','observed_arms'),arm_after=weighted('arm_after','observed_arms'),joint_before=float(np.mean([r['joint_before'] for r in rows if r['joint_before'] is not None])),joint_after=float(np.mean([r['joint_after'] for r in rows if r['joint_after'] is not None])),improved=sum(r['limb_after']<r['limb_before']-1e-5 for r in rows),worsened=sum(r['limb_after']>r['limb_before']+1e-5 for r in rows),unchanged=sum(abs(r['limb_after']-r['limb_before'])<=1e-5 for r in rows)))
    out=dict(schema='v5-p2-p5-independent-evaluation-v1',test_used=False,rough_camera_calibrated=False,orientation_observed=False,metric='native_body_relative_visible_bone_direction_deg_and_root_torso_normalized_3d_joint_NME',artist_blind=False,summary=summary,records=records)
    (run/{'observation':'evaluation.json','learned_balanced':'evaluation-balanced.json','geometric':'evaluation-geometric.json'}[a.fusion]).write_text(json.dumps(out,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
