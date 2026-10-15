"""Local, semantic-first part search/composition probe; not production promotion.

No rough keypoints participate in retrieval, selection or synthesis here. Hidden
regions abstain, then receive an explicitly inferred neutral library reference.
Existing split-composition safety gates and returned-byte checks remain intact.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from .bvh import parse_bvh, fk, single_frame_bvh_text,coco17_from_fk
from .experimental.split_compose import compose_upper_lower
from .natural_parts import validate_part, digest

DEFAULTS={
    'upper':dict(group='arms_down',confidence='unknown',visible=True,observations=[],
        description='몸통을 바로 세우고 양팔을 몸 옆으로 편안히 내려 가만히 있는 기본 자세.'),
    'lower':dict(group='standing',confidence='unknown',visible=True,observations=[],
        description='두 다리를 나란히 펴고 양발을 바닥에 두고 가만히 서 있는 기본 자세.')}


def align_inferred_lower(base_joints,base_frame,donor_joints,donor_frame):
    """Only unobserved default legs may follow the observed upper's world frame.

    Otherwise a prone upper donor would be stood upright by a standing default
    pelvis. This is NOT allowed when lower is observed. Shapes/lengths remain
    unchanged and normal composition gates still run against the aligned base.
    """
    from .experimental.split_compose import _mapping,_children,_upper_root,_pelvis_frame
    from .experimental.part_composition import world_transforms,_set_rotation
    def pelvis(joints,frame):
        root,start,_=_upper_root(joints,_mapping(joints),_children(joints))
        world=world_transforms(joints,frame);points,_=coco17_from_fk(joints,fk(joints,frame))
        return root,world,_pelvis_frame(joints,world,root,start,points)
    root,world,basis=pelvis(base_joints,base_frame)
    _,_,target=pelvis(donor_joints,donor_frame)
    delta=target@basis.T;out=np.array(base_frame,dtype=float,copy=True)
    _set_rotation(base_joints,out,root,delta@world[root][:3,:3])
    _,_,after=pelvis(base_joints,out)
    error=float(np.max(np.abs(after-target)))
    if error>1e-6:raise ValueError('inferred lower frame alignment failed')
    return out,dict(rule='unobserved_default_pelvis_follows_observed_upper_world_frame',
        alignment_error=error,root_rotation_changed=True,observed_lower=False)


class NaturalPartSearch:
    def __init__(self,index,*,top_k=5,pool_size=8,max_attempts=64,budget_ms=30000):
        if not 1<=top_k<=5 or not top_k<=pool_size<=20:raise ValueError('invalid candidate limits')
        self.index=index;self.top_k=top_k;self.pool_size=pool_size
        self.max_attempts=max_attempts;self.budget_ms=budget_ms
        self._cache={};self._defaults={}

    def _load(self,pid):
        if pid not in self._cache:
            path=self.index.source(pid);joints,frames=parse_bvh(str(path))
            self._cache[pid]=(path,joints,frames[0],digest(path))
        return self._cache[pid]

    def default(self,part):
        if part not in self._defaults:
            candidates=self.index.search(DEFAULTS[part],part,top_k=60)['hits']
            candidates=[c for c in candidates if c['tag']['group']==DEFAULTS[part]['group']]
            self._defaults[part]=[dict(candidates[0],inferred=True,
                inference_reason='unseen_region_neutral_fill_not_observed',default_query=DEFAULTS[part])
                ] if candidates else []
        return deepcopy(self._defaults[part])

    def search(self,description):
        started=perf_counter();out=dict(version='natural-part-search-v1',route='natural_language',
            skeleton_used_for_ranking=False,generation_enabled=False,release_allowed=False,
            actual_skin_collision_validated=False,upper={},lower={},selected=[],attempts=[],
            budget_exhausted=False,semantic_match_verified=False)
        if description.get('subject_kind','human')!='human':
            out.update(status='policy_skip',reason='nonhuman_or_uncertain_owned_target',
                subject_kind=description['subject_kind']);return out
        observed={p:description[p]['visible'] and description[p].get('searchable',True) for p in ('upper','lower')}
        if not any(observed.values()):
            for p in observed:out[p]=dict(status='abstained',hits=[],query=description[p],reason='no_searchable_part')
            out.update(status='not_run',reason='no_visible_searchable_body_part');return out
        pools={}
        for part in ('upper','lower'):
            validate_part(description[part],part,rough=True)
            report=self.index.search(description[part],part,top_k=self.pool_size)
            pool=report['hits']
            out[part]=dict(report,hits=pool[:self.top_k],query=description[part])
            if report['status']=='abstained':
                pool=self.default(part);out[part]['default_fill']=pool
            pools[part]=pool
        out['output_scope']='half' if not observed['lower'] else 'full'
        out['remove_lower_body']=out['output_scope']=='half'
        if any(not pool for pool in pools.values()):
            out.update(status='no_result',reason='empty_part_or_neutral_default_unavailable');return out
        pairs=[]
        for ui,upper in enumerate(pools['upper']):
            for li,lower in enumerate(pools['lower']):
                visible_scores=[h['score'] for p,h in (('upper',upper),('lower',lower)) if observed[p]]
                score=float(np.mean(visible_scores))
                pairs.append((-score,ui+li,ui,li,upper,lower))
        seen=set()
        for minus_score,_,ui,li,upper,lower in sorted(pairs,key=lambda row:row[:4])[:self.max_attempts]:
            if (perf_counter()-started)*1000>self.budget_ms:
                out['budget_exhausted']=True;break
            attempt=dict(upper_id=upper['pose_id'],lower_id=lower['pose_id'],upper_rank=ui+1,lower_rank=li+1,
                semantic_score=-minus_score)
            out['attempts'].append(attempt)
            try:
                bp,bj,bf,bh=self._load(lower['pose_id']);up,uj,uf,uh=self._load(upper['pose_id'])
                attempt['source_sha256']={'upper':uh,'lower':bh}
                if upper['pose_id']==lower['pose_id']:
                    frame=bf;text=single_frame_bvh_text(str(bp),bf)
                    diagnostics=dict(pass_through=True,**{'pass':True,'reasons':[],
                        'note':'same_source_no_splice_not_anatomical_certification'})
                    kind='library_reference'
                else:
                    alignment=None
                    if not observed['lower'] and observed['upper']:
                        bf,alignment=align_inferred_lower(bj,bf,uj,uf)
                    composition=compose_upper_lower(bj,bf,uj,uf);diagnostics=composition.diagnostics
                    if alignment:diagnostics=dict(diagnostics,inferred_lower_alignment=alignment)
                    attempt['diagnostics']=diagnostics
                    if not diagnostics['pass']:
                        attempt['status']='rejected';continue
                    frame=composition.frame;text=single_frame_bvh_text(str(bp),frame)
                    kind='composed_preview'
                with tempfile.TemporaryDirectory(prefix='natural-parts-roundtrip-') as temp:
                    saved=Path(temp)/'pose.bvh';saved.write_text(text)
                    rj,rf=parse_bvh(str(saved))
                before,after=fk(bj,frame),fk(rj,rf[0])
                scale=max(np.linalg.norm(before[1]-before[0]),1e-6)
                delta=float(max(np.linalg.norm(before[i]-after[i]) for i in before)/scale)
                if delta>1e-6:raise ValueError('serialization_check_failed')
                if digest(bp)!=bh or digest(up)!=uh:raise ValueError('source_changed_during_composition')
                sha=hashlib.sha256(text.encode()).hexdigest()
                if sha in seen:attempt['status']='duplicate';continue
                seen.add(sha)
                out['selected'].append(dict(pose_id='parts_'+sha[:20] if kind=='composed_preview' else lower['pose_id'],
                    kind=kind,bvh=text,bvh_sha256=sha,upper_id=upper['pose_id'],lower_id=lower['pose_id'],
                    upper_rank=ui+1,lower_rank=li+1,semantic_score=-minus_score,
                    upper_description=upper['tag']['description'],lower_description=lower['tag']['description'],
                    lateral={'upper':upper.get('lateral'),'lower':lower.get('lateral')},
                    inferred_parts=[p for p in pools if not observed[p]],
                    source_sha256={'upper':uh,'lower':bh},diagnostics=diagnostics,roundtrip_delta_torso=delta,
                    refine_allowed=False,release_allowed=False,actual_mesh_validated=False,
                    semantic_match_verified=False))
                attempt['status']='accepted'
                if len(out['selected'])>=self.top_k:break
            except Exception as error:
                attempt.update(status='error',reason=type(error).__name__+':'+str(error)[:160])
        out.update(status='preview_ready' if out['selected'] else 'no_result',
            elapsed_ms=(perf_counter()-started)*1000)
        return out
