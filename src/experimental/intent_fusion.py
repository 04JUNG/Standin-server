"""Offline proposal path joining semantic retrieval with unchanged A/B1.

No API hook, semantic asset I/O, confidence mutation or refine authorization.
A keeps its original one-way gate. Optional support ordering is a separate
soft proposal, not a verified stand/sit classifier. B1 only reorders within
support-compatible retrieval bands, preserving coarse semantic relevance.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from ..search import knn_geometric, pose_family_id
from .a_minimal_support import classify_minimal_support_2d
from .search_bundle import ExperimentalSearchBundle

POLICY_VERSION = 'intent-fusion-shadow-v1'

@dataclass(frozen=True)
class SemanticBatch:
    query: str
    source: str  # semantic_user or semantic_vlm; never geometry
    geometry_db_sha256: str
    build_id: str
    pose_ids: tuple[str, ...]  # semantic ranking, concrete original/mirror IDs

    def __post_init__(self):
        if self.source not in ('semantic_user','semantic_vlm'):
            raise ValueError('semantic source must be preserved')
        if not self.query.strip() or not self.build_id or not self.geometry_db_sha256:
            raise ValueError('semantic query/build/lineage are required')
        if len(self.pose_ids)>100 or len(set(self.pose_ids))!=len(self.pose_ids):
            raise ValueError('semantic pool must contain at most 100 unique members')

@dataclass(frozen=True)
class QueryEligibility:
    skeleton_state: str
    coverage_class: str
    search_scope: str

    def allows_a(self, mask):
        return (self.skeleton_state=='valid' and self.coverage_class=='full'
                and self.search_scope=='full_body' and bool(mask[11:17].all()))

@dataclass(frozen=True)
class IntentFusionResult:
    # Caller continues to publish original geometry results in shadow mode.
    published_candidates: Sequence
    proposals: tuple[dict, ...]
    trace: dict

class IntentFusionIndex:
    def __init__(self, entries, *, geometry_db_sha256):
        if not geometry_db_sha256:raise ValueError('geometry lineage required')
        self.entries=tuple(entries)
        self.geometry_db_sha256=geometry_db_sha256
        self.bundle=ExperimentalSearchBundle.build(self.entries,enable_a=True,enable_b1=True)
        self.by_pose={}
        for e in self.entries:self.by_pose.setdefault(e.pose_id,[]).append(e)

    def evaluate(self, feature, valid_mask, baseline, *, eligibility,
                 semantic=None, top_k=5, support_ordering=True,
                 enable_a=True, enable_b1=True, semantic_band_size=3):
        if not 1<=top_k<=100 or not 1<=semantic_band_size<=10:
            raise ValueError('invalid result count or retrieval band size')
        mask=np.asarray(valid_mask,dtype=bool)
        if mask.shape!=(17,):raise ValueError('COCO17 mask required')
        if not baseline:return IntentFusionResult(baseline,(),{'reason':'no_geometry_baseline'})
        trace={'policy_version':POLICY_VERSION,'mode':'shadow',
               **self.bundle.trace_identity(),'geometry_db_sha256':self.geometry_db_sha256,
               'support_is_soft':True,'intent_match_verified':False,
               'requires_geometry_safety_revalidation':True}
        a_candidates=tuple(baseline)
        if enable_a and eligibility.allows_a(mask):
            a=self.bundle.a_support_gate.evaluate(feature,mask,baseline,metric='pos',max_distance_ratio=1.25)
            a_candidates=a.suggested_candidates;trace['a']=a.trace
        else:trace['a']={'gate_eligible':False,'rollback_reason':'disabled_or_ineligible'}
        observation=classify_minimal_support_2d(feature,mask)
        requested=observation.value if eligibility.allows_a(mask) and support_ordering else 'unknown'
        trace['query_support']=observation.to_trace()
        trace['support_ordering_active']=requested!='unknown'
        sem_rank={};sem_candidates=[];semantic_reason='not_supplied'
        if semantic is not None:
            # Whole-batch rejection avoids silently testing another pose universe.
            if semantic.geometry_db_sha256!=self.geometry_db_sha256:
                semantic_reason='geometry_lineage_mismatch'
            elif any(pid not in self.by_pose for pid in semantic.pose_ids):
                semantic_reason='unknown_pose_member'
            elif not semantic.pose_ids:semantic_reason='empty_results'
            else:
                semantic_reason='accepted'
                for rank,pid in enumerate(semantic.pose_ids,1):
                    # Preview view is never used as a camera match. Reproject the
                    # exact semantic member against the same observed query.
                    projected=knn_geometric(self.by_pose[pid],feature,top_k=1,
                                           query_valid_mask=mask,metric='pos')
                    if projected:
                        sem_rank[pid]=rank;sem_candidates.extend(projected)
        trace['semantic']={'reason':semantic_reason,'query':semantic.query if semantic else None,
                           'source':semantic.source if semantic else None,
                           'build_id':semantic.build_id if semantic else None}
        a_rejected=set()
        if trace['a'].get('gate_eligible'):
            a_rejected={pid for pid,o in self.bundle.a_support_gate.support_by_pose_id.items() if o.value=='stand'}
        # Semantic retrieval must not reintroduce a member A excluded.
        sem_candidates=[c for c in sem_candidates if c.pose_id not in a_rejected]
        trace['semantic']['a_excluded_members']=sorted(set(sem_rank)&a_rejected)
        pool={c.pose_id:c for c in a_candidates}
        for c in sem_candidates:pool.setdefault(c.pose_id,c)
        candidates=list(pool.values())
        geo_rank={c.pose_id:i+1 for i,c in enumerate(a_candidates)}
        # Ranks, not raw dense score + L2 distance, define fusion. Uncalibrated
        # relevance bands are experimental and never an exact semantic claim.
        def rrf(c):
            return (1/(60+geo_rank[c.pose_id]) if c.pose_id in geo_rank else 0)+(1/(60+sem_rank[c.pose_id]) if c.pose_id in sem_rank else 0)
        fused=sorted(candidates,key=lambda c:(-rrf(c),c.distance,c.pose_id))
        fused_rank={c.pose_id:i for i,c in enumerate(fused)}
        a_support=self.bundle.a_support_gate.support_by_pose_id
        def tier(c):
            support=a_support[c.pose_id].value
            return 0 if requested=='unknown' or support==requested else 1 if support=='unknown' else 2
        groups={}
        for c in candidates:
            # Without semantic input, B1 retains full scope within each support
            # stratum. With it, B1 cannot jump across coarse relevance bands.
            band=fused_rank[c.pose_id]//semantic_band_size if sem_candidates else 0
            groups.setdefault((tier(c),band),[]).append(c)
        order=[];b1_traces=[]
        for key,group in sorted(groups.items()):
            group.sort(key=lambda c:fused_rank[c.pose_id])
            if enable_b1:
                b1=self.bundle.b1_pose_fact_reranker.evaluate(feature,mask,group)
                order.extend(b1.suggested_candidates);b1_traces.append({'support_tier':key[0],'retrieval_band':key[1],**b1.trace})
            else:order.extend(group)
        proposals=[];seen=set()
        for c in order:
            family=c.pose_family_id or pose_family_id(c.pose_id)
            if family in seen:continue
            seen.add(family)
            sources=[]
            if c.pose_id in geo_rank:sources.append('geometry')
            if c.pose_id in sem_rank and c.pose_id not in a_rejected:sources.append(semantic.source)
            proposals.append({'pose_id':c.pose_id,'pose_family_id':family,'view':c.view.value,
                'distance':float(c.distance),'bvh_path':c.bvh_path,'sources':sources,
                'refine_allowed':False,'support':a_support[c.pose_id].to_trace(),
                'support_tier':tier(c),'support_match_verified':False,
                'geometry_rank':geo_rank.get(c.pose_id),'semantic_rank':sem_rank.get(c.pose_id),
                'rrf':rrf(c),'retrieval_band':fused_rank[c.pose_id]//semantic_band_size if sem_candidates else 0})
            if len(proposals)==top_k:break
        trace['b1_groups']=b1_traces
        trace['pool_size']=len(pool)
        trace['would_change_top1']=bool(proposals and proposals[0]['pose_id']!=baseline[0].pose_id)
        trace['support_candidate_status']=('not_requested' if requested=='unknown' else
            'compatible_proposal_available' if any(p['support_tier']==0 for p in proposals) else
            'no_known_compatible_proposal')
        return IntentFusionResult(baseline,tuple(proposals),trace)
