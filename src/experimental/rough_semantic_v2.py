"""Factorized rough semantic retrieval with BVH structural verification.

E5 recalls free-form meanings; support and each recognized limb requirement are
scored separately over every eligible library member. No Top-5-only reranking,
no view-word retrieval, no implicit composition, no exact-match claims.
"""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np
from .rough_semantic import RoughSemanticSearch,sha
from . import semantic_pose_structure as structure
from .. import bvh,posecode


class StructuredRoughSemanticSearch(RoughSemanticSearch):
    version='rough-semantic-structured-v2'

    def __init__(self,**kwargs):
        self.structures={}
        super().__init__(**kwargs)
        self._queries=OrderedDict()
        self._region_indices={r:[i for i,d in enumerate(self.documents) if d['region']==r] for r in ('full','upper','lower')}

    def extension_identity(self):
        return {Path(p).name:sha(p) for p in (__file__,structure.__file__,bvh.__file__,posecode.__file__)}

    def make_documents(self,row,path):
        facts=structure.structure_from_bvh(path);self.structures[row['pose_id']]=facts
        for region in ('full','upper','lower'):
            phrases=(facts['upper'] if region=='upper' else facts['lower'] if region=='lower' else facts['lower']+facts['upper'])
            yield dict(pose_id=row['pose_id'],region=region,bvh_sha256=row['bvh_sha256'],
                text=region+' body pose: '+'; '.join(phrases))

    def _encode_query(self,text):
        if text not in self._queries:
            self._queries[text]=self.encoder.encode([text],kind='query')[0][0]
            if len(self._queries)>256:self._queries.popitem(last=False)
        self._queries.move_to_end(text)
        return self._queries[text]

    def _dense_scores(self,clauses):
        return np.mean([self.vectors @ self._encode_query(c) for c in clauses],axis=0)

    def _candidate_metadata(self,pid,region):
        return {}

    def search(self,query,*,region='full',top_k=5,support_hypotheses=None,evidence_notes=()):
        if region not in self._region_indices or not 1<=top_k<=100:raise ValueError('invalid search request')
        query=' '.join(query.split())
        if not query:raise ValueError('empty semantic query')
        from ..pose_quarantine import pose_quarantine_sha256
        if sha(self.geometry_db)!=self.db_hash or pose_quarantine_sha256()!=self.quarantine_hash:raise ValueError('live library changed')
        clauses=[x.strip() for x in query.split(';') if x.strip() and not structure.VIEW_ONLY.fullmatch(x.strip())]
        output=dict(query=query,region=region,semantic_build_id=self.build_id,results=[],
            ranking_policy=self.version,score_kind='dense_similarity_not_rank_or_probability')
        if not clauses:
            return {**output,'status':'view_only_not_pose_query','constraints':{},'unverified_meanings':[]}
        clauses.extend(n for n in evidence_notes if n and n not in clauses)
        output['evidence_notes']=list(evidence_notes)
        clean='; '.join(clauses)
        constraints=structure.parse_constraints(clean,region)
        support_weight=1.
        if support_hypotheses is not None and region!='upper':
            recognized=[(h,structure.parse_constraints(h['value'])['support']) for h in support_hypotheses]
            recognized=[(h,v) for h,v in recognized if v]
            strong=[(h,v) for h,v in recognized if h.get('status')=='supported' and h.get('evidence_kind')!='scene_context']
            selected=strong or recognized
            constraints['support']=sorted({v for h,values in selected for v in values})
            support_weight=1. if strong else .3
        indices=self._region_indices[region]
        # Average independent clauses so repeated text cannot win by repetition.
        clauses=list(dict.fromkeys(clauses))
        dense=self._dense_scores(clauses)
        ranked=[]
        for i in indices:
            doc=self.documents[i];pid=doc['pose_id'];f=self.structures[pid]
            support, support_state=structure.support_cost(constraints['support'],f['support'])
            costs={key:f['costs'][key] for key in constraints['features']}
            worst=max(costs.values(),default=0);mean=float(np.mean(list(costs.values()))) if costs else 0.
            loss=.60*support_weight*support+.30*worst+.25*mean-.12*float(dense[i])
            ranked.append((loss,pid,i,support_state,support,costs))
        # Explicit support evidence is reviewed before fine limb similarity.
        # Tentative/context-only support stays a soft preference.
        def order(item):
            tier=(0 if item[3]=='compatible' else 1 if item[3] in ('unknown','partial','different_sitting_support') else 2) if constraints['support'] and support_weight==1. else 0
            return tier,item[0],item[1]
        ranked.sort(key=order)
        results=[];families=set()
        for loss,pid,i,state,support,costs in ranked:
            family=pid.removesuffix('_mirror')
            if family in families:continue
            families.add(family);doc=self.documents[i];path=self.paths[pid]
            if sha(path)!=doc['bvh_sha256'].removeprefix('sha256:'):raise ValueError('candidate BVH changed')
            violations=[k for k,v in costs.items() if v>.45]
            results.append(dict(pose_id=pid,score=float(dense[i]),rank_cost=loss,
                bvh_path=str(path.resolve()),bvh_sha256=doc['bvh_sha256'],document=doc['text'],
                query_region=region,candidate_scope='whole_pose_member',match_source='semantic_rough',
                exact_match_status='approximate_unverified',refine_allowed=False,
                support_geometry=self.structures[pid]['support'],support_comparison=state,
                support_cost=support,constraint_costs=costs,violations=violations,
                preview_urls={v:f'/pose/{quote(pid,safe="")}/thumbnail?view={v}' for v in ('front','three_quarter')},
                ground_contact_verified=False,**self._candidate_metadata(pid,region)))
            if len(results)>=top_k:break
        unknown=[c for c in clauses if not any(structure.parse_constraints(c,region).values())]
        # Partial body geometry coverage cannot verify objects or another person.
        import re
        unknown=list(dict.fromkeys(unknown+[c for c in clauses if re.search(r'object|sword|phone|device|book|person|child|stretcher|desk|door|hugging(?! oneself)|pockets',c,re.I)]))
        best=results[0] if results else None
        gap=bool(best and (best['support_cost']>=.45 or best['violations']))
        return {**output,'results':results,'status':'approximate_with_structure_gap' if gap else 'approximate_candidates',
            'constraints':constraints,'support_weight':support_weight,'unverified_meanings':unknown,
            'searched_members':len(self.paths),'family_diversity':True,'library_gap_suspected':gap,
            'coverage_note':'Recognized geometry only; props, contact surfaces and action intent remain unverified.'}


def execute_structured_plan(plan,runtime,*,slots=None,top_k=5):
    row=plan.to_dict() if hasattr(plan,'to_dict') else plan
    slots=slots.to_dict() if hasattr(slots,'to_dict') else slots
    support=dict(slots.get('meanings',[])).get('support_state',[]) if slots else None
    requests=[]
    for channel in row['channels']:
        if channel['kind']!='S':continue
        for query in dict.fromkeys(channel['queries']):
            fields=('upper_action','lower_configuration') if channel['region']=='full' else (('upper_action',) if channel['region']=='upper' else ('lower_configuration',))
            meanings=dict(slots.get('meanings',[])) if slots else {}
            notes=[m.get('evidence_note','') for name in fields for m in meanings.get(name,[]) if m['value'] in query and m.get('evidence_kind')!='scene_context']
            requests.append(runtime.search(query,region=channel['region'],top_k=top_k,support_hypotheses=support,evidence_notes=notes))
    # A coherent whole-pose request combines each alternative independently;
    # pick no hypothesis here. The UI displays all alternatives, not a fake merge.
    return dict(person_index=row['person_index'],route=row['route'],semantic_requests=requests,
        executed_query_count=len(requests),composition_executed=False,geometry_executed=False,
        facts_rerank_executed=True,refine_allowed=False,ranking_policy=runtime.version)


def build_rough_semantic_runtime(*,implementation='structured_v2',**kwargs):
    if implementation=='structured_v2':return StructuredRoughSemanticSearch(**kwargs)
    if implementation=='dense_v1':return RoughSemanticSearch(**kwargs)
    raise ValueError('unknown rough semantic implementation')
