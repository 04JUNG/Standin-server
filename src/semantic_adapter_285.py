"""04J-285 text service -> 04J-286 routing contract, explicit opt-in.

Keep the existing query/result types and geometry path unchanged. Search the
whole above-threshold pool BEFORE BVH relation/body validation and family Top-K.
No user feedback ranking is enabled by default. Scores stay 285's scale.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

from .body_language import query_observations, compare_observations, specific_observation_count
from .config import CFG
from .pose_quarantine import load_pose_quarantine
from .semantic_search import SemanticHit, SemanticSearchResult, UnavailableSemanticSearcher
from .semantic_search_285 import SemanticIndex, SemanticSearchService, SemanticSettings
from .semantic_shapes import critical_relations, choose_camera, camera_hint, load_shape_index
from .semantic_body_index import load_body_index

ROOT=Path(__file__).resolve().parents[1]
BACKEND='semantic-285-text+bvh-validation-v1'


class Semantic285Adapter:
    backend=BACKEND

    def __init__(self,service,*,shape_metadata=None,body_metadata=None):
        self.service=service;self.shapes=shape_metadata or {};self.body=body_metadata or {}
        self.pool_service=None
        if service.index is not None:
            settings=replace(service.settings,max_top_k=max(service.settings.max_top_k,service.index.pose_count),collapse_mirrors=False)
            self.pool_service=SemanticSearchService(service.index,settings)

    def search(self,query,top_k=5,*,source='vlm'):
        started=perf_counter()
        def result(status,hits=None,reason=None):
            return SemanticSearchResult(status,hits or [],reason,self.backend,(perf_counter()-started)*1000)
        if isinstance(top_k,bool) or not isinstance(top_k,int) or top_k<=0:
            return result('error',reason='semantic285:invalid_top_k')
        if self.pool_service is None:
            raw=self.service.search(query.text,source=source,top_k=top_k)
            return result('error',reason='semantic285:'+(raw.get('error') or {}).get('code','index_not_loaded'))
        required=critical_relations(query)
        if required and not self.shapes:return result('error',reason='semantic285:critical_shape_index_missing')
        observed=query_observations(query.body_details,query.body_observations)
        if required:observed.pop('arms',None) # Fold/V relation outranks coarse raised/spread.
        use_body=bool(self.body) and (bool(required) or specific_observation_count(observed)>=query.min_body_matches)
        raw=self.pool_service.search(query.text,source=source,top_k=self.service.index.pose_count)
        if raw['status']=='error':
            return result('error',reason='semantic285:'+(raw.get('error') or {}).get('code','unknown'))
        hits=[];seen=set();postures=set(query.postures)-set(query.generic_postures)
        for candidate in raw['candidates']:
            pose_id=candidate['pose_id'];camera=None
            matched={'semantic285':{'score_components':candidate['score_components'],'evidence':candidate['evidence'],
                    'query':raw['query'],'index':raw['index'],'threshold':raw['threshold'],
                    'refine_allowed':False,'whole_pool_before_top_k':True}}
            if required:
                profile=self.shapes.get(pose_id)
                if not profile or not all(profile['relations'].get(k) is True for k in required):continue
                tags={t['value'] for group in candidate['pose'].get('tags',{}).values() for t in group if t.get('field')=='posture'}
                if postures and not postures & tags:continue
                camera=choose_camera(profile,required,camera_hint(query))
                if camera is None:continue
                matched['critical_relations']={'required':sorted(required),'verified':sorted(required),
                    'source':'reviewed_action+BVH' if 'fold_arms' in required else 'BVH_body_relative'}
                matched['camera_comparison']={'kind':'qualitative_visibility_not_pixel_fit','selected':camera['camera'],
                                               'hint':camera_hint(query),'views':profile['views']}
            if use_body:
                measured=self.body.get(pose_id)
                if measured is not None:
                    check=compare_observations(observed,measured['observations'])
                    # Numeric scores from two backends are never added together.
                    # Known contradictory geometry is rejected; unknown is neutral.
                    if check['conflicts']:continue
                    matched['bvh_body_validation']={**check,'source':'BVH','unknown_is_neutral':True}
            family=self.service.index.family.get(pose_id,pose_id)
            if family in seen:continue
            seen.add(family)
            pose={**candidate['pose'],'refine_allowed':False}
            if camera is not None:pose.update(preview_camera=camera['camera'],view=camera['camera']['view'])
            hits.append(SemanticHit(pose_id,candidate['score'],matched,pose))
            if len(hits)>=top_k:break
        if hits:return result('ok',hits)
        reason='semantic285:critical_combination_unverified' if required else 'semantic285:no_pose_above_score_and_body_validation'
        return result('no_result',reason=reason)


def build_semantic285_searcher(*,tags_path,taxonomy_path,db_path,library_pose_ids=None,
        lexicon_path=None,shape_index_path=None,body_index_path=None,dense_model_dir=None,
        dense_cache_dir=None,settings=None,review_preferences_path=None):
    """Bind tags/DB/caches, union safety exclusions. No model download or writes."""
    try:
        settings=settings or SemanticSettings()
        settings.validate();dense=None
        if dense_model_dir:
            from .semantic_embedding import E5OnnxEncoder
            dense=E5OnnxEncoder(str(dense_model_dir),cache_dir=str(dense_cache_dir or ROOT/'tmp/semantic285-dense-cache'),threads=1)
        excluded=set(load_pose_quarantine(CFG))
        policy=json.loads((ROOT/'config/semantic_quarantine_285.v1.json').read_text())
        if policy.get('schema_version')!=1 or not isinstance(policy.get('entries'),list):raise ValueError('invalid imported quarantine')
        for entry in policy['entries']:
            if not isinstance(entry.get('pose_id'),str) or not entry.get('reason'):raise ValueError('invalid imported quarantine entry')
            excluded.add(entry['pose_id'])
        index=SemanticIndex.load(tags_path=str(tags_path),taxonomy_path=str(taxonomy_path),db_path=str(db_path),
            lexicon_path=str(lexicon_path or ROOT/'config/semantic_lexicon.v1.json'),
            allowed_pose_ids=library_pose_ids,excluded_pose_ids=excluded,dense_encoder=dense,
            thumbs_dir=str(Path(db_path).parent/'thumbs'),review_preferences_path=review_preferences_path)
        raw=json.loads(Path(tags_path).read_text());records=raw['poses']
        shapes=load_shape_index(shape_index_path,records) if shape_index_path else None
        body=load_body_index(body_index_path,records) if body_index_path else None
        return Semantic285Adapter(SemanticSearchService(index,settings),shape_metadata=shapes,body_metadata=body)
    except Exception as error:
        code=getattr(error,'code',type(error).__name__)
        return UnavailableSemanticSearcher('semantic285_init:'+str(code))


def build_from_env(tags_path,*,library_pose_ids,body_index_path,shape_index_path):
    dense_on=os.getenv('SEMANTIC_DENSE_ENABLED','0').lower() in ('1','true','yes','on')
    if dense_on and not os.getenv('SEMANTIC_EMBEDDING_DIR'):
        return UnavailableSemanticSearcher('semantic285_init:embedding_dir_missing')
    settings=SemanticSettings(
        min_score_vlm=float(os.getenv('SEMANTIC_285_MIN_SCORE_VLM','0.35')),
        weight_dense=float(os.getenv('SEMANTIC_WEIGHT_DENSE','0')) if dense_on else 0.,
        dense_lo=float(os.getenv('SEMANTIC_DENSE_LO','0.78')),dense_hi=float(os.getenv('SEMANTIC_DENSE_HI','0.90')))
    return build_semantic285_searcher(tags_path=tags_path,
        taxonomy_path=os.getenv('SEMANTIC_TAXONOMY_PATH',str(Path(tags_path).with_name('search-taxonomy-v2.json'))),
        db_path=os.getenv('SEMANTIC_285_DB_PATH') or os.getenv('DB_PATH') or str(Path(CFG.data_dir)/'poses.db'),library_pose_ids=library_pose_ids,
        lexicon_path=os.getenv('SEMANTIC_LEXICON_PATH') or None,
        body_index_path=body_index_path,shape_index_path=shape_index_path,settings=settings,
        dense_model_dir=(os.getenv('SEMANTIC_EMBEDDING_DIR') or None) if dense_on else None)
