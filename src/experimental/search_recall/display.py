"""Compose the first candidate search with distinct geometric and text limits."""

from copy import copy
from dataclasses import replace
from time import perf_counter
from types import MethodType
from .evaluate import VARIANTS
from .evaluate import geometry_variants
from .evaluate import SoftSemanticValidator
from src.semantic_search import SemanticQuery
from src.semantic_search_285 import SemanticSearchService

def prepare_engine(original_engine, variant, joint):
    pool, geo_mode, text_mode, text_weight=VARIANTS[variant]
    engine=copy(original_engine)
    engine.pool_k=pool
    engine.semantic_weight=text_weight
    engine.geometry_weight=1-text_weight
    engine.ranking=f'weighted_rrf_k60_geometry_{1-text_weight:g}_semantic_{text_weight:g}'
    if geo_mode!='arm':
        def geometry(self,evidence,required=None):
            started=perf_counter()
            rows=geometry_variants(self,evidence,required).get(geo_mode,[])[:pool]
            return dict(status='ok' if rows else 'no_result', hits=rows, backend='geometry_117',
                        reason=None if rows else 'no_original_within_geometry_cutoff',
                        elapsed_ms=(perf_counter()-started)*1000)
        engine.geometry_search=MethodType(geometry,engine)
    # Keep the text cap 40 even when the geometry cap grows.
    def semantic(self,description,required=None):
        started=perf_counter()
        adapter=copy(self.semantic)
        settings=replace(adapter.pool_service.settings,min_score_vlm=.26 if text_mode in ('threshold','both') else .35)
        adapter.pool_service=SemanticSearchService(adapter.pool_service.index,settings)
        validator=SoftSemanticValidator(self.validator) if joint and text_mode in ('soft','both') else self.validator
        adapter.whole_pose_validator=validator
        result=adapter.search(SemanticQuery.from_description(description),top_k=40)
        hits=self._clean([h.to_dict() for h in result.hits],'semantic')
        if required and required['active']:
            accepted=[]
            for hit in hits:
                checked=validator.check(required,hit['pose_id'])
                if checked['status'] in ('passed','near_match'):
                    accepted.append(dict(hit,whole_pose_validation=checked))
            hits=accepted
        return dict(status='error' if result.status=='error' else 'ok' if hits else 'no_result',
                    hits=hits,reason=result.reason,backend=result.backend,query_evidence=result.query_evidence,
                    elapsed_ms=(perf_counter()-started)*1000)
    engine.semantic_search=MethodType(semantic,engine)
    return engine
