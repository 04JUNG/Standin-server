"""Recover the combined geometry and captured semantic candidates."""

from copy import copy
from copy import deepcopy
from time import perf_counter
from types import MethodType
from .analyze import geometry

def prepare_engine(original_engine, previous, cap=200):
    engine=copy(original_engine)
    engine.pool_k=cap
    engine.geometry_weight=.7
    engine.semantic_weight=.3
    engine.ranking='weighted_rrf_k60_geometry_0.7_semantic_0.3'
    captured=previous['branches'].get('semantic',{})

    def search_geometry(self,evidence,required=None):
        started=perf_counter()
        rows=geometry(self,evidence,required,bone=None)[0][f'position_{cap}']
        return dict(status='ok' if rows else 'no_result',hits=rows,
            backend='geometry_117',reason=None if rows else 'no_original_within_geometry_cutoff',
            elapsed_ms=(perf_counter()-started)*1000)

    def search_semantic(self,description,required=None):
        result=deepcopy(captured)
        references=result.pop('framing_references',[])
        if references:
            result['hits']+=references
            result['status']='ok'
            result['reason']=None
        return result

    engine.geometry_search=MethodType(search_geometry,engine)
    engine.semantic_search=MethodType(search_semantic,engine)
    return engine
