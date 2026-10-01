"""Optional image-slot semantic output for /analyze; geometric output preserved."""
from __future__ import annotations
from pathlib import Path
from threading import RLock
_INIT_LOCK=RLock()
from urllib.parse import quote
from .rough_semantic_v2 import build_rough_semantic_runtime,execute_structured_plan
from .rough_router_shadow import plans_from_result
from ..vlm.rough_slots import parse_slots


class RoughSemanticService:
    def __init__(self,*,facts_path,geometry_db,build_dir,captions_path=""):
        self.options=dict(facts_path=Path(facts_path),geometry_db=Path(geometry_db),build_dir=Path(build_dir))
        self.captions_path=captions_path
        self.runtime=None
        self.lock=RLock()

    def analyze(self,vlm,result):
        with self.lock:
            if self.runtime is None:
                if not (self.options['build_dir']/'manifest.json').is_file():
                    raise ValueError('rough semantic index must be built offline first')
                if self.captions_path:
                    from .rough_multiview import MultiviewRoughSemanticSearch
                    self.runtime=MultiviewRoughSemanticSearch(captions_path=self.captions_path,**self.options)
                else:
                    self.runtime=build_rough_semantic_runtime(**self.options)
            parsed=parse_slots(vlm.raw,vlm.num_people)
            if not parsed.people:
                return dict(status='insufficient_semantics',people=[],slot_issues=list(parsed.issues),refine_allowed=False,geometry_fused=False)
            people=[]
            for plan in plans_from_result(vlm,result):
                index=plan.get('person_index');person=parsed.get(index)
                if person is None:continue
                execution=execute_structured_plan(plan,self.runtime,slots=person)
                for q in execution['semantic_requests']:
                    for c in q['results']:
                        c.pop('bvh_path',None)
                        c['bvh_url']='/pose/'+quote(c['pose_id'],safe='')+'/bvh'
                people.append(dict(person_index=index,route=plan['route'],**{
                    'queries':execution['semantic_requests']}))
            return dict(status='ok',implementation=self.runtime.version,
                semantic_build_id=self.runtime.build_id,people=people,
                geometry_fused=False,composition_executed=False,refine_allowed=False,
                exact_match_verified=False,slot_issues=list(parsed.issues))


def attach_semantic_candidates(pipe,vlm,result,cfg):
    """No model imports/initialization at all when disabled (caller guards import)."""
    try:
        with _INIT_LOCK:
            if getattr(pipe,'_rough_semantic_service',None) is None:
                pipe._rough_semantic_service=RoughSemanticService(facts_path=cfg.rough_semantic_facts_path,
                    geometry_db=cfg.rough_semantic_geometry_db,build_dir=cfg.rough_semantic_build_dir,
                    captions_path=getattr(cfg,"rough_semantic_captions_path",""))
        result.rough_semantic=pipe._rough_semantic_service.analyze(vlm,result)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('rough_semantic_failed type=%s',type(exc).__name__)
        result.rough_semantic=dict(status='unavailable',error_type=type(exc).__name__,
            reason='semantic_index_or_execution_failed',people=[],geometry_fused=False,refine_allowed=False)
    return result
