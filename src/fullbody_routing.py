"""Whole-library pose routing. No region retrieval, donor selection or composition.

Distances and semantic scores retain their own units. Reciprocal rank fusion
combines ranked original-pose lists; the resulting score is not confidence.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .camera_search import CameraIndex
from .config import CFG
from .semantic_adapter_285 import excluded_pose_ids
from .search import pose_family_id
from .search_routing import PoseDescription
from .semantic_search import SemanticQuery

VERSION = 'fullbody-dynamic-v1'


class FullBodySearch:
    def __init__(self, entries, semantic, camera_index, *, top_k=5, pool_k=40,
                 geometry_max_distance=1.0, geometry_upper_max_distance=1.5,
                 generator=None, startup_error=None):
        if not 1 <= top_k <= pool_k <= 200:
            raise ValueError('invalid search limits')
        if any(not math.isfinite(v) or v <= 0 for v in (geometry_max_distance, geometry_upper_max_distance)):
            raise ValueError('geometry cutoffs must be finite and positive')
        self.meta = {e.pose_id: e.meta for e in entries}
        self.allowed = set(self.meta)
        self.semantic, self.camera = semantic, camera_index
        self.top_k, self.pool_k = top_k, pool_k
        self.cutoffs = {'full': geometry_max_distance, 'upper': geometry_upper_max_distance}
        self.generator, self.startup_error = generator, startup_error

    def _clean(self, hits, branch):
        quarantined = excluded_pose_ids()
        out, seen = [], set()
        for hit in hits:
            pid = hit.get('pose_id')
            if pid not in self.allowed:
                raise ValueError('candidate_not_in_bound_library')
            # Do not let old precomposed assets leak into this original-only path.
            if pid in quarantined or self.meta[pid].get('source') == 'generated_composition':
                continue
            if pid in seen:
                continue
            seen.add(pid)
            value = hit.get('distance' if branch == 'geometry' else 'score')
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('invalid_candidate_score')
            out.append(hit)
        return out

    def geometry_search(self, evidence):
        started = perf_counter()
        try:
            if self.camera is None:
                raise ValueError('camera_index_unavailable')
            if len(self.camera.cameras) != 117:
                raise ValueError('camera_count_must_be_117')
            points = np.asarray(evidence['keypoints'], float)
            mask = np.asarray(evidence['joint_mask'], bool)
            if points.shape != (17, 2) or mask.shape != (17,) or not np.isfinite(points[mask]).all():
                raise ValueError('invalid_observed_joints')
            part = 'upper' if evidence.get('coverage_class') == 'upper_only' else 'full'
            anchors = [5, 6] if part == 'upper' else [5, 6, 11, 12]
            if not mask[anchors].all():
                raise ValueError('sufficient_gate_missing_anchors')
            hits = self._clean(self.camera.search(points, mask, part), 'geometry')
            hits = [h for h in hits if 0 <= h['distance'] <= self.cutoffs[part]][:self.pool_k]
            return dict(status='ok' if hits else 'no_result', hits=hits, backend='geometry_117',
                reason=None if hits else 'no_original_within_geometry_cutoff', camera_count=117,
                observation_scope=part, retrieval_scope='whole_library_pose', cutoff=self.cutoffs[part],
                cutoff_calibration='experimental_not_validated_on_holdout', elapsed_ms=(perf_counter()-started)*1000)
        except Exception as error:
            return dict(status='error', hits=[], backend='geometry_117',
                        reason='geometry_search:'+type(error).__name__, elapsed_ms=(perf_counter()-started)*1000)

    def semantic_search(self, description):
        started = perf_counter()
        try:
            result = self.semantic.search(SemanticQuery.from_description(description), top_k=self.pool_k)
            if result.status not in ('ok', 'no_result', 'error'):
                raise ValueError('invalid_search_status')
            if (result.status == 'ok') != bool(result.hits):
                raise ValueError('inconsistent_search_status')
            hits = self._clean([h.to_dict() for h in result.hits], 'semantic')
            return dict(status=('ok' if hits else 'no_result') if result.status != 'error' else 'error',
                        hits=hits, reason=result.reason, backend=result.backend,
                        elapsed_ms=(perf_counter()-started)*1000)
        except Exception as error:
            return dict(status='error', hits=[], backend='semantic',
                        reason='semantic_search:'+type(error).__name__, elapsed_ms=(perf_counter()-started)*1000)

    def _fuse(self, branches):
        rows = {}
        active = {k:v for k,v in branches.items() if v['status'] in ('ok', 'no_result')}
        for name, branch in active.items():
            weight = (.65 if name == 'geometry' else .35) if len(active) == 2 else 1.
            for rank, hit in enumerate(branch['hits'], 1):
                pid = hit['pose_id']
                row = rows.setdefault(pid, dict(pose_id=pid, kind='library_original',
                    rank_score=0., evidence={}, camera=None, refine_allowed=False,
                    lookup_allowed=True, release_allowed=False, actual_mesh_validated=False))
                row['rank_score'] += weight / (60+rank)
                row['evidence'][name] = dict(rank=rank, **hit)
                if name == 'geometry':
                    row['camera'] = hit.get('camera')
                elif row['camera'] is None:
                    row['semantic_preview_camera'] = hit.get('pose', {}).get('preview_camera')
        result, seen = [], set()
        for row in sorted(rows.values(), key=lambda r:(-r['rank_score'], r['pose_id'])):
            family = pose_family_id(row['pose_id'], self.meta[row['pose_id']])
            if family in seen:
                continue
            seen.add(family)
            row['family'] = family
            result.append(row)
            if len(result) == self.top_k:
                break
        return result

    def search(self, decision, evidence, *, generation_block=False):
        start = perf_counter()
        out = dict(version=VERSION, route='undetermined', status='not_run', selected=[],
            branches={}, generation=None, composition_enabled=False, split_search_enabled=False,
            ranking='weighted_rrf_k60_geometry_0.65_semantic_0.35', rank_score_is_accuracy=False,
            geometry_evidence=decision.joint.to_dict(), description_evidence=decision.semantic.to_dict())
        def finish(status, reason=None):
            out.update(status=status, reason=reason, elapsed_ms=(perf_counter()-start)*1000)
            return out
        if decision.joint.policy_block:
            return finish('blocked', decision.joint.policy_block)
        if self.startup_error:
            out['route'] = 'error'
            return finish('error', self.startup_error)
        sufficient = decision.joint.sufficient
        semantic_valid = decision.semantic.valid
        description = PoseDescription(**decision.semantic.description) if semantic_valid else None
        out['route'] = 'hybrid' if sufficient and semantic_valid else 'geometric' if sufficient else 'semantic' if semantic_valid else 'undetermined'
        if not sufficient and not semantic_valid:
            if decision.semantic.status in ('error', 'vlm_error'):
                out['route'] = 'error'
                return finish('error', 'description_failed')
            return finish('undetermined', 'no_trusted_joint_or_text_evidence')
        # Two WHOLE-POSE searches only. No upper/lower retrieval or BVH assembly.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='fullbody-search') as pool:
            futures = {}
            if sufficient:
                futures['geometry'] = pool.submit(self.geometry_search, evidence)
            if semantic_valid:
                futures['semantic'] = pool.submit(self.semantic_search, description)
            out['branches'] = {k:v.result() for k,v in futures.items()}
        if not semantic_valid:
            failed = decision.semantic.status in ('error', 'vlm_error')
            out['branches']['semantic'] = dict(status='error' if failed else 'not_run',
                hits=[], reason='description_failed' if failed else 'description_not_specific')
        out['selected'] = self._fuse(out['branches'])
        errors = any(b['status'] == 'error' for b in out['branches'].values())
        if errors:
            return finish('partial_error' if out['selected'] else 'error', 'search_or_description_error_no_generation')
        if out['selected']:
            return finish('ok')
        if not all(b['status'] == 'no_result' for b in out['branches'].values()):
            return finish('undetermined', 'not_all_required_searches_completed')
        if generation_block:
            out['generation'] = dict(status='not_requested', reason='cropped_or_unsupported_scope')
        elif self.generator is None:
            out['generation'] = dict(status='not_requested', reason='generation_disabled')
        else:
            from .experimental.text_pose_generation import after_no_result
            out['generation'] = after_no_result(self.generator, description,
                search_status='no_result', hit_count=0).to_dict()
        return finish('no_result', 'successful_empty_search_generation_separate')


def build_api_pipeline(entries, *, db_path):
    """This branch defaults to whole-pose routing; no model/license gate bypass."""
    from .pipeline import Pipeline
    if os.getenv('ROUTING_FULLBODY_ENABLED', '1') != '1':
        return Pipeline(entries)
    if CFG.pose_backend != 'rtmlib' or CFG.pose_model_variant != 'humanart-m':
        raise ValueError('fullbody routing requires POSE_BACKEND=rtmlib and POSE_MODEL_VARIANT=humanart-m')
    from .pose_humanart import HumanArtPoseModel
    pose = HumanArtPoseModel()  # strict manifest/calibration validation, no mock fallback
    from .vlm.client import build_vlm_client, MockVLMClient
    vlm = build_vlm_client()
    if isinstance(vlm, MockVLMClient):
        raise ValueError('fullbody routing requires a configured real VLM')
    root = Path(CFG.data_dir)
    from .semantic_adapter_285 import build_semantic285_searcher
    semantic = build_semantic285_searcher(
        tags_path=os.getenv('SEMANTIC_285_TAGS_PATH', str(root/'search-v2.json')),
        taxonomy_path=os.getenv('SEMANTIC_285_TAXONOMY_PATH', str(root/'search-taxonomy-v2.json')),
        db_path=db_path, library_pose_ids={e.pose_id for e in entries},
        body_index_path=os.getenv('SEMANTIC_285_BODY_INDEX_PATH', str(root/'body.json')),
        shape_index_path=os.getenv('SEMANTIC_285_SHAPE_INDEX_PATH', str(root/'shapes.json')),
        dense_model_dir=os.getenv('SEMANTIC_EMBEDDING_DIR') or None)
    if getattr(semantic, 'service', None) is None:
        raise ValueError('fullbody semantic index unavailable: '+getattr(semantic, 'reason', 'unknown'))
    allowed = {e.pose_id for e in entries if e.meta.get('source') != 'generated_composition'} - excluded_pose_ids()
    camera = CameraIndex.from_library(allowed, os.getenv('ROUTING_BVH_ROOT', str(root/'bvh')), count=117)
    if camera.errors:
        raise ValueError('fullbody camera index incomplete; invalid BVH assets require curation')
    generator = None
    if os.getenv('POSE_GENERATION_ENABLED', '1') == '1':
        from .natural_pose import NaturalLanguagePoseGenerator
        generator = NaturalLanguagePoseGenerator(
            os.getenv('POSE_GENERATION_RIG', str(root/'bvh/webtoon_combat_03.bvh')),
            os.getenv('POSE_GENERATION_TAXONOMY', str(root/'search-taxonomy-v2.json')),
            model=os.getenv('POSE_GENERATION_MODEL', 'gemini-3.8-flash'),
            review_dir=os.getenv('POSE_GENERATION_REVIEW_DIR', 'tmp/generated-pose-review'))
    engine = FullBodySearch(entries, semantic, camera, generator=generator,
        geometry_max_distance=float(os.getenv('ROUTING_GEOMETRY_MAX_DISTANCE', '1.0')),
        geometry_upper_max_distance=float(os.getenv('ROUTING_GEOMETRY_UPPER_MAX_DISTANCE', '1.5')))
    return Pipeline(entries, vlm_client=vlm, pose_model=pose, semantic_searcher=semantic, fullbody_search=engine)
