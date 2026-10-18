"""Whole-library pose routing. No region retrieval, donor selection or composition.

Distances and semantic scores retain their own units. Reciprocal rank fusion
combines ranked original-pose lists; the resulting score is not confidence.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
API_SEMANTIC_WEIGHT = .6


class FullBodySearch:
    def __init__(self, entries, semantic, camera_index, *, top_k=5, pool_k=40,
                 geometry_max_distance=1.0, geometry_upper_max_distance=1.5,
                 generator=None, startup_error=None, include_stored_compositions=True,
                 semantic_weight=.35, hand_skin_gate=None, hand_spacing_repairer=None):
        if not 1 <= top_k <= pool_k <= 200:
            raise ValueError('invalid search limits')
        if any(not math.isfinite(v) or v <= 0 for v in (geometry_max_distance, geometry_upper_max_distance)):
            raise ValueError('geometry cutoffs must be finite and positive')
        if not math.isfinite(semantic_weight) or not 0 < semantic_weight < 1:
            raise ValueError('semantic_weight must be finite and between 0 and 1')
        self.semantic_weight = semantic_weight
        self.geometry_weight = 1. - semantic_weight
        self.ranking = f'weighted_rrf_k60_geometry_{self.geometry_weight:g}_semantic_{self.semantic_weight:g}'
        self.meta = {e.pose_id: e.meta for e in entries}
        self.allowed = set(self.meta)
        self.semantic, self.camera = semantic, camera_index
        self.top_k, self.pool_k = top_k, pool_k
        self.cutoffs = {'full': geometry_max_distance, 'upper': geometry_upper_max_distance}
        self.generator, self.startup_error = generator, startup_error
        self.hand_skin_gate = hand_skin_gate
        self.hand_spacing_repairer = hand_spacing_repairer
        self.include_stored_compositions = include_stored_compositions
        self.validator=None
        if hasattr(camera_index,'pose_ids') and hasattr(semantic,'core_records'):
            from .pose_requirements import WholePoseValidator
            self.validator=WholePoseValidator(camera_index,semantic.core_records,getattr(semantic,'body',{}))
            semantic.whole_pose_validator=self.validator

    def _clean(self, hits, branch):
        quarantined = excluded_pose_ids()
        out, seen = [], set()
        for hit in hits:
            pid = hit.get('pose_id')
            if pid not in self.allowed:
                raise ValueError('candidate_not_in_bound_library')
            # Stored compositions are complete library poses, not runtime assembly.
            # Exclude by quality policy, not provenance. Opt-out is for paired replay.
            if pid in quarantined or (not self.include_stored_compositions
                    and self.meta[pid].get('source') == 'generated_composition'):
                continue
            if pid in seen:
                continue
            seen.add(pid)
            value = hit.get('distance' if branch == 'geometry' else 'score')
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('invalid_candidate_score')
            out.append(hit)
        return out

    def geometry_search(self, evidence, required=None):
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
            hits = [h for h in hits if 0 <= h['distance'] <= self.cutoffs[part]]
            validation_counts={};examples=[]
            if self.validator and required and required['active']:
                accepted=[]
                for hit in hits:
                    checked=self.validator.check(required,hit['pose_id'])
                    validation_counts[checked['status']]=validation_counts.get(checked['status'],0)+1
                    if checked['status'] in ('passed','near_match'):
                        hit['whole_pose_validation']=checked;accepted.append(hit)
                    elif len(examples)<20:examples.append(dict(pose_id=hit['pose_id'],**checked))
                hits=accepted
            hits=hits[:self.pool_k]
            return dict(status='ok' if hits else 'no_result', hits=hits, backend='geometry_117',
                reason=None if hits else 'no_original_within_geometry_cutoff', camera_count=117,
                observation_scope=part, retrieval_scope='whole_library_pose', cutoff=self.cutoffs[part],
                cutoff_calibration='experimental_not_validated_on_holdout',
                whole_pose_validation=dict(counts=validation_counts,excluded_examples=examples),
                elapsed_ms=(perf_counter()-started)*1000)
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
                        query_evidence=getattr(result,'query_evidence',None),
                        elapsed_ms=(perf_counter()-started)*1000)
        except Exception as error:
            return dict(status='error', hits=[], backend='semantic',
                        reason='semantic_search:'+type(error).__name__, elapsed_ms=(perf_counter()-started)*1000)

    def _fuse(self, branches, core_evidence=None, allow_mirror_pair=False, *, detail_evidence=None, limit=None):
        rows = {}
        active = {k:v for k,v in branches.items() if v['status'] in ('ok', 'no_result')}
        for name, branch in active.items():
            weight = (self.geometry_weight if name == 'geometry' else self.semantic_weight) if len(active) == 2 else 1.
            for rank, hit in enumerate(branch['hits'], 1):
                pid = hit['pose_id']
                row = rows.setdefault(pid, dict(pose_id=pid, kind='library_original',
                    library_source=self.meta[pid].get('source'),
                    rank_score=0., evidence={}, camera=None, refine_allowed=False,
                    lookup_allowed=True, release_allowed=False, actual_mesh_validated=False))
                row['rank_score'] += weight / (60+rank)
                row['evidence'][name] = dict(rank=rank, **hit)
                if hit.get('matched',{}).get('visible_region_reference'):
                    row['kind']='framing_reference'
                if name == 'geometry':
                    row['camera'] = hit.get('camera')
                elif row['camera'] is None:
                    row['semantic_preview_camera'] = hit.get('pose', {}).get('preview_camera')
        result, seen = [], set()
        paired=set()
        for row in rows.values():
            if core_evidence and row['pose_id'] in core_evidence:
                row['core_conditions']=core_evidence[row['pose_id']]
            if detail_evidence and row['pose_id'] in detail_evidence:
                row['pose_detail_fit']=detail_evidence[row['pose_id']]
            validation=next((hit.get('whole_pose_validation') or hit.get('matched',{}).get('whole_pose_validation')
                for hit in row['evidence'].values() if hit.get('whole_pose_validation') or hit.get('matched',{}).get('whole_pose_validation')),None)
            if validation:
                row['match_level']='near_match' if validation['status']=='near_match' else 'core_match'
                row.setdefault('core_conditions',{})['whole_pose_validation']=validation
        for row in sorted(rows.values(), key=lambda r:(
                r.get('match_level')=='near_match',
                (r.get('core_conditions') or {}).get('priority',[0,0]),
                math.floor(r['pose_detail_fit']['cost']/.5) if r.get('pose_detail_fit',{}).get('cost') is not None else 0,
                -r['rank_score'],r['pose_id'])):
            family = pose_family_id(row['pose_id'], self.meta[row['pose_id']])
            if family in seen:
                first=next(r for r in result if r['family']==family)
                if (not allow_mirror_pair or paired or
                    bool(self.meta[row['pose_id']].get('is_mirror'))==bool(self.meta[first['pose_id']].get('is_mirror'))):
                    continue
                paired.add(family)
                row['side_alternative']=True
            seen.add(family)
            row['family'] = family
            result.append(row)
            if len(result) == (limit or self.top_k):
                break
        return result

    def _pose_detail_evidence(self,evidence,ids):
        """Use the same observed joints and fixed cameras for detail comparison.

        No pose rotations, hidden-joint guesses or BVH edits are performed.
        """
        if not self.validator:return {}
        from .camera_search import frame
        from .pose_requirements import compare_pose_details
        points=np.asarray(evidence.get('keypoints'),float);mask=np.asarray(evidence.get('joint_mask'),bool)
        if points.shape!=(17,2) or mask.shape!=(17,) or not mask[[5,6,11,12]].all():return {}
        selected=np.arange(5,17)[mask[5:17]]
        if len(selected)<6:return {}
        query,ok=frame(points,'full',mask)
        if not ok:return {}
        frames,valid=self.camera.frames['full'];out={}
        for pid in ids:
            i=self.validator.positions[pid]
            distances=np.linalg.norm(frames[:,i,selected]-query[selected],axis=-1).mean(axis=-1)
            distances=np.where(self.camera.valid[:,i]&valid[:,i],distances,np.inf)
            best=int(np.argmin(distances))
            if not np.isfinite(distances[best]):continue
            fit=compare_pose_details(points,self.camera.projections[best,i],mask)
            fit.update(camera_index=best,geometry_distance=float(distances[best]))
            out[pid]=fit
        return out

    def search(self, decision, evidence, *, generation_block=False, review_geometry=False):
        start = perf_counter()
        out = dict(version=VERSION, route='undetermined', status='not_run', selected=[],near_candidates=[],
            branches={}, generation=None, composition_enabled=False, split_search_enabled=False,
            stored_compositions_included=self.include_stored_compositions,
            ranking=self.ranking, rank_score_is_accuracy=False,
            geometry_evidence=decision.joint.to_dict(), description_evidence=decision.semantic.to_dict())
        def finish(status, reason=None):
            out.update(status=status, reason=reason, elapsed_ms=(perf_counter()-start)*1000)
            return out
        if decision.joint.policy_block:
            return finish('blocked', decision.joint.policy_block)
        if self.startup_error:
            out['route'] = 'error'
            return finish('error', self.startup_error)
        from .basic_closeup import basic_closeup_plan
        basic=basic_closeup_plan(evidence,decision.semantic.description)
        if basic is not None:
            out.update(route='basic_model',basic_model=basic,
                branches={name:dict(status='not_run',hits=[],reason='explicit_closeup_basic_model')
                          for name in ('geometry','semantic')},
                generation=dict(status='not_requested',reason='basic_model_not_pose_generation'))
            return finish('ok','explicit_closeup_basic_model')
        sufficient = decision.joint.sufficient
        semantic_valid = decision.semantic.valid
        description = PoseDescription(**decision.semantic.description) if semantic_valid else None
        if description is not None and not description.visible_scope:
            description.visible_scope=evidence.get('output_scope')
        required=None
        if description is not None and self.validator:
            from .pose_query import prepare_pose_query
            from .pose_requirements import requirements
            q=SemanticQuery.from_description(description)
            required=requirements(q,prepare_pose_query(q,self.semantic.service.index.lexicon))
            out['candidate_validation']=dict(version='whole-pose-conditions-v2',requirements=required,
                unknown_policy='separate_review_not_selected',split_search=False)
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
                futures['geometry'] = pool.submit(self.geometry_search, evidence, required)
            if semantic_valid:
                futures['semantic'] = pool.submit(self.semantic_search, description)
            out['branches'] = {k:v.result() for k,v in futures.items()}
        if not semantic_valid:
            failed = decision.semantic.status in ('error', 'vlm_error')
            out['branches']['semantic'] = dict(status='error' if failed else 'not_run',
                hits=[], reason='description_failed' if failed else 'description_not_specific')
        core={}
        if description is not None and hasattr(self.semantic,'core_evidence'):
            ids={h['pose_id'] for b in out['branches'].values() for h in b.get('hits',[])}
            core=self.semantic.core_evidence(SemanticQuery.from_description(description),ids)
        if core:
            out['condition_ranking']='held_object_and_relations_then_weighted_rrf'
        ambiguous=bool(description and (any(s=='unknown' for s in description.held_object_hands.values())
            or any(r.get('side')=='unknown' for r in description.pose_relations)))
        ids={h['pose_id'] for b in out['branches'].values() for h in b.get('hits',[])}
        fits=self._pose_detail_evidence(evidence,ids) if sufficient else {}
        ranked=self._fuse(out['branches'],core,allow_mirror_pair=ambiguous,detail_evidence=fits,limit=self.pool_k*2)
        out['selected']=[r for r in ranked if r.get('match_level')!='near_match'][:self.top_k]
        out['near_candidates']=[dict(r,review_required=True,not_exact_match=True) for r in ranked
                                if r.get('match_level')=='near_match'][:self.top_k]
        if fits:out['condition_ranking']='hard_core_then_detail_bands_then_existing_rrf'
        # Framing references are not successful pose matches and must not
        # suppress the empty-search generation path.
        out['framing_references']=[r for r in out['selected'] if r['kind']=='framing_reference']
        out['selected']=[r for r in out['selected'] if r['kind']!='framing_reference']
        skin_session = None
        if self.hand_skin_gate is not None:
            skin_session = self.hand_skin_gate.session(evidence.get('hand_presets'))
            eligible = [r for r in ranked if r.get('match_level') != 'near_match'
                        and r['kind'] != 'framing_reference']
            out['selected'] = skin_session.filter(eligible, limit=self.top_k)
            out['hand_skin_screen'] = deepcopy(skin_session.report())
        if self.validator and required and required['active']:
            for hit in out['selected']:
                checked=self.validator.check(required,hit['pose_id'])
                hit.setdefault('core_conditions',{})['whole_pose_validation']=checked
        for branch in out['branches'].values():
            refs=[h for h in branch.get('hits',[]) if h.get('matched',{}).get('visible_region_reference')]
            if refs:
                branch['framing_references']=refs
                branch['hits']=[h for h in branch['hits'] if h not in refs]
                if not branch['hits'] and branch['status']=='ok':
                    branch.update(status='no_result',reason='framing_reference_not_pose_match')
        errors = any(b['status'] == 'error' for b in out['branches'].values())
        if errors:
            return finish('partial_error' if out['selected'] else 'error', 'search_or_description_error_no_generation')
        if skin_session and skin_session.unverified:
            return finish('partial_error' if out['selected'] else 'error', 'hand_skin_screen_unavailable_no_generation')
        if out['selected']:
            from .mirror_reference import mirror_reference
            try:
                reference, comparison = mirror_reference(self, out['selected'], description, required, skin_session)
            except (ValueError, KeyError, TypeError, OSError, AttributeError):
                reference, comparison = None, dict(status='unavailable', reason='mirror_validation_failed')
            out['mirror_comparison'] = comparison
            if reference:
                # The reference occupies one existing display slot, never a sixth.
                budget = min(self.top_k, 5)
                out['selected'] = [r for r in out['selected'] if r['pose_id'] != reference['pose_id']][:budget - 1]
                out['mirror_references'] = [reference]
                out['near_candidates'] = out['near_candidates'][:max(0, budget - len(out['selected']) - 1)]
                out['framing_references'] = []
            if skin_session:
                # Optional, rejected references cannot invalidate already screened matches.
                out['mirror_hand_skin_screen'] = skin_session.report()
            return finish('ok')
        # A branch may have only near candidates. These are review evidence,
        # not qualified matches, and must not suppress empty-search generation.
        if not all(b['status'] in ('ok','no_result') for b in out['branches'].values()):
            return finish('undetermined', 'not_all_required_searches_completed')
        if generation_block:
            out['generation'] = dict(status='not_requested', reason='cropped_or_unsupported_scope')
        elif self.generator is None:
            out['generation'] = dict(status='not_requested', reason='generation_disabled')
        else:
            from .experimental.text_pose_generation import after_no_result
            out['generation'] = after_no_result(self.generator, description,
                search_status='no_result', hit_count=0).to_dict()
            out['generation']['diagnostics']['output_scope']=description.visible_scope
            out['generation']['diagnostics']['unobserved_body_is_default']=description.visible_scope in ('head','shoulder_head','half','bust')
            if required and out['generation']['status']=='preview_ready':
                from .pose_requirements import audit_generated
                try:
                    audits=[audit_generated(required,c) for c in out['generation']['candidates']]
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    audits=[dict(status='error',reason='generated_condition_audit:'+type(exc).__name__)]
                out['generation']['diagnostics']['whole_pose_conditions']=audits
                if any(a['status'] not in ('passed','not_required') for a in audits):
                    out['generation'].update(status='rejected',reason='generated_observation_conditions_failed',candidates=[])
            if skin_session and out['generation']['status'] == 'preview_ready':
                generated = out['generation']['candidates']
                if self.hand_spacing_repairer:
                    passed, repair_report = self.hand_spacing_repairer.filter(
                        generated, out['generation'], required, skin_session)
                    out['generation']['diagnostics']['hand_spacing_repair'] = repair_report
                else:
                    passed = skin_session.filter(generated, generated=True, limit=len(generated))
                out['generation']['candidates'] = passed
                out['hand_skin_screen'] = skin_session.report()
                if not passed:
                    out['generation'].update(status='rejected', reason='generated_hand_skin_screen_not_clear')
                if skin_session.unverified:
                    return finish('error', 'generated_hand_skin_screen_unavailable')
        # Keep qualified selection and generation unchanged. When optional
        # integrated recovery is enabled, expose bounded geometry references as
        # review-only choices rather than turning text overconstraints into an
        # empty pose gallery. Never relabel them as qualified core matches.
        if review_geometry and sufficient:
            retained = dict(active=bool(required and required.get('posture')),
                            posture=required.get('posture') if required else None,
                            arms={}, body_states={}, hand_counts={})
            fallback = self.geometry_search(evidence, retained)
            for hit in fallback.get('hits', []):
                if hit.get('whole_pose_validation'):
                    hit['whole_pose_validation'] = {**hit['whole_pose_validation'], 'condition_scope': 'retained_posture_only'}
            out['geometry_review_fallback'] = dict(
                status=fallback['status'], reason=fallback.get('reason'),
                qualified_count=0, constraints_satisfied=False,
                original_requirements=required, retained_hard_conditions=retained, cutoff=fallback.get('cutoff'))
            if fallback['status'] == 'ok':
                rows = self._fuse({'geometry': fallback}, limit=self.pool_k)
                for row in rows:
                    row.update(match_level='geometry_fallback', review_required=True,
                               not_exact_match=True, constraints_satisfied=False)
                    if self.validator and required and required['active']:
                        row['core_conditions'] = dict(whole_pose_validation=
                            self.validator.check(required, row['pose_id']))
                if skin_session:
                    rows = skin_session.filter(rows, limit=self.top_k)
                    out['hand_skin_screen'] = skin_session.report()
                    if skin_session.unverified:
                        out['selected'] = rows
                        return finish('partial_error' if rows else 'error',
                                      'geometry_review_hand_skin_screen_unavailable')
                out['selected'] = rows[:self.top_k]
                if rows:
                    return finish('ok', 'geometry_review_fallback_not_qualified')
        return finish('no_result', 'successful_empty_search_generation_separate')


def build_api_pipeline(entries, *, db_path):
    """This branch defaults to whole-pose routing; no model/license gate bypass."""
    from .pipeline import Pipeline
    if os.getenv('ROUTING_FULLBODY_ENABLED', '1') != '1':
        return Pipeline(entries)
    pose, semantic, engine = _build_fullbody_resources(entries, db_path=db_path)
    from .vlm.client import build_vlm_client, MockVLMClient
    vlm = build_vlm_client()
    if isinstance(vlm, MockVLMClient):
        raise ValueError('fullbody routing requires a configured real VLM')
    return Pipeline(entries, vlm_client=vlm, pose_model=pose, semantic_searcher=semantic, fullbody_search=engine)


def _build_fullbody_resources(entries, *, db_path):
    """Shared by real startup and deployment file/model preflight (no VLM calls)."""
    if CFG.pose_backend != 'rtmlib' or CFG.pose_model_variant != 'humanart-m':
        raise ValueError('fullbody routing requires POSE_BACKEND=rtmlib and POSE_MODEL_VARIANT=humanart-m')
    from .pose_humanart import HumanArtPoseModel
    bundle = None
    if os.getenv('POSE_CALIBRATION_FROM_MANIFEST', '0') == '1':
        from .pose_contract import load_pose_bundle, apply_bundle_calibration
        bundle = load_pose_bundle(CFG.pose_model_manifest, expected_model_id='humanart-m')
        apply_bundle_calibration(bundle, CFG)
    pose = HumanArtPoseModel(bundle=bundle)  # strict validation remains, no mock fallback
    root = Path(CFG.data_dir)
    include_stored = os.getenv('ROUTING_INCLUDE_STORED_COMPOSITIONS', '1') == '1'
    allowed = {e.pose_id for e in entries if include_stored
               or e.meta.get('source') != 'generated_composition'} - excluded_pose_ids()
    paths = dict(tags=Path(os.getenv('SEMANTIC_285_TAGS_PATH', str(root/'search-v2.json'))),
                 taxonomy=Path(os.getenv('SEMANTIC_285_TAXONOMY_PATH', str(root/'search-taxonomy-v2.json'))),
                 body=Path(os.getenv('SEMANTIC_285_BODY_INDEX_PATH', str(root/'body.json'))),
                 shapes=Path(os.getenv('SEMANTIC_285_SHAPE_INDEX_PATH', str(root/'shapes.json'))),
                 bvh=Path(os.getenv('ROUTING_BVH_ROOT', str(root/'bvh'))))
    if os.getenv('ROUTING_PREPARE_ASSETS', '0') == '1':
        from .fullbody_assets import prepare_assets, validate_generation_assets
        paths = prepare_assets(db_path=db_path, root=root, allowed=allowed)
        validate_generation_assets(root)
        from .logging_setup import log_info
        log_info('fullbody_assets', **paths['receipt'])
    from .semantic_adapter_285 import build_semantic285_searcher
    semantic = build_semantic285_searcher(
        tags_path=paths['tags'],
        taxonomy_path=paths['taxonomy'],
        db_path=db_path, library_pose_ids={e.pose_id for e in entries},
        body_index_path=paths['body'],
        shape_index_path=paths['shapes'],
        dense_model_dir=os.getenv('SEMANTIC_EMBEDDING_DIR') or None)
    if getattr(semantic, 'service', None) is None:
        raise ValueError('fullbody semantic index unavailable: '+getattr(semantic, 'reason', 'unknown'))
    camera = CameraIndex.from_library(allowed, paths['bvh'], count=117)
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
        semantic_weight=API_SEMANTIC_WEIGHT,
        include_stored_compositions=include_stored,
        geometry_max_distance=float(os.getenv('ROUTING_GEOMETRY_MAX_DISTANCE', '1.0')),
        geometry_upper_max_distance=float(os.getenv('ROUTING_GEOMETRY_UPPER_MAX_DISTANCE', '1.5')))
    from .hand_skin_screen import build_hand_skin_gate
    engine.hand_skin_gate = build_hand_skin_gate(camera, paths['bvh'])
    if generator is not None:
        from .hand_spacing_repair import GeneratedHandSpacingRepair
        engine.hand_spacing_repairer = GeneratedHandSpacingRepair(generator.repair_hand_spacing)
    return pose, semantic, engine
