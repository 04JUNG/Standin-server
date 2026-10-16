"""Natural language -> constrained spec -> IK -> audited, private review BVH.

The entire Gemini/IK/retry job runs in a killable child process. Never inserts
generated poses into the live library, CandidateOut, refine, or export-order.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .experimental.text_pose_generation import GenerationResult

VERSION = 'natural-language-spec-ik-audit-v1'
RIG_SHA256 = '38eb05b7432893fefd4cf31cb6f5fb07e21009946541d5a9d48d19c949991ee1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_loop(text, planner, solver, taxonomy, *, budget_s=60., max_retries=3, clock=perf_counter):
    """Injectable orchestration; the production caller also enforces a hard deadline."""
    from .natural_pose_schema import validate_proposal
    started = clock(); deadline = started + budget_s
    attempts = []; previous = None; feedback = []; near_candidates=[]
    for _ in range(max_retries + 1):
        remaining = deadline - clock()
        if remaining <= 0:
            break
        row = {}; attempts.append(row)
        try:
            raw = planner(text, previous, feedback, remaining)
            previous = raw
            proposal = validate_proposal(raw, taxonomy)
        except (ValueError, TypeError, KeyError) as exc:
            # Pydantic errors omit untrusted input snippets, never echo credentials.
            feedback = ['자세 JSON/태그 형식 오류: ' + type(exc).__name__]
            if hasattr(exc, 'errors'):
                feedback += [str(e['loc']) + ': ' + e['type'] for e in exc.errors(include_input=False)][:8]
            row.update(status='invalid_spec', reasons=feedback)
            continue
        if clock() >= deadline:
            break
        result = solver(proposal)
        feedback = result.get('reasons', [])
        row.update(status='passed' if result['passed'] else 'rejected', reasons=feedback,
                   validation=result.get('validation', {}), proposal=proposal.model_dump(exclude_none=True))
        if result.get('near_candidate'):
            near=dict(result['near_candidate'],kind='generated_near_match',refine_allowed=False,
                      release_allowed=False,lookup_allowed=False,actual_mesh_validated=False,
                      review_required=True,not_exact_match=True)
            if all(c.get('bvh_sha256')!=near.get('bvh_sha256') for c in near_candidates):near_candidates.append(near)
        if clock() >= deadline:
            break
        if result['passed']:
            candidate = dict(result['candidate'], kind='generated_preview',
                             refine_allowed=False, release_allowed=False, lookup_allowed=False,
                             actual_mesh_validated=False)
            return GenerationResult('preview_ready', candidates=[candidate], backend=VERSION,
                diagnostics={'attempts': attempts, 'review_required': True,
                             'semantic_alignment': 'not_verified_requires_human_review',
                             'skeleton_constraints_applied': False}, elapsed_ms=(clock()-started)*1000)
    timed_out = clock() >= deadline
    return GenerationResult('timeout' if timed_out else 'rejected',
        'generation_deadline_exhausted' if timed_out else 'validation_failed_after_retries',
        diagnostics={'attempts': attempts, 'review_required': True,'near_candidates':near_candidates[:3]}, backend=VERSION,
        elapsed_ms=(clock()-started)*1000)


class NaturalLanguagePoseGenerator:
    def __init__(self, rig_path, taxonomy_path, *, budget_s=60., max_retries=3,
                 model='gemini-3.8-flash', review_dir=None):
        if not 0 < budget_s <= 60 or type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError('generation budget must be <=60s; retries <=3')
        self.rig = Path(rig_path).resolve()
        self.taxonomy = Path(taxonomy_path).resolve()
        self.budget_s, self.max_retries, self.model = budget_s, max_retries, model
        self.review_dir = Path(review_dir).resolve() if review_dir else None

    def generate(self, description):
        if description.visible is not True or description.certainty not in ('high', 'medium'):
            return GenerationResult('unsupported', 'description_not_trusted', backend=VERSION)
        text = '. '.join(dict.fromkeys(part.strip() for part in (
            description.search_text, description.body_details) if part and part.strip()))
        scope=getattr(description,'visible_scope',None)
        if scope in ('head','shoulder_head','half'):
            text += (' . 출력은 '+scope+' 범위만 사용한다. 화면 밖 하체는 중립 서기 지지 자세로 둔다. '
                     '보이지 않는 부위는 관측 정답이 아닌 기본값이다. 보이는 머리/어깨/팔 자세를 우선한다.')
        relations=getattr(description,'pose_relations',[])
        if relations:text+=' . 관찰 동작 관계: '+json.dumps(relations,ensure_ascii=False)
        from .pose_requirements import requirements
        from .semantic_search import SemanticQuery
        required=requirements(SemanticQuery.from_description(description),dict(objects=description.objects,hand_count=None))
        return self.generate_text(text,requirements=required)

    def generate_text(self, text, *, requirements=None):
        started = perf_counter()
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            return GenerationResult('unsupported', 'invalid_text', backend=VERSION)
        try:
            if file_hash(self.rig) != RIG_SHA256:
                raise ValueError('rig_hash_mismatch')
            taxonomy_hash = file_hash(self.taxonomy)
            payload = dict(text=text.strip(), taxonomy=str(self.taxonomy), taxonomy_sha256=taxonomy_hash,
                           budget_s=self.budget_s, max_retries=self.max_retries, model=self.model,
                           requirements=requirements)
            env = {**os.environ, 'POSE_GENERATION_RIG': str(self.rig),
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
            remaining = self.budget_s - (perf_counter()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired('generation', self.budget_s)
            run = subprocess.run([sys.executable, '-m', 'src.natural_pose_worker'],
                input=json.dumps(payload), text=True, capture_output=True, timeout=remaining,
                cwd=Path(__file__).resolve().parents[1], env=env)
            if run.returncode:
                return GenerationResult('error', 'generation_worker_failed', backend=VERSION,
                                        elapsed_ms=(perf_counter()-started)*1000)
            result = GenerationResult(**json.loads(run.stdout))
            if self.review_dir and result.status == 'preview_ready':
                self._enqueue(result)
            result.elapsed_ms = (perf_counter()-started)*1000
            return result
        except subprocess.TimeoutExpired:
            return GenerationResult('timeout', 'generation_deadline_exhausted', backend=VERSION,
                                    elapsed_ms=(perf_counter()-started)*1000)
        except Exception as exc:
            return GenerationResult('error', 'generation_setup_error:'+type(exc).__name__, backend=VERSION,
                                    elapsed_ms=(perf_counter()-started)*1000)

    def _enqueue(self, result):
        """Append-only private queue, never the live library; random review IDs."""
        import uuid
        self.review_dir.mkdir(parents=True, exist_ok=True)
        target = self.review_dir / ('review-' + uuid.uuid4().hex)
        target.mkdir(mode=0o700)
        (target/'pose.bvh').write_text(result.candidates[0]['bvh'], encoding='utf-8')
        (target/'review.json').write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        result.diagnostics['review_id'] = target.name


def build_natural_generator():
    """Local runtime enabled by explicit setting, not by attaching API keys alone."""
    if os.getenv('POSE_GENERATION_ENABLED', '0') != '1':
        return None
    root = Path(os.getenv('DATA_DIR', 'data'))
    return NaturalLanguagePoseGenerator(
        os.getenv('POSE_GENERATION_RIG', str(root/'bvh/webtoon_combat_03.bvh')),
        os.getenv('POSE_GENERATION_TAXONOMY', str(root/'search-taxonomy-v2.json')),
        model=os.getenv('POSE_GENERATION_MODEL', 'gemini-3.8-flash'),
        review_dir=os.getenv('POSE_GENERATION_REVIEW_DIR', 'tmp/generated-pose-review'))
