"""Score structured text against uniformly measured whole-library BVH cues."""

from collections import Counter
from collections import defaultdict
import math
import re
import numpy as np
from .evaluate import validation
from .partial_arm_search import PoseCueIndex
from .partial_arm_search import read_cues
from .robust_geometry import soften
from src.semantic_search import SemanticQuery
from src.semantic_search_285 import normalize_text

FIELDS = {
    'postures': ('posture', 1.2), 'actions': ('action', 2.1),
    'objects': ('object', 2.2), 'domains': ('domain', .3),
    'activities': ('activity', .9), 'intents': ('intent', .3), 'emotions': ('emotion', .25),
}


GENERIC_ACTIONS = frozenset({'gesture', 'look', 'pose_for_display', 'rest', 'wait', 'hold', 'talk'})


LEG_TAGS = {
    '다리 교차하기': 'crossed', '다리 꼬기': 'crossed', '다리 벌리기': 'apart',
    '한쪽 다리에 체중 싣기': 'one_side_support', '한 다리 앞으로 내밀기': 'staggered',
    '한 무릎 들기': 'raised_knee', '한 다리 들기': 'raised_foot',
    '한 무릎 굽히기': 'one_knee_bent', '무릎 굽히기': 'knees_bent',
}


ARM_TAGS = {
    '양손 얼굴 옆에 두기': 'hands_near_face', '양손 허리': 'hands_at_hips',
    '팔을 가슴 앞에 포개기': 'forearms_across', '양손 앞으로 내밀기': 'hands_forward',
    '반대 손목 잡기': 'wrists_close',
}


CONTACTS = (
    (r'반대(?:쪽)?(?:팔꿈치|팔).{0,10}(?:잡|감싸|받치|당기)', 'other_elbow'),
    (r'반대(?:쪽)?손목.{0,8}(?:잡|감싸|받치)', 'other_wrist'),
    (r'반대(?:쪽)?어깨.{0,8}(?:잡|감싸|얹|대)', 'other_shoulder'),
    (r'(?:무릎|허벅지).{0,7}(?:손|잡|짚|얹|대)', 'knee'),
    (r'(?:이마|얼굴|턱|볼|귀|관자놀이|머리).{0,7}(?:손|대|짚|만지|받치)', 'head'),
    (r'(?:양손|두손).{0,10}(?:모으|맞대|포개|깍지|맞잡)', 'hands_together'),
    (r'(?:한쪽|한|반대(?:쪽)?)다리.{0,10}(?:뒤로|뒤쪽).{0,5}(?:들|올리|굽)', 'leg_behind'),
    (r'(?:상체|몸통).{0,7}(?:앞으로|깊이).{0,5}(?:숙|기울)', 'torso_inclined'),
)


def token(value):
    return normalize_text(value.replace('_', ' ')).replace(' ', '')


def contact_cues(body):
    fragments = [token(part) for part in re.split(r'[·,;.]', body or '')
                 if not re.search(r'않|아니|불명|가려|안보|추정', part)]
    return {cue for pattern, cue in CONTACTS if any(re.search(pattern, text) for text in fragments)}


def supports_semantic_fallback(raw):
    if not raw or not raw.get('visible') or raw.get('visible_scope') in ('head', 'shoulder_head'):
        return False
    contacts = contact_cues(raw.get('body_details'))
    arm_roles = set(read_cues(raw).roles) - contacts
    return bool(contacts) and len(contacts) + len(arm_roles) >= 2


def rerank_semantic_geometry(rows, semantic, distances, positions):
    """Let one candidate backed by both inputs enter the first page."""
    eligible = [i for i, row in enumerate(rows) if row.get('match_level') != 'near_match'
                and row['kind'] != 'framing_reference'][:24]
    if len(eligible) < 6:
        return rows
    row_by_id = {rows[i]['pose_id']: i for i in eligible}
    proposals = []
    for place, hit in enumerate(semantic[:32], 1):
        pid = hit['pose_id']
        if pid not in row_by_id or hit['score'] < .35:
            continue
        distance = float(distances[positions[pid]])
        if distance > .09:
            continue
        score = distance + .026 * math.log(place) - .05 * hit['score']
        proposals.append((score, row_by_id[pid], pid))
    ordered = sorted(proposals)[:3]
    fifth = eligible[4]
    boundary = next((score for score, index, _ in proposals if index == fifth), float('inf'))
    for score, index, _ in ordered:
        if index <= fifth or score + .008 >= boundary:
            continue
        conflicts = (rows[index].get('core_conditions') or {}).get('conflicts') or ()
        if any(k == 'held_object' or k == 'hand_count' or k.startswith('held_hand:') for k in conflicts):
            continue
        result = list(rows)
        result.insert(fifth, result.pop(index))
        return result
    return rows


class StructuredIndex:
    def __init__(self, engine):
        self.index = engine.semantic.pool_service.index
        self.ids = engine.camera.pose_ids
        self.positions = engine.validator.positions
        self.postings = defaultdict(set)
        self.object_terms = defaultdict(set)
        self.prop_ids = set()
        for doc in self.index.docs:
            if doc.pose_id not in self.positions:
                continue
            i = self.positions[doc.pose_id]
            for field, value in doc.match_tags:
                # Every pose is treated alike even when older source tags omit
                # body arrangement. Physical features are measured separately.
                if field != 'body_configuration':
                    self.postings[field, token(value)].add(i)
            for tag in doc.groups.get('objects', []):
                if tag['field'] == 'object':
                    self.object_terms[token(tag['value'])].add(i)
                    self.prop_ids.add(doc.pose_id)
                    self.object_terms[token(tag['label'])].add(i)
        self.arm = PoseCueIndex(engine.camera)
        points, scores = engine.camera.points, engine.camera.scores
        pelvis = points[:, [11, 12]].mean(axis=1)
        shoulder = points[:, [5, 6]].mean(axis=1)
        torso = np.linalg.norm(shoulder - pelvis, axis=-1).clip(1e-6)
        up = (shoulder - pelvis) / torso[:, None]
        lateral = points[:, 6] - points[:, 5]
        lateral -= (lateral * up).sum(axis=1, keepdims=True) * up
        lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True).clip(1e-6)
        front = np.cross(lateral, up)

        def near(a, b, limit):
            distance = np.linalg.norm(points[:, a] - points[:, b], axis=-1) / torso
            reliable = (scores[:, a] >= .3) & (scores[:, b] >= .3)
            return (distance < limit) & reliable

        ankles = points[:, [15, 16]] - pelvis[:, None]
        depth = np.einsum('nid,nd->ni', ankles, front) / torso[:, None]
        ankles_valid = (scores[:, [15, 16]] >= .3).all(axis=1)
        self.cues = {
            'other_elbow': near(9, 8, .35) | near(10, 7, .35),
            'other_wrist': near(9, 10, .26),
            'other_shoulder': near(9, 6, .35) | near(10, 5, .35),
            'knee': near(9, 13, .4) | near(10, 14, .4) | near(9, 14, .4) | near(10, 13, .4),
            'head': near(9, 0, .45) | near(10, 0, .45),
            'hands_together': near(9, 10, .27),
            'leg_behind': ((depth < -.45).any(axis=1) & (np.ptp(depth, axis=1) > .4) & ankles_valid),
            'torso_inclined': ((np.abs(up[:, 1]) >= .30) & (np.abs(up[:, 1]) <= .86)),
        }

        across = np.einsum('nid,nd->ni', ankles, lateral) / torso[:, None]
        ankle_height = np.ptp(points[:, [15, 16], 1], axis=1) / torso
        knee_height = np.ptp(points[:, [13, 14], 1], axis=1) / torso
        upper = points[:, [11, 12]] - points[:, [13, 14]]
        lower = points[:, [15, 16]] - points[:, [13, 14]]
        length = np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1)
        cosine = np.sum(upper*lower, axis=-1) / length.clip(1e-6)
        bent = np.clip((cosine + .85) / .85, 0., 1.)
        support_distance = np.linalg.norm(ankles[..., [0, 2]], axis=-1) / torso[:, None]
        valid = (scores[:, 11:17] >= .3).all(axis=1) & (length > 1e-6).all(axis=1)
        measures = {
            'crossed': (across[:, 0] - across[:, 1] + .14) / .44,
            'apart': (np.abs(across[:, 0] - across[:, 1]) - .34) / .64,
            'one_side_support': (np.ptp(support_distance, axis=1) - .08) / .45,
            'staggered': (np.ptp(depth, axis=1) - .22) / .65,
            'raised_knee': (knee_height - .12) / .48,
            'raised_foot': (ankle_height - .13) / .58,
            'one_knee_bent': np.ptp(bent, axis=1),
            'knees_bent': bent.mean(axis=1),
        }
        self.leg_cues = {name: np.clip(value, 0., 1.) * valid for name, value in measures.items()}

        wrist = points[:, [9, 10]]
        hand = wrist - pelvis[:, None]
        reach = wrist - points[:, [5, 6]]
        hand_depth = np.einsum('nid,nd->ni', reach, front) / torso[:, None]
        hand_height = np.einsum('nid,nd->ni', hand, up) / torso[:, None]
        hand_lateral = np.einsum('nid,nd->ni', hand, lateral) / torso[:, None]
        face_dist = np.linalg.norm(wrist - points[:, 0, None], axis=-1) / torso[:, None]
        waist_dist = np.linalg.norm(wrist - points[:, [11, 12]], axis=-1) / torso[:, None]
        same_depth = np.sign(hand_depth[:, 0]) == np.sign(hand_depth[:, 1])
        arms_valid = (scores[:, [5, 6, 7, 8, 9, 10, 11, 12]] >= .3).all(axis=1)
        cross_body = (hand_lateral * [1, -1]).min(axis=1)
        in_front = np.abs(hand_depth).min(axis=1)
        arm_measures = {
            'hands_near_face': (1. - face_dist.max(axis=1)) / .58 * (scores[:, 0] >= .3),
            'hands_at_hips': (.76 - waist_dist.max(axis=1)) / .44,
            'forearms_across': (cross_body + .12) / .36 *
                ((hand_height.min(axis=1) >= .18) & (hand_height.max(axis=1) <= 1.34)),
            'hands_forward': (in_front - .18) / .57 * same_depth,
            'wrists_close': (.43 - np.linalg.norm(wrist[:, 0] - wrist[:, 1], axis=-1) / torso) / .36,
        }
        self.arm_cues = {name: np.clip(value, 0., 1.) * arms_valid for name, value in arm_measures.items()}

    def scores(self, query, raw, *, legs_observed=False, refine_arms=False):
        matches = np.zeros(len(self.ids), dtype=np.float32)
        total = 0.
        physical = 0.
        for attribute, (field, weight) in FIELDS.items():
            for value in set(getattr(query, attribute)):
                if value in ('standing', 'daily', 'pose_for_display', 'gesture'):
                    continue
                hits = self.postings.get((field, token(value)))
                if not hits:
                    continue
                specificity = min(2.6, 1. + .35 * math.log(len(self.ids) / len(hits)))
                contribution = weight * specificity
                matches[list(hits)] += contribution
                total += contribution
                if field in ('action', 'posture', 'object'):
                    physical += contribution
        for cue in contact_cues(query.body_details):
            weight = 1.3 if cue in ('head', 'knee', 'torso_inclined') else 1.9
            matches += weight * self.cues[cue]
            total += weight
            physical += weight
        arms = read_cues(raw)
        if arms.roles:
            weight = min(2., .85 * len(arms.roles))
            matches += weight * self.arm.match_scores(arms)
            total += weight
            physical += weight
        if legs_observed or refine_arms:
            tags = set((query.body_details or '').split(' · '))
            for enabled, mapping, cues, weight in ((legs_observed, LEG_TAGS, self.leg_cues, 1.65),
                                                   (refine_arms, ARM_TAGS, self.arm_cues, 1.85)):
                if enabled:
                    for cue in {mapping[tag] for tag in tags if tag in mapping}:
                        matches += weight * cues[cue]
                        total += weight
                        physical += weight
        if total <= 0. or physical < 1.25:
            return None
        return matches / total

    def compound_matches(self, query):
        actions = [self.postings.get(('action', token(value)), set())
                   for value in query.actions if value not in GENERIC_ACTIONS]
        objects = []
        for value in query.objects:
            label = self.index.lexicon.labels.get(('object', value), value)
            objects.append(self.object_terms.get(token(value), set()) |
                           self.object_terms.get(token(label), set()))
        return set.union(*actions) & set.intersection(*objects) if actions and objects else set()

    def joint_candidates(self, engine, description, required, prior, fitted_costs, *,
                         preserve_matches=False, legs_observed=False, refine_arms=False, contacts=None):
        """Find agreement across the full library before either shortlist cuts it off."""
        query = SemanticQuery.from_description(description)
        if query.visible_scope in ('head', 'shoulder_head'):
            return prior
        values = self.scores(query, description.to_dict(), legs_observed=legs_observed, refine_arms=refine_arms)
        if values is None and contacts is None:
            return prior
        if values is None:
            values = np.zeros(len(self.ids), dtype=np.float32)
        contact_values = np.nan_to_num(contacts, nan=0.) if contacts is not None else np.zeros_like(values)
        cost = fitted_costs - .45 * values - .29 * contact_values
        best = float(cost.min())
        additions = {}
        families = set()
        for i in np.argsort(cost, kind='stable')[:16]:
            if cost[i] > best + .065:
                break
            if not np.isfinite(cost[i]) or (values[i] < .30 and contact_values[i] < .70):
                continue
            pid = self.ids[i]
            family = self.index.family.get(pid, pid)
            if family in families:
                continue
            checked = None
            if required and required['active']:
                checked, penalty = validation(engine.validator.check(required, pid), 'unknown_soft')
                checked, _ = soften(checked, penalty, strong_geometry=True)
                if checked['status'] != 'passed':
                    continue
            hit = self.make_hit(engine, pid, float(values[i]), checked)
            if hit is not None:
                hit['matched']['joint_geometry_and_tags'] = dict(
                    fitted_cost=float(fitted_costs[i]), joint_rank=len(additions) + 1)
                if contact_values[i] >= .70:
                    hit['matched']['measured_fingertip_agreement'] = float(contact_values[i])
                additions[pid] = hit
                families.add(family)
            if len(additions) >= 3:
                break
        if not additions:
            return prior
        existing, references = [], []
        for hit in prior.get('hits', []):
            if hit.get('matched', {}).get('visible_region_reference'):
                references.append(hit)
            else:
                proposal = additions.pop(hit['pose_id'], None)
                if proposal and preserve_matches:
                    proposal['matched'] = dict(hit.get('matched', {}), **proposal['matched'])
                existing.append(proposal or hit)
        hits = existing[:40 - len(additions)] + list(additions.values()) + references
        return dict(prior, hits=hits, status='ok', reason=None)

    def semantic_candidates(self, engine, description, required, prior, *,
                            soft_arms=False, legs_observed=False, refine_arms=False, contacts=None):
        query = SemanticQuery.from_description(description)
        if query.visible_scope in ('head', 'shoulder_head'):
            return prior
        values = self.scores(query, description.to_dict(), legs_observed=legs_observed, refine_arms=refine_arms)
        if values is None and contacts is None:
            return prior
        if contacts is not None:
            if values is None:
                values = .80 * np.nan_to_num(contacts, nan=0.)
            else:
                values = np.where(np.isfinite(contacts), .65 * values + .35 * contacts, values)
        refs = [h for h in prior.get('hits', []) if h.get('matched', {}).get('visible_region_reference')]
        old = [h for h in prior.get('hits', []) if h not in refs]
        compound = self.compound_matches(query) if soft_arms else set()
        picked = {h['pose_id']: h for h in old[:12]}
        families = Counter(self.index.family.get(pid, pid) for pid in picked)
        qualified = []
        for i in np.argsort(-values, kind='stable'):
            score = float(values[i])
            if score < .28 or (not soft_arms and len(picked) >= 40):
                break
            pid = self.ids[i]
            family = self.index.family.get(pid, pid)
            if not soft_arms and (pid in picked or families[family]):
                continue
            checked = None
            if required and required['active']:
                checked = engine.validator.check(required, pid)
                if soft_arms and score >= .40:
                    checked, cost = validation(checked, 'unknown_soft')
                    score = max(0., score - .6 * cost)
                if checked['status'] != 'passed':
                    continue
            if soft_arms:
                qualified.append((i in compound, score, pid, checked))
                continue
            hit = self.make_hit(engine, pid, score, checked)
            if hit is not None:
                picked[pid] = hit
                families[family] += 1
        if soft_arms:
            ranked = sorted(qualified, key=lambda row: (not row[0], -row[1], row[2]))
            preferred = [row for row in ranked if row[0] and row[1] >= .52][:4]
            if contacts is not None:
                contact_matches = [row for row in ranked
                                   if row[1] >= .52 and contacts[self.positions[row[2]]] >= .78][:3]
                preferred.extend(row for row in contact_matches if row not in preferred)
            picked = {}
            families.clear()
            for exact, score, pid, checked in preferred:
                hit = self.make_hit(engine, pid, score, checked, exact=exact)
                if hit is not None:
                    if contacts is not None and contacts[self.positions[pid]] >= .78:
                        hit['matched']['measured_fingertip_agreement'] = float(contacts[self.positions[pid]])
                    picked[pid] = hit
                    families[self.index.family.get(pid, pid)] += 1
            for hit in old[:12]:
                if hit['pose_id'] not in picked:
                    picked[hit['pose_id']] = hit
                    families[self.index.family.get(hit['pose_id'], hit['pose_id'])] += 1
            for exact, score, pid, checked in ranked:
                if len(picked) >= 40:
                    break
                family = self.index.family.get(pid, pid)
                if families[family] or pid in picked:
                    continue
                hit = self.make_hit(engine, pid, score, checked, exact=exact)
                if hit is not None:
                    picked[pid] = hit
                    families[family] += 1
        for hit in old[12:]:
            if len(picked) >= 40:
                break
            picked.setdefault(hit['pose_id'], hit)
        hits = list(picked.values()) + refs
        return dict(prior, hits=hits, status='ok' if hits else prior['status'],
                    reason=None if hits else prior.get('reason'))

    def make_hit(self, engine, pid, score, checked, *, exact=False):
        pose = self.index.pose_json(pid)
        if not pose:
            return None
        matched = {'structured_bvh': True}
        if exact:
            matched['specific_action_and_objects'] = True
        hit = dict(pose_id=pid, score=score, pose=pose, matched=matched)
        if checked:
            hit['whole_pose_validation'] = checked
        return hit if engine._clean([hit], 'semantic') else None
