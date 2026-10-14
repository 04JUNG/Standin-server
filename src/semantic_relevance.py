"""Query-specific feedback and action-role checks; no geometry/refine mutations.

Explicit preferences only reorder already eligible, above-threshold candidates for
the same normalized query. They are replayed feedback, not learned generalization.
"""
import hashlib
import json
import re
from pathlib import Path


def load_review_preferences(index, path, normalize):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get('schema_version') != 1:
        raise ValueError('unsupported review preference schema')
    allowed = {d.pose_id for d in index.docs}
    preferences = {}
    for entry in data['entries']:
        key = normalize(entry['query'])
        if key in preferences:
            raise ValueError('duplicate normalized review query')
        preferred = entry.get('preferred_pose_id')
        ordered = entry.get('preferred_pose_ids', [preferred] if preferred else [])
        if (not isinstance(ordered, list) or len(ordered) > 20
                or any(not isinstance(pid, str) or not pid for pid in ordered)
                or len(set(ordered)) != len(ordered)):
            raise ValueError('invalid ordered review preferences')
        if preferred and ordered and preferred != ordered[0]:
            raise ValueError('conflicting primary review preference')
        for pid in ordered:
            if pid not in allowed:
                raise ValueError('preferred pose absent or quarantined: ' + pid)
        preferences[key] = {
            'preferred_pose_id': ordered[0] if ordered else None,
            'preferred_pose_ids': tuple(ordered),
            'rejected_pose_ids': frozenset(entry.get('rejected_pose_ids', [])),
        }
    index.review_preferences = preferences
    index.identity['review_preferences_sha256'] = hashlib.sha256(raw).hexdigest()


def relevance_constraints(query, description, tags):
    """Return failed named requirements. Scene names alone cannot satisfy them."""
    failures = []
    # Required body relations cannot be replaced by a shared scene/tool label.
    if re.search(r'엎드(?:려|린)', query) and not re.search(r'엎드|복와|prone', description):
        failures.append('prone_not_generic_lying')
    if re.search(r'팔짱', query):
        # Crossed combat guard is not resting with arms folded.
        folded = ('action', 'fold_arms') in tags or bool(re.search(r'팔짱', description))
        if not folded:
            failures.append('folded_arms_required')
        if re.search(r'쪼그|쭈그|웅크', query) and not (
                ('posture', 'crouching') in tags or re.search(r'쪼그|쭈그|웅크|낮게 앉', description)):
            failures.append('folded_arms_and_crouch')
    if re.search(r'(?:이마|앞머리).*(?:짚|쓸|얹|대고)|손.*이마', query):
        if not re.search(r'이마|앞머리|머리|헤어|관자', description):
            failures.append('hand_to_forehead_not_other_grooming')
        if re.search(r'서 있는|선 자세', query) and re.search(r'앉|누워|누운|엎드', description):
            failures.append('standing_forehead_not_seated_or_lying')
    if re.search(r'(?:입|입가|입과 턱).*(?:틀어막|가리|막는)', query):
        if not (re.search(r'입|입가', description) and re.search(r'손|팔', description)):
            failures.append('hand_at_mouth_required')
    if re.search(r'양손|두 손', query) and re.search(r'휴대폰|스마트폰', query):
        if not re.search(r'양손|두 손', description):
            failures.append('twohand_phone_not_onehand')
        if re.search(r'앉', query) and not (
                ('posture', 'sitting') in tags or re.search(r'앉|좌식', description)):
            failures.append('seated_twohand_phone')
    hanging_bar = re.search(r'턱걸이|철봉', query) or (re.search(r'수평 (?:봉|바)|가로.*(?:난간|바)', query)
                                                        and re.search(r'끌어올|매달', query))
    if hanging_bar and not re.search(r'턱걸이|철봉|pull.?up|매달|수평 (?:봉|바)', description):
        failures.append('pullup_not_generic_arms_up')
    if re.search(r'탁자|책상', query) and re.search(r'설명|손짓|제스처|손바닥.*펼', query):
        desk = ('object', 'desk') in tags or bool(re.search(r'탁자|책상', description))
        gesture = ('action', 'gesture') in tags or bool(re.search(r'설명|손짓|제스처|손바닥.*펼', description))
        if not (desk and gesture):
            failures.append('desk_explanation_not_any_hand_gesture')
    if re.search(r'(?:한 팔|팔을?).*가슴.*(?:당|가로질)|가슴.*팔.*당', query) and re.search(r'스트레칭|몸을 푸|어깨를 푸', query):
        if not (re.search(r'가슴', description) and re.search(r'당|가로질|안쪽', description)
                and re.search(r'팔|어깨', description)):
            failures.append('crossbody_shoulder_stretch')
    if re.search(r'악수|handshake', query) and not re.search(r'악수|handshake', description):
        failures.append('handshake_action')
    if re.search(r'팔굽혀|푸시업|push.?up', query) and not re.search(r'팔굽혀|푸시업|push.?up', description):
        failures.append('push_up_not_swimming_or_crawling')
    helper = bool(re.search(r'부축|(?:사람|친구|상대|동료).{0,15}일으켜', query))
    if helper:
        assisted_person = bool(re.search(r'부축|상대|동료|친구|사람', description))
        assist_action = bool(re.search(r'부축|일으켜|일으키|끌어올려|끌어 올려|받쳐|받치|어깨로.*지지', description))
        if not (assisted_person and assist_action):
            failures.append('helper_not_fallen_person')
    if re.search(r'(?:배|복부)를?\s*맞|배.{0,10}아파', query):
        if not re.search(r'배|복부', description):
            failures.append('abdominal_reaction')
    if re.search(r'칼끝|검끝', query) and re.search(r'겨누|조준|겨냥', query):
        # A vertical sword guard is not tip-forward aiming, even if mislabeled aim.
        forward = bool(re.search(r'정면|전방|앞.{0,12}(?:뻗|내밀|찌르)|찌르', description))
        if not forward or ('object', 'sword') not in tags:
            failures.append('sword_tip_forward')
    return failures
