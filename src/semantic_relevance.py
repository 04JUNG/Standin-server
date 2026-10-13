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
        if preferred and preferred not in allowed:
            raise ValueError('preferred pose absent or quarantined: ' + preferred)
        preferences[key] = {
            'preferred_pose_id': preferred,
            'rejected_pose_ids': frozenset(entry.get('rejected_pose_ids', [])),
        }
    index.review_preferences = preferences
    index.identity['review_preferences_sha256'] = hashlib.sha256(raw).hexdigest()


def relevance_constraints(query, description, tags):
    """Return failed named requirements. Scene names alone cannot satisfy them."""
    failures = []
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
