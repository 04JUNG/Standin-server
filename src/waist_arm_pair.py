"""Ranking policy for a hand at the waist with the opposite arm down."""

import re


GEOMETRY_CAP = 80
SEMANTIC_CAP = 40
SEMANTIC_WEIGHT = .3
MIN_TEXT_SCORE = .26
SIDE_CONFLICT_COST = .01
PAIR_MISMATCH_COST = .05
ARM_UNKNOWN_COST = .025

_PAIR_TAGS = {'한 손 허리', '한 팔 내리기'}
_CONTACT = re.compile(r'한\s*(?:쪽\s*)?손.{0,20}허리\s*(?:에|를)\s*(?:대|짚|얹|올려)')
_OPPOSITE_DOWN = re.compile(r'반대(?:쪽)?\s*(?:손|팔).{0,26}(?:내리|내린|내려|내렸|늘어뜨|아래로)')
_UNCERTAIN = re.compile(r'않|아닌|불명|안\s*보|가려|보이지')


def describes_pair(description):
    """Recognize the compact Luna tags or an explicit equivalent observation."""
    details = description.body_details or ''
    if _PAIR_TAGS.issubset(details.split(' · ')):
        return True
    for text in (details, description.search_text or ''):
        contact = _CONTACT.search(text)
        opposite = _OPPOSITE_DOWN.search(text)
        if contact and opposite and not _UNCERTAIN.search(text):
            return True
    return False


def soften_arm_checks(checked, *, conflicts):
    """Move only coarse arm checks into ranking, leaving other conditions intact."""
    original_conflicts, original_unknown = checked['conflicts'], checked['unknown']
    remaining_conflicts = [
        key for key in original_conflicts if not (conflicts and key.startswith('arm:'))
    ]
    remaining_unknown = [key for key in original_unknown if not key.startswith('arm:')]
    cost = (len(original_conflicts) - len(remaining_conflicts)) * PAIR_MISMATCH_COST
    cost += (len(original_unknown) - len(remaining_unknown)) * ARM_UNKNOWN_COST
    if not cost:
        return checked, cost
    status = ('conflict' if remaining_conflicts else 'needs_review' if remaining_unknown else
              'near_match' if checked['detail_conflicts'] or checked['detail_unknown'] else 'passed')
    return dict(checked, status=status, conflicts=remaining_conflicts, unknown=remaining_unknown,
                original_status=checked['status'], arm_penalty=cost), cost
