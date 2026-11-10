"""Find body-clause candidates corroborated by a specific activity or contact."""

from collections import Counter
import re
from .bound_body_recall_probe import description_roles
from .bound_body_recall_probe import geometry_support

GENERIC_ACTIVITIES = frozenset({'model_pose', 'rest', 'waiting', 'fitness', 'standing', 'walking'})


GENERIC_ACTIONS = frozenset({'look', 'pose_for_display', 'hold'})


CONTACT_ROLES = frozenset({'chin', 'head', 'other_arm', 'waist'})


PROP_TERMS = {'umbrella': 'umbrella', 'smartphone': 'phone', 'sports ball': 'ball'}


def normalized(label):
    return re.sub(r'[\s_]+', '', label)


class ContextSlots:
    def __init__(self, records):
        self.labels, self.activities, self.actions, self.objects = {}, {}, {}, {}
        for row in records:
            pid = row['pose_id']
            groups = row['groups']
            self.labels[pid] = {normalized(tag['label']) for tag in groups.get('pose_action', [])}
            self.activities[pid] = {tag['value'] for tag in groups.get('scene', []) if tag['field'] == 'activity'}
            self.actions[pid] = {tag['value'] for tag in groups.get('pose_action', []) if tag['field'] == 'action'}
            self.objects[pid] = {tag['value'] for tag in groups.get('objects', []) if tag['field'] == 'object'}
        self.frequency = Counter(term for labels in self.labels.values() for term in labels)

    @staticmethod
    def clear_winner(candidates, gap):
        if not candidates:
            return None
        if len(candidates) == 1 or candidates[0][1] - candidates[1][1] >= gap:
            return candidates[0][0]
        return None

    def propose(self, description, text):
        if description.get('certainty') not in ('medium', 'high'):
            return None
        top = (text.get('aligned_clauses') or {}).get('top', ())
        if not top:
            return None
        contexts = set(description.get('activities') or ()) - GENERIC_ACTIVITIES
        actions = set(description.get('actions') or ()) - GENERIC_ACTIONS
        if contexts and actions:
            relevant = [(pid, score) for pid, score in top[:12]
                        if contexts & self.activities.get(pid, set()) and actions & self.actions.get(pid, set())]
            winner = self.clear_winner(relevant, .025)
            if winner is not None:
                return dict(pose_id=winner, basis='specific_activity_and_action')
        if description.get('visible_scope') not in ('half', 'bust'):
            return None
        contacts = set()
        for clause in (description.get('body_details') or '').split('·'):
            limb_roles = description_roles({'body_details': clause}, False)[0]
            term = normalized(clause)
            if any(role in CONTACT_ROLES for role, _ in limb_roles) and 1 <= self.frequency[term] <= 50:
                contacts.add(term)
        if contacts:
            relevant = [(pid, score) for pid, score in top[:12] if contacts & self.labels.get(pid, set())]
            winner = self.clear_winner(relevant, .016)
            if winner is not None:
                return dict(pose_id=winner, basis='specific_visible_contact')
        return None


    def head_only(self, person, description, text, models):
        search = person['dynamic_search']
        joint = search['geometry_evidence']
        if (search['route'] != 'undetermined' or joint.get('policy_block') or not description.get('visible')
                or description.get('visible_scope') not in ('half', 'bust')
                or description.get('body_details') or description.get('objects')):
            return None
        direction = description.get('head_view') or ''
        sentence = description.get('search_text') or ''
        if not direction.endswith(('left', 'right')) or not re.search(r'(고개|머리).{0,32}돌', sentence):
            return None
        head_turn = normalized('고개 옆으로 돌리기')
        shortlist = (text.get('text') or {}).get('top', ())
        for pid, _ in shortlist[:15]:
            if head_turn not in self.labels.get(pid, ()):
                continue
            independent = set()
            for name, model in models.items():
                if any(candidate == pid for profile in model['profiles'].values()
                       for method in profile.values() for candidate, _ in method['top'][:25]):
                    family = ('body_x' if name.startswith('body_x') else name
                              if name in ('original', 'rtmw', 'consensus') else 'crop')
                    independent.add(family)
            if len(independent) >= 2:
                return dict(pose_id=pid, basis='visible_lateral_head_turn')
        return None


    def detected_prop(self, person, description, text, models, hits):
        joint = person['dynamic_search']['geometry_evidence']
        if (joint['sufficient'] or joint.get('policy_block') or description.get('objects')
                or not description.get('visible')):
            return None
        props = {PROP_TERMS[hit['label']] for hit in hits
                 if hit['label'] in PROP_TERMS and hit['score'] >= .30}
        if not props:
            return None
        support = geometry_support(models)
        clauses = (text.get('aligned_clauses') or {}).get('top', ())
        candidates = [(pid, score) for pid, score in clauses[:12]
                      if props & self.objects.get(pid, set()) and support.get(pid, 0.) >= .16]
        winner = self.clear_winner(candidates, .025)
        if winner is not None:
            return dict(pose_id=winner, basis='locally_detected_prop_with_pose_support')
        return None
