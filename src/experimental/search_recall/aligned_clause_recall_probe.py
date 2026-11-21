"""Assign observed pose clauses to distinct library tags before scoring."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .bound_body_recall_probe import description_roles
from .bound_body_recall_probe import geometry_support
from .clause_recall_probe import ClauseIndex
from .clause_recall_probe import SPLIT

def roles(text):
    arms, legs, body = description_roles({'body_details': text}, True)
    return dict((*arms, *legs, *((value, 'one') for value in body)))


def aligned_proposal(top, models):
    support = geometry_support(models)
    candidates = sorted(((pid, score + .25 * support.get(pid, 0.)) for pid, score in top[:12]),
                        key=lambda row: row[1], reverse=True)
    if len(candidates) < 2:
        return None
    (pid, strength), (_, runner_up) = candidates[:2]
    if support.get(pid, 0.) >= .5 and strength - runner_up > .018:
        return pid
    return None


class AlignedClauseIndex(ClauseIndex):
    def __init__(self, records, encoder):
        super().__init__(records, encoder)
        self.tag_roles = [roles(tag) for tag in self.tag_names]
        self.valid_entries = [row[mask] for row, mask in zip(self.entries, self.present, strict=True)]

    def search(self, description):
        phrases = list(dict.fromkeys(part.strip() for part in SPLIT.split(description.get('body_details') or '')
                                      if part.strip()))
        if len(phrases) < 2:
            return None
        matches = self.tag_scores(phrases)
        for i, phrase in enumerate(phrases):
            observed = roles(phrase)
            for j, tagged in enumerate(self.tag_roles):
                same = observed.keys() & tagged.keys()
                if same:
                    match_count = any((observed[role] == 'both') == (tagged[role] == 'both') for role in same)
                    matches[i, j] += .085 if match_count else .025
        strongest = matches.max(axis=1)
        weight = np.clip(strongest, .10, .55)
        weight /= weight.sum()
        weighted = matches * weight[:, None]
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for i, indexes in enumerate(self.valid_entries):
            if not len(indexes):
                continue
            candidate = weighted[:, indexes]
            row, tag = linear_sum_assignment(candidate, maximize=True)
            # The residual allows a compound library tag to retain some credit
            # when a description spells out its parts in separate clauses.
            scores[i] = .8 * candidate[row, tag].sum() + .2 * candidate.max(axis=1).sum()
        return scores
