"""Evaluate a rough-grouped reranker without exposing the held-out pose labels."""

from collections import defaultdict
import math
import numpy as np

GROUPS = ('original', 'tight', 'wide', 'rotated', 'rtmw')


PROFILES = ('upper_xy', 'upper_head', 'full_xy', 'full_head', 'full_depth')


MODES = ('geometry', 'hybrid')


TEXT = ('text', 'details', 'both')


DENSE = tuple(f'dense_{kind}' for kind in TEXT)


CLAUSES = ('dense_clauses', 'dense_action_tags')


BOUND_ROLES = ('bound_roles', 'bound_joint_roles')


ALIGNED_CLAUSES = 'aligned_clauses'


def group(name):
    if name == 'consensus':
        return name
    if name in ('body_x', 'body_x_mirror'):
        return 'body_x'
    return 'rotated' if name in ('tight1', 'tight2', 'tight3') else next(
        g for g in GROUPS if name.startswith(g))


def inputs_for_person(probe, language, baseline, *, relational=False, dense=False, outline=False, enhanced=None,
                      evidence_margin=False, limb_ink=False, clauses=False, silhouette=False):
    # Score all source lists on the same footing. No ID-derived feature is used.
    features = defaultdict(dict)
    for place, (pid, _) in enumerate(baseline, 1):
        features[pid]['prior'] = 1 / math.sqrt(place)
        features[pid]['prior_present'] = 1.
    for kind, data in language.items():
        for place, (pid, score) in enumerate(data['top'], 1):
            features[pid]['lexical_rank', kind] = 1 / math.sqrt(place)
            features[pid]['lexical_score', kind] = max(0., (score - .65) * 3.) if kind in DENSE else score
    for name, model in probe.items():
        model_group = group(name)
        confidence = np.asarray(model['scores'])[5:]
        mask = np.asarray(model['joints'])[5:]
        reliability = float(np.clip(confidence, 0, 1)[mask].mean()) if mask.any() else 0.
        for profile, methods in model['profiles'].items():
            for mode, data in methods.items():
                best = data['top'][0][1] if data['top'] else 0.
                cutoff = data['top'][5][1] if evidence_margin and len(data['top']) >= 6 else best
                for place, (pid, cost) in enumerate(data['top'], 1):
                    row = features[pid]
                    value = 1 / math.sqrt(place)
                    for key, signal in (
                        (('model_rank', model_group, mode), value),
                        (('profile_rank', profile, mode), value),
                        (('relative', mode), math.exp(-9 * (cost - best))),
                        (('absolute', mode), math.exp(-5 * cost)),
                        (('reliability', mode), value * reliability),
                    ):
                        row[key] = max(row.get(key, 0.), signal)
                    key = 'support', mode
                    row[key] = row.get(key, 0.) + value / 10
                    if evidence_margin and cost < cutoff:
                        margin = min(1., (cutoff - cost) * 16.) * reliability
                        key = 'winner_margin', model_group, mode
                        row[key] = max(row.get(key, 0.), margin)
    ids = list(features)
    profiles = PROFILES + ('upper_bend', 'full_bend') if relational else PROFILES
    groups = GROUPS + ('consensus',) if enhanced else GROUPS
    if enhanced in ('occluded', 'body_frame', 'continuous_camera', 'limb_consensus'):
        profiles += ('torso', 'trusted_full', 'steep', 'proportion')
    if enhanced in ('body_frame', 'continuous_camera', 'limb_consensus'):
        profiles += ('body_frame',)
        groups += ('body_x',)
    if enhanced in ('continuous_camera', 'limb_consensus'):
        profiles += ('refit_upper', 'refit_full')
    if enhanced == 'limb_consensus':
        profiles += ('limb_consensus',)
    text = TEXT + DENSE if dense else TEXT
    if outline:
        text += ('outline',)
    if limb_ink:
        text += ('limb_ink',)
    if clauses:
        text += CLAUSES
    if silhouette:
        text += ('silhouette',)
    names = (
        'prior', 'prior_present',
        *((key, kind) for key in ('lexical_rank', 'lexical_score') for kind in text),
        *(('model_rank', g, mode) for g in groups for mode in MODES),
        *(('profile_rank', p, mode) for p in profiles for mode in MODES),
        *((key, mode) for key in ('relative', 'absolute', 'reliability', 'support') for mode in MODES),
    )
    if evidence_margin:
        names += tuple(('winner_margin', group, mode) for group in groups for mode in MODES)
        names += ('independent_geometry', 'independent_hybrid', 'geometry_with_text', 'text_with_prior')
        for row in features.values():
            for name, mode in (('independent_geometry', 'geometry'), ('independent_hybrid', 'hybrid')):
                # A close fit from two different recognizers/crops counts more than
                # several related fitting profiles from the same skeleton.
                votes = sorted((row.get(('model_rank', g, mode), 0.) for g in groups), reverse=True)
                row[name] = votes[1] * votes[0]
            lexical = max((row.get(('lexical_rank', kind), 0.) for kind in text), default=0.)
            row['geometry_with_text'] = lexical * max(
                row.get(('model_rank', g, 'geometry'), 0.) for g in groups)
            row['text_with_prior'] = lexical * row.get('prior', 0.)
    matrix = np.array([[features[pid].get(name, 0.) for name in names] for pid in ids],
                      dtype=float).reshape(len(ids), len(names))
    return ids, matrix
