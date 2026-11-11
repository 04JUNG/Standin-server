"""Use rough line contours to check candidates proposed by independent joint fits."""

from collections import defaultdict
from itertools import zip_longest
import numpy as np

def proposal_pool(model_profiles, language):
    ballots = defaultdict(float)
    for model in model_profiles.values():
        for profile in model['profiles'].values():
            for name, data in profile.items():
                for rank, (pose_id, _) in enumerate(data['top'][:10], 1):
                    ballots[pose_id] += (1. if name == 'geometry' else .9) / (3 + rank)
    for name, data in language.items():
        for rank, (pose_id, _) in enumerate(data['top'][:12], 1):
            ballots[pose_id] += (1.1 if name == 'details' else .65) / (3 + rank)
    votes = sorted(ballots, key=ballots.get, reverse=True)
    leaders = [
        data['top'][0][0] for model in model_profiles.values() for profile in model['profiles'].values()
        for data in profile.values() if data['top']
    ]
    interleaved = (pose_id for pair in zip_longest(votes, leaders)
                   for pose_id in pair if pose_id is not None)
    return list(dict.fromkeys(interleaved))[:55]


def prepare_queries(ranker, path, person, models, person_count):
    def confidence(model):
        scores = np.asarray(model['scores'])[5:]
        mask = np.asarray(model['joints'])[5:]
        return float(np.mean(np.where(mask, scores, 0.)))

    candidates = sorted(models.items(), key=lambda item: confidence(item[1]), reverse=True)
    first = [('original', models['original'])] if 'original' in models else []
    alternates = [item for item in candidates if item[0] in ('tight_mirror', 'wide_mirror', 'rtmw')]
    queries = []
    for _, model in first + alternates[:2]:
        evidence = dict(person_count=person_count, keypoints=model['keypoints'], joint_mask=model['joints'],
                        joint_scores=model['scores'], lower_body_observed=person['lower_body_observed'])
        query = ranker.prepare(path, person.get('box'), evidence)
        if query is not None:
            queries.append(query)
    return queries
