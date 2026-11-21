"""Match observed body arrangements to individual pose tags."""

import re
import numpy as np

SPLIT = re.compile(r'[·;/,\n]+')


class ClauseIndex:
    def __init__(self, records, encoder):
        self.encoder = encoder
        self.ids = np.asarray([row['pose_id'] for row in records])
        rows = [list(dict.fromkeys(tag['label'] for tag in row.get('groups', {}).get('pose_action', [])
                                   if tag.get('label'))) for row in records]
        tags = sorted({tag for row in rows for tag in row})
        if not tags:
            raise ValueError('No pose tags in library')
        vocab = {tag: i for i, tag in enumerate(tags)}
        self.entries = np.full((len(rows), max(map(len, rows))), len(tags), dtype=int)
        for i, row in enumerate(rows):
            self.entries[i, :len(row)] = [vocab[tag] for tag in row]
        self.present = self.entries != len(tags)
        self.tag_names = tuple(tags)
        self.tags = encoder.fit(tags)
        self.encoded = {}

    def tag_scores(self, queries):
        new = [phrase for phrase in queries if phrase not in self.encoded]
        for phrase in new:
            self.encoded[phrase] = self.encoder.encode(phrase)[0]
        similarities = np.array([self.encoded[phrase] for phrase in queries]) @ self.tags.T
        # Remove each clause's common background similarity before matching.
        floor = np.quantile(similarities, .70, axis=1, keepdims=True)
        return np.maximum(0., similarities - floor) / np.maximum(1. - floor, .01)

    def similarities(self, queries):
        return np.pad(self.tag_scores(queries), ((0, 0), (0, 1)))[:, self.entries].max(axis=-1)

    def search(self, description):
        detail = list(dict.fromkeys(part.strip() for part in SPLIT.split(description.get('body_details') or '')
                                    if part.strip()))
        actions = [str(value).replace('_', ' ') for value in description.get('actions') or ()]
        posture = [str(value).replace('_', ' ') for value in description.get('postures') or ()]
        for field, phrases in (('dense_clauses', detail), ('dense_action_tags', detail + actions + posture)):
            phrases = list(dict.fromkeys(phrases))
            if not phrases:
                continue
            by_clause = self.similarities(phrases)
            strength = by_clause.max(axis=1)
            weights = np.where(strength > .10, strength, 0.)
            if field == 'dense_action_tags' and detail:
                weights[len(detail):] *= .65
            if weights.sum() <= 0:
                continue
            mean_match = np.average(by_clause, weights=weights, axis=0)
            scores = .75 * mean_match + .25 * np.max(by_clause, axis=0)
            yield field, np.where(self.present.any(axis=1), scores, 0.)
