"""Bounded worker-side latest-revision fence; BFF persists selection authority.

One token per owner/person. Superseding revokes old running/queued work at its
next safe checkpoint. Restart revokes every old token; it never revives ready.
"""
import threading
from converter.solved_state import canonical, digest, validate_identity
from converter.interactive_export import Superseded


class LatestRevisionGate:
    def __init__(self, max_people=128):
        if type(max_people) is not int or max_people < 1:
            raise ValueError('invalid revision gate capacity')
        self._max = max_people
        self._latest = {}
        self._lock = threading.Lock()
        self._epoch = object()

    def restart(self):
        with self._lock:
            self._latest.clear(); self._epoch = object()

    def select(self, identity):
        validate_identity(identity)
        key = (identity['ownerBinding'], identity['selection']['person'])
        revision = identity['selection']['revision']
        signature = digest(canonical(identity))
        with self._lock:
            prior = self._latest.get(key)
            if prior and (revision < prior[0] or (revision == prior[0] and signature != prior[1])):
                raise Superseded('old or conflicting selection revision')
            if prior is None and len(self._latest) >= self._max:
                raise ValueError('revision gate capacity exceeded')
            self._latest[key] = (revision, signature)
            epoch = self._epoch
        def checkpoint():
            with self._lock:
                if epoch is not self._epoch or self._latest.get(key) != (revision, signature):
                    raise Superseded('selection superseded/cancelled/restarted')
        return checkpoint

    def cancel(self, owner_binding, person):
        with self._lock:
            # Retain the revision tombstone; an old request cannot restart work.
            key = (owner_binding, person)
            prior = self._latest.get(key)
            if prior:
                self._latest[key] = (prior[0], None)
