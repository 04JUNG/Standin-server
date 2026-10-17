"""Bounded post-retrieval mesh gate. No ranking, IK, BVH or hand edits.

Only the opposite-hand skin triangle test is certified by a clear receipt.
Unknown/errors are review evidence, never eligible library/generated results.
"""
import hashlib
import json
import re
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen

from converter.hand_contact import SCREEN_VERSION, validate_screen
from converter.hand_library import load_library, normalize_request
from converter.protocol import SOLVER_VERSION
from .refine_thumbnail import _multipart


def unknown(reason):
    return dict(status='unavailable', reason=reason, method=SCREEN_VERSION,
                scope='opposite_hands_only', complete_collision_certificate=False)


class HandSkinClient:
    def __init__(self, base_url, character_id, character_sha256, *, opener=None):
        self.base_url = base_url.rstrip('/')
        self.character_id, self.character_sha256 = character_id, character_sha256
        self.opener = opener or urlopen

    def check(self, text, hands, timeout):
        if not self.base_url or not self.character_sha256:
            return unknown('hand_screen_converter_not_configured')
        try:
            hands = normalize_request(hands)
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            fields = dict(character_id=self.character_id, expected_bvh_sha256=digest,
                          expected_character_sha256=self.character_sha256)
            if hands:
                fields.update({f'hand_{side}': preset for side, preset in hands.items() if preset})
            body, content_type = _multipart(fields=fields, file_field='bvh',
                filename='screen.bvh', file_bytes=text.encode('utf-8'))
            request = Request(self.base_url+'/screen-hand-contact', data=body, method='POST',
                              headers={'Content-Type': content_type, 'Accept': 'application/json'})
            with self.opener(request, timeout=timeout) as response:
                if response.status != 200:
                    return unknown('hand_screen_upstream_error')
                data = response.read(128*1024+1)
            if len(data) > 128*1024:
                raise ValueError('oversized receipt')
            receipt = json.loads(data)
            validate_screen(receipt)
            expected = dict(source_bvh_sha256=digest, character_id=self.character_id,
                character_sha256=self.character_sha256, solver_version=SOLVER_VERSION,
                hands=hands, hand_library_sha256=load_library().sha256)
            if any(receipt.get(k) != v for k, v in expected.items()):
                raise ValueError('skin receipt input lineage mismatch')
            if not re.fullmatch(r'[0-9a-f]{64}', receipt.get('screened_artifact_sha256', '')):
                raise ValueError('missing artifact hash')
            return receipt
        except Exception:
            # Timeout, old converter (404), invalid receipt, or network error:
            # do not expose private URLs/paths and never silently pass a pose.
            return unknown('hand_screen_request_or_integrity_failed')


class HandSkinGate:
    def __init__(self, client, bvh_root, hashes, *, budget_seconds=30., max_checks=10):
        self.client, self.root, self.hashes = client, Path(bvh_root), dict(hashes)
        self.budget_seconds, self.max_checks = budget_seconds, max_checks

    def session(self, hands=None):
        return SkinSession(self, hands)

    def library_text(self, pose_id):
        if pose_id not in self.hashes:
            raise ValueError('unbound BVH')
        path = self.root/(pose_id+'.bvh')
        if path.resolve().parent != self.root.resolve() or path.is_symlink():
            raise ValueError('unsafe BVH path')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.hashes[pose_id]:
            raise ValueError('changed BVH')
        return data.decode('utf-8')


class SkinSession:
    def __init__(self, gate, hands):
        self.gate = gate
        self.hands = hands
        self.spent_seconds = 0.
        self.checks = 0
        self.failed = False
        self.cache = {}
        self.rejected, self.unverified = [], []

    def check(self, text):
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if key in self.cache:
            return self.cache[key]
        remaining = self.gate.budget_seconds-self.spent_seconds
        if self.failed or remaining <= 0 or self.checks >= self.gate.max_checks:
            return unknown('hand_screen_budget_or_upstream_unavailable')
        self.checks += 1
        started = monotonic()
        try:
            result = self.gate.client.check(text, self.hands, timeout=min(remaining, 15.))
            validate_screen(result)
        except Exception:
            result = unknown('hand_screen_failed')
        self.spent_seconds += monotonic()-started
        self.failed = result['status'] == 'unavailable'
        self.cache[key] = result
        return result

    def filter(self, candidates, *, generated=False, limit=5):
        accepted = []
        for candidate in candidates:
            if len(accepted) >= limit:
                break
            try:
                text = candidate['bvh'] if generated else self.gate.library_text(candidate['pose_id'])
                if not isinstance(text, str) or not text:
                    raise ValueError('missing BVH')
                receipt = self.check(text)
            except Exception:
                receipt = unknown('hand_screen_bvh_unavailable_or_changed')
            result = {**candidate, 'hand_skin_screen': receipt}
            if receipt['status'] == 'clear_screen':
                accepted.append(result)
            else:
                # No BVH payload in failed diagnostics; do not accidentally
                # advertise an unsafe/unchecked generated artifact for reuse.
                diagnostic = dict(pose_id=candidate.get('pose_id'), generated=generated,
                    hand_skin_screen=receipt, review_required=True,
                    release_allowed=False, refine_allowed=False, actual_mesh_validated=False)
                target = self.rejected if receipt['status'] == 'intersection' else self.unverified
                if len(target) < 20:
                    target.append(diagnostic)
        return accepted

    def report(self):
        return dict(method=SCREEN_VERSION, checks=self.checks,
                    budget_seconds=self.gate.budget_seconds, max_checks=self.gate.max_checks,
                    elapsed_ms=round(self.spent_seconds*1000, 2),
                    scope='opposite_hands_only', complete_collision_certificate=False,
                    rejected=self.rejected, unverified=self.unverified)


def build_hand_skin_gate(camera, bvh_root):
    from .config import CFG
    characters = json.loads((Path(__file__).resolve().parents[1]/'config/characters.example.json').read_text())
    character_id = CFG.thumbnail_character_id
    digest = characters['characters'].get(character_id, {}).get('sha256')
    client = HandSkinClient(CFG.thumbnail_converter_url, character_id, digest)
    return HandSkinGate(client, bvh_root, camera.hashes)
