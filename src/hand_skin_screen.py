"""Bounded post-retrieval mesh gate. No ranking, IK, BVH or hand edits.

Only the opposite-hand skin triangle test is certified by a clear receipt.
Unknown/errors are review evidence, never eligible library/generated results.
"""
from contextlib import contextmanager
from copy import copy
import hashlib
import json
import math
import re
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from converter.hand_contact import SCREEN_VERSION, validate_screen
from converter.hand_library import load_library, normalize_request
from converter.protocol import SOLVER_VERSION
from .refine_thumbnail import _multipart


def unknown(reason, **diagnostics):
    return dict(status='unavailable', reason=reason, method=SCREEN_VERSION,
                scope='opposite_hands_only', complete_collision_certificate=False,
                **diagnostics)


WORK_CODES = {'PREVIEW_PREEMPTED', 'REQUEST_CANCELLED', 'WORK_BUDGET_EXHAUSTED', 'WORK_QUEUE_FULL'}
UPSTREAM_CODES = WORK_CODES | {'CONVERSION_TIMEOUT', 'BLENDER_UNAVAILABLE', 'WORKER_INTEGRITY_ERROR',
                               'CHARACTER_UNAVAILABLE', 'INVALID_BVH', 'SOURCE_BVH_SHA256_MISMATCH'}


def availability(headers):
    """Fixed enum/numeric diagnostics only, including untrusted HTTP headers."""
    result = {}
    for key, field in [('X-Standin-Queue-Wait-Ms', 'queue_wait_ms'),
                       ('X-Standin-Execution-Ms', 'execution_ms')]:
        try:
            value = float(headers.get(key))
            if math.isfinite(value) and 0 <= value <= 600_000:
                result[field] = value
        except (TypeError, ValueError):
            pass
    phase = headers.get('X-Standin-Work-Phase')
    if phase in ('request', 'queue', 'execution'):
        result['phase'] = phase
    conversion = headers.get('X-Standin-Conversion-Id', '')
    if re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}', conversion):
        result['conversion_id'] = conversion
    reason = headers.get('X-Standin-Work-Reason')
    if reason in WORK_CODES:
        result['cause'] = reason
    return result


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
            from .logging_setup import request_id_var
            request_id = request_id_var.get()
            trace = ({'X-Request-Id': request_id} if isinstance(request_id, str) and
                     re.fullmatch(r'req_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}', request_id) else {})
            request = Request(self.base_url+'/screen-hand-contact', data=body, method='POST',
                              headers={'Content-Type': content_type, 'Accept': 'application/json', **trace,
                                  'X-Standin-Work-Budget-Ms': str(max(1, min(15000, int(timeout*1000)-150))),
                                  **({'X-Standin-Screen-Optional': '1'}
                                     if getattr(self, '_screen_optional', False) else {})})
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
            mismatched = [k for k, v in expected.items() if receipt.get(k) != v]
            if mismatched:
                # Fixed field names only; no private values, URLs, body or exception text.
                return unknown('hand_screen_receipt_lineage_mismatch',
                               mismatched_fields=mismatched)
            if not re.fullmatch(r'[0-9a-f]{64}', receipt.get('screened_artifact_sha256', '')):
                raise ValueError('missing artifact hash')
            timing = availability(getattr(response, 'headers', {}))
            # Reconstruct from safe response headers, not arbitrary receipt extras.
            if receipt['status'] == 'unavailable':
                return unknown('hand_screen_mesh_unavailable',
                               availability={'status': 'unavailable', **timing})
            receipt.pop('reason', None)
            receipt['availability'] = {'status': 'available', **timing}
            return receipt
        except HTTPError as exc:
            status = exc.code
            timing = availability(exc.headers or {})
            try:
                code = json.loads(exc.read(4096)).get('error', {}).get('code')
            except (ValueError, TypeError, AttributeError, OSError):
                code = None
            finally:
                exc.close()
            if not isinstance(code, str):
                code = None
            diagnostics = {'availability': {'status': 'unavailable', **timing}}
            if code in UPSTREAM_CODES:
                diagnostics['availability']['cause'] = code
            reason = {'WORK_BUDGET_EXHAUSTED': 'hand_screen_work_budget_exhausted',
                      'CONVERSION_TIMEOUT': 'hand_screen_conversion_timeout',
                      'BLENDER_UNAVAILABLE': 'hand_screen_blender_unavailable'}.get(code, 'hand_screen_http_error')
            return unknown(reason, upstream_status=status, **diagnostics)
        except TimeoutError:
            return unknown('hand_screen_request_timeout')
        except URLError as exc:
            return unknown('hand_screen_request_timeout' if isinstance(exc.reason, TimeoutError)
                           else 'hand_screen_network_error')
        except (ValueError, TypeError, KeyError, UnicodeError):
            return unknown('hand_screen_invalid_receipt')
        except Exception:
            # Unexpected errors still fail closed and never expose exception text.
            return unknown('hand_screen_request_or_integrity_failed')


class HandSkinGate:
    def __init__(self, client, bvh_root, hashes, *, budget_seconds=45., max_checks=10,
                 concurrency=1):
        if not math.isfinite(budget_seconds) or not 0 < budget_seconds <= 60:
            raise ValueError('hand screen budget must be finite and in (0,60]')
        self.client, self.root, self.hashes = client, Path(bvh_root), dict(hashes)
        self.budget_seconds, self.max_checks = budget_seconds, max_checks
        if type(concurrency) is not int or concurrency not in (1, 2):
            raise ValueError('hand screen concurrency must be 1 or 2')
        self.concurrency = concurrency
        # Bound aggregate checks across people/requests sharing this engine.
        if concurrency == 2:
            from .hand_skin_parallel import ScreenSlots
            self.parallel_slots = ScreenSlots()

    def session(self, hands=None):
        if self.concurrency == 2:
            from .hand_skin_parallel import ParallelSkinSession
            return ParallelSkinSession(self, hands)
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
        # Optional mirror priority is request/session-local, never shared between analyses.
        self.client = copy(gate.client) if isinstance(gate.client, HandSkinClient) else gate.client
        self.hands = hands
        self.spent_seconds = 0.
        self.checks = 0
        self.failed = False
        self.cache = {}
        self.rejected, self.unverified = [], []
        self.events = []

    @contextmanager
    def optional_phase(self):
        previous = getattr(self.client, '_screen_optional', False)
        self.client._screen_optional = True
        try:
            yield
        finally:
            self.client._screen_optional = previous

    def check(self, text, *, timeout_seconds=None):
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if key in self.cache:
            return self.cache[key]
        remaining = self.gate.budget_seconds-self.spent_seconds
        if timeout_seconds is not None:
            remaining = min(remaining, timeout_seconds)
        if self.failed:
            return unknown('hand_screen_upstream_unavailable')
        if remaining <= 0:
            return unknown('hand_screen_budget_exhausted')
        if self.checks >= self.gate.max_checks:
            return unknown('hand_screen_check_limit_exhausted')
        self.checks += 1
        started = monotonic()
        try:
            result = self.client.check(text, self.hands, timeout=min(remaining, 15.))
            validate_screen(result)
        except Exception:
            result = unknown('hand_screen_failed')
        self.spent_seconds += monotonic()-started
        self.failed = result['status'] == 'unavailable'
        self.cache[key] = result
        timing = result.get('availability') or {}
        event = dict(check=self.checks, status=result['status'],
                     reason=result.get('reason'), elapsed_ms=round((monotonic()-started)*1000, 3),
                     remaining_before_ms=round(remaining*1000, 3),
                     call_budget_ms=round(min(remaining, 15.)*1000, 3),
                     optional=getattr(self.client, '_screen_optional', False),
                     **{key: timing[key] for key in ('queue_wait_ms', 'execution_ms', 'phase', 'cause', 'conversion_id')
                        if key in timing})
        self.events.append(event)
        from .logging_setup import log_info
        log_info('hand_skin_screen_check', **event)
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
                    rejected=self.rejected, unverified=self.unverified,
                    checks_diagnostics=list(self.events))


def build_hand_skin_gate(camera, bvh_root):
    from .config import CFG
    characters = json.loads((Path(__file__).resolve().parents[1]/'config/characters.example.json').read_text())
    character_id = CFG.thumbnail_character_id
    digest = characters['characters'].get(character_id, {}).get('sha256')
    client = HandSkinClient(CFG.thumbnail_converter_url, character_id, digest)
    return HandSkinGate(client, bvh_root, camera.hashes,
                        budget_seconds=CFG.hand_skin_screen_budget_seconds,
                        concurrency=CFG.hand_skin_screen_concurrency)
