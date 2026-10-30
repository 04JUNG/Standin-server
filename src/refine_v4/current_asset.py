"""Resolve the current confirmed pose independently of a historical baseline.

A failed next refinement returns this current file, never silently an older
search result. Approval is visual/user provenance, not a skin certification.
"""
from pathlib import Path
import hashlib


def current_source(request, root):
    value = request.get('current_asset')
    if value is None:
        path = Path(root)/request['source_fbx']
        expected = request['source_sha256']
    else:
        if value.get('confirmation') != 'user_confirmed' or not value.get('evidence'):
            raise ValueError('attributed current-pose confirmation required')
        path = Path(root)/value['fbx']
        expected = value['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('current pose fingerprint mismatch')
    return path
