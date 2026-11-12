"""A confirmed current result cannot be replaced by an old baseline on retry."""
from pathlib import Path
import hashlib,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.refine_v4.current_asset import current_source

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);old=root/'initial.fbx';current=root/'confirmed.fbx'
    old.write_bytes(b'old-search-result');current.write_bytes(b'accepted-deep-knee-fold')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    request={'source_fbx':old.name,'source_sha256':sha(old)}
    assert current_source(request,root)==old
    request['current_asset']={'fbx':current.name,'sha256':sha(current),
        'confirmation':'user_confirmed','evidence':'prior_user_shape_confirmation'}
    assert current_source(request,root)==current
    current.write_bytes(b'changed-under-the-same-file-name')
    try:current_source(request,root)
    except ValueError as e:assert 'fingerprint' in str(e)
    else:raise AssertionError('tampered current asset admitted')
print('PASS historical source, confirmed current pose, and hash refusal')
