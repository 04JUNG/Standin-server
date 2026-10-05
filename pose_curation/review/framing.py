"""Bounded, revision-bound preview jobs. No Blender imports in the web process."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from converter.framing import validate_scope
from ..rendering.batch import PROJECT, VIEWS, render_identity
from ..storage import read_json, sha256, write_json
from ..orientation import Orientation

ORIENTED_FILES = {'preview': 'oriented__front.jpg', 'fbx': 'oriented.fbx',
                  'settings': 'orientation.json'}


@dataclass(frozen=True)
class NativeReference:
    """Character rest pose; never added to the reviewed/captured pose catalog."""
    character: Path
    content_hash: str
    pose_id: str = 'standin_neutral_reference'
    group: str = 'reference'
    metadata: dict = field(default_factory=dict)


def source_file(pose):
    return pose.character if isinstance(pose, NativeReference) else pose.bvh


class FramedPreviews:
    def __init__(self, curation_dir: Path):
        self.root = curation_dir / 'framed-previews'
        self.character = curation_dir / 'characters/standin-master-v2.fbx'
        self.blender = PROJECT / 'data/tools/blender-5.2.0-windows-x64/blender.exe'
        self._identity = None
        self._lock = threading.RLock()
        self._jobs = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='framed-preview')

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def identity(self, pose, scope, orientation=None):
        validate_scope(scope)
        with self._lock:
            if self._identity is None:
                identity = render_identity(self.character, self.blender)
                identity['framed_worker'] = sha256(PROJECT / 'pose_curation/rendering/framed_worker.py')
                identity['orientation'] = sha256(PROJECT / 'pose_curation/orientation.py')
                identity['oriented_worker'] = sha256(PROJECT / 'pose_curation/rendering/oriented_worker.py')
                self._identity = identity
        value = {'renderer': self._identity, 'bvh': pose.content_hash,
                 'scope': scope, 'profile': pose.metadata.get('rig_profile', '100style' if pose.group == 'new' else None)}
        if isinstance(pose, NativeReference):
            if scope not in {'head', 'bust'}:
                raise ValueError('기본 얼굴 모델은 두상·흉상 범위만 지원합니다.')
            value['source_kind'] = 'native_reference'
            value['native_worker'] = sha256(PROJECT / 'pose_curation/rendering/neutral_worker.py')
        if orientation is not None:
            value['orientation'] = orientation.public()
            if pose.metadata.get('bust_body') is not None:
                if not isinstance(pose, NativeReference) or scope != 'bust':
                    raise ValueError('몸통 방향 분리는 기본 흉상에서만 지원합니다.')
                from ..head.bust import relative_rotation
                relative_rotation(orientation, Orientation(**pose.metadata['bust_body']))
                value['bust_body'] = pose.metadata['bust_body']
                value['bust_worker'] = sha256(PROJECT / 'pose_curation/rendering/bust_worker.py')
                value['bust_math'] = sha256(PROJECT / 'pose_curation/head/bust.py')
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _cached(self, fingerprint):
        directory = self.root / fingerprint
        try:
            result = read_json(directory / 'result.json')
            if not result.get('ok') or result.get('fingerprint') != fingerprint:
                return False
            if result.get('kind') == 'orientation':
                return set(result['files']) == set(ORIENTED_FILES) and all(
                    sha256(directory / filename) == result['files'][kind]['sha256']
                    for kind, filename in ORIENTED_FILES.items())
            if set(result['thumbnails']) != VIEWS:
                return False
            for view, record in result['thumbnails'].items():
                path = directory / f'preview__{view}.jpg'
                if sha256(path) != record['sha256']:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def status(self, pose, scope, *, start=False, orientation=None):
        fingerprint = self.identity(pose, scope, orientation)
        with self._lock:
            if self._cached(fingerprint):
                self._jobs.pop(fingerprint, None)
                return {'status': 'ready', 'scope': scope, 'version': fingerprint}
            future = self._jobs.get(fingerprint)
            error = None
            if future and future.done():
                if future.cancelled():
                    error = '미리보기 생성이 취소되었습니다. 다시 생성해 주세요.'
                else:
                    error = str(future.exception()) if future.exception() else '미리보기 파일 검증에 실패했습니다.'
                if start:
                    del self._jobs[fingerprint]
                    future = None
            if start and future is None:
                if sum(not item.done() for item in self._jobs.values()) >= 8:
                    raise RuntimeError('생성 대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.')
                self._jobs[fingerprint] = self._executor.submit(self._render, pose, scope, fingerprint, orientation)
                return {'status': 'rendering', 'scope': scope}
            if error:
                return {'status': 'failed', 'scope': scope, 'error': error}
            return {'status': 'rendering' if future else 'missing', 'scope': scope}

    def wait_ready(self, pose, scope, *, orientation=None, timeout=360):
        """CLI verification helper; HTTP callers use status polling instead."""
        state = self.status(pose, scope, start=True, orientation=orientation)
        if state['status'] != 'ready':
            with self._lock:
                future = self._jobs[self.identity(pose, scope, orientation)]
            future.result(timeout=timeout)
        return self.status(pose, scope, orientation=orientation)

    def image(self, pose, scope, view, version):
        fingerprint = self.identity(pose, scope)
        if version != fingerprint or not self._cached(fingerprint):
            return None
        return self.root / fingerprint / f'preview__{view}.jpg'

    def artifact(self, pose, scope, orientation, kind, version):
        fingerprint = self.identity(pose, scope, orientation)
        if kind not in ORIENTED_FILES or version != fingerprint or not self._cached(fingerprint):
            return None
        if sha256(source_file(pose)) != pose.content_hash:
            raise ValueError('원본 파일이 변경되었습니다. 다시 생성해 주세요.')
        return self.root / fingerprint / ORIENTED_FILES[kind]

    def _render(self, pose, scope, fingerprint, orientation=None):
        directory = self.root / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        job = {'directory': str(directory.resolve()), 'character': str(self.character.resolve()),
               'scope': scope, 'fingerprint': fingerprint, 'bvh': str(source_file(pose).resolve()),
               'bvh_sha256': pose.content_hash,
               'rig_profile': pose.metadata.get('rig_profile', '100style' if pose.group == 'new' else None)}
        worker = 'framed_worker.py'
        if isinstance(pose, NativeReference):
            job.pop('bvh')
            job.pop('bvh_sha256')
            job.update(character=str(pose.character.resolve()), character_sha256=pose.content_hash)
            worker = 'neutral_worker.py'
        if orientation is not None:
            # Same executor serializes base and oriented jobs: no nested future or deadlock.
            base_id = self.identity(pose, scope)
            base_dir = self.root / base_id
            base_fbx = base_dir / 'character.fbx'
            base_valid = self._cached(base_id)
            if base_valid:
                base_valid = base_fbx.is_file() and sha256(base_fbx) == read_json(base_dir / 'result.json').get('fbx_sha256')
            if not base_valid:
                self._render(pose, scope, base_id)
            if sha256(source_file(pose)) != pose.content_hash:
                raise ValueError('원본 파일이 변경되었습니다. 다시 불러와 주세요.')
            job.update(base_fbx=str(base_fbx.resolve()), base_fbx_sha256=sha256(base_fbx),
                       angles={name: getattr(orientation, name) for name in ('yaw', 'pitch', 'roll')},
                       metadata={'pose_id': pose.pose_id, 'bvh_sha256': pose.content_hash,
                                 'scope': scope, 'orientation': orientation.public(),
                                 'angle_export_format': 'fbx', 'bvh_policy': 'original_pose_only',
                                 'character': 'standin-master-v2'})
            if isinstance(pose, NativeReference):
                job['metadata'].pop('bvh_sha256')
                job['metadata'].pop('bvh_policy')
                job['metadata'].update(source_kind='native_reference', character_sha256=pose.content_hash,
                                       shoulder_pose='neutral_not_inferred', face_expression='neutral')
            worker = 'oriented_worker.py'
            if pose.metadata.get('bust_body') is not None:
                job['bust_body'] = pose.metadata['bust_body']
                worker = 'bust_worker.py'
        write_json(directory / 'job.json', job)
        command = [str(self.blender), '--background', '--factory-startup', '--threads', '2',
                   '--python-exit-code', '1', '--python',
                   str(PROJECT / 'pose_curation/rendering' / worker), '--',
                   str((directory / 'job.json').resolve())]
        with (directory / 'worker.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=180, check=True,
                           env=dict(os.environ, BLENDER_USER_RESOURCES=str((directory / 'blender-user').resolve())),
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result = read_json(directory / 'result.json')
        if not result.get('ok'):
            raise ValueError(result.get('error', '부분 미리보기 생성 실패'))
        if not self._cached(fingerprint):
            raise ValueError('생성된 미리보기의 무결성 검증에 실패했습니다.')
