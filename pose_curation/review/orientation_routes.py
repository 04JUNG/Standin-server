"""Local review-only orientation preview/export contract."""
from typing import Annotated, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..orientation import Orientation


class ExportSpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    scope: Literal['full', 'half', 'bust', 'head'] = 'full'
    yaw: float = Field(0, ge=-180, le=180)
    pitch: float = Field(0, ge=-90, le=90)
    roll: float = Field(0, ge=-180, le=180)
    content_hash: str = Field(pattern=r'^[a-f0-9]{64}$')

    def orientation(self):
        return Orientation(self.yaw, self.pitch, self.roll)


def orientation_router(previews, get_pose, *, base_path='/api/poses', scopes=None):
    router = APIRouter()

    def current_pose(key, spec):
        if scopes is not None and spec.scope not in scopes:
            raise HTTPException(422, '이 모델에서 지원하지 않는 출력 범위입니다.')
        pose = get_pose(key)
        if pose.content_hash != spec.content_hash:
            raise HTTPException(409, '포즈가 변경되었습니다. 다시 불러와 주세요.')
        return pose

    def status(key, spec, start=False):
        pose = current_pose(key, spec)
        try:
            result = previews.status(pose, spec.scope, start=start, orientation=spec.orientation())
            return {**result, 'orientation': spec.orientation().public(), 'content_hash': pose.content_hash}
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.post(base_path + '/{key}/oriented')
    def start(key: str, spec: ExportSpec, request: Request):
        origin = request.headers.get('origin')
        if request.headers.get('x-pose-review') != '1' or (origin and urlparse(origin).netloc != request.url.netloc):
            raise HTTPException(403, '검수 화면에서 생성해 주세요.')
        return status(key, spec, start=True)

    @router.get(base_path + '/{key}/oriented')
    def poll(key: str, spec: Annotated[ExportSpec, Depends()]):
        return status(key, spec)

    @router.get(base_path + '/{key}/oriented/{kind}')
    def download(key: str, kind: Literal['preview', 'fbx', 'settings'], v: str,
                 spec: Annotated[ExportSpec, Depends()]):
        pose = current_pose(key, spec)
        try:
            path = previews.artifact(pose, spec.scope, spec.orientation(), kind, v)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        if path is None:
            raise HTTPException(409, '현재 각도의 출력이 준비되지 않았습니다. 미리보기를 생성해 주세요.')
        media = {'preview': 'image/jpeg', 'settings': 'application/json'}.get(kind, 'application/octet-stream')
        filename = None if kind == 'preview' else f'{pose.pose_id}_{spec.scope}_y{spec.yaw:g}_p{spec.pitch:g}_r{spec.roll:g}{path.suffix}'
        return FileResponse(path, media_type=media, filename=filename,
                            headers={'Cache-Control': 'private, no-store', 'X-Orientation-Version': spec.orientation().public()['version']})

    return router
