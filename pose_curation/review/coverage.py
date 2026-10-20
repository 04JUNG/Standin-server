"""Local-only read routes for an explicitly generated rough coverage report."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..storage import contained_path, read_json


def coverage_router(curation: Path, assets: Path) -> APIRouter:
    router = APIRouter()

    def report():
        paths = sorted((curation / 'coverage').glob('*/report.json'))
        if not paths:
            raise HTTPException(404, '아직 생성된 러프 비교 보고서가 없습니다.')
        path = paths[-1]
        return path.parent, read_json(path)

    @router.get('/coverage')
    def page():
        return FileResponse(assets / 'coverage.html')

    @router.get('/api/coverage')
    def data():
        _, value = report()
        # Do not expose local paths, S3 keys or installation identifiers.
        return JSONResponse({k: v for k, v in value.items() if k != 'files'},
                            headers={'Cache-Control': 'no-store'})

    @router.get('/api/coverage/image/{identity}')
    def image(identity: str):
        root, value = report()
        relative = value.get('files', {}).get(identity)
        if relative is None:
            raise HTTPException(404, '비교 이미지가 없습니다.')
        try:
            path = contained_path(root, relative)
        except ValueError as exc:
            raise HTTPException(404, '잘못된 이미지 경로입니다.') from exc
        if not path.is_file():
            raise HTTPException(404, '이미지 파일이 없습니다.')
        return FileResponse(path, headers={'Cache-Control': 'no-store'})

    return router
