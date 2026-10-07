"""Selected body + ordered pose batch -> verified converter previews.

Stateless: BFF owns the current revision and must discard stale responses.
No user URL is fetched; library poses resolve through the server DB.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
from pathlib import Path
from threading import BoundedSemaphore
from typing import Literal
from urllib import error, request
from urllib.parse import quote, urlparse

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import Field

from api.body_handoff import BodyRenderRequest, HandoffModel
from api.body_models import BodyAssetRef, ProjectionView, Sha256
from src.config import CFG
from src.experimental.body_matching.catalog import load_catalog
from src.pose_quarantine import quarantine_record
from src.refine_thumbnail import _multipart

MAX_BVH_BYTES = 2_000_000
MAX_IMAGE_BYTES = 4_000_000
RENDER_SIZE = 256
# Each worker permits one batch at a time. Busy requests fail quickly for BFF retry.
_RENDER_SLOT = BoundedSemaphore(1)


class BodyPreview(HandoffModel):
    candidate_index: int
    pose_id: str
    view: ProjectionView
    source_pose_sha256: Sha256
    media_type: Literal['image/png'] = 'image/png'
    data: str = Field(description='base64 PNG of the selected FBX body')
    preview_sha256: Sha256
    width: Literal[256] = 256
    height: Literal[256] = 256


class BodyPreviewResult(HandoffModel):
    schema_version: Literal['body-preview.v1'] = 'body-preview.v1'
    request_id: str
    person_id: str
    selection_revision: int
    body: BodyAssetRef
    character_id: str
    catalog_sha256: Sha256
    retarget_version: str
    renderer_version: str
    status: Literal['ready'] = 'ready'
    previews: list[BodyPreview]


class ConverterBodyRenderer:
    def __init__(self, base_url: str, timeout: float = 60, opener=None):
        parsed = urlparse(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('invalid_body_converter_url')
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError('invalid_body_render_timeout')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.open = opener or request.urlopen

    def render(self, bvh: bytes, *, character_id: str, asset_sha256: str,
               view: str, retarget_version: str, renderer_version: str) -> bytes:
        body, content_type = _multipart(
            fields={'character_id': character_id, 'expected_character_sha256': asset_sha256,
                    'view': view, 'size': str(RENDER_SIZE), 'format': 'png'},
            file_field='bvh', filename='body-pose.bvh', file_bytes=bvh,
        )
        req = request.Request(self.base_url + '/render-thumbnail', data=body, method='POST',
                              headers={'Content-Type': content_type, 'Accept': 'image/png'})
        try:
            with self.open(req, timeout=self.timeout) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
                headers = response.headers
                if response.status != 200:
                    raise HTTPException(502, 'body_converter_failed')
        except error.HTTPError as exc:
            raise HTTPException(504 if exc.code == 504 else 502, 'body_converter_failed') from exc
        except TimeoutError as exc:
            raise HTTPException(504, 'body_converter_timeout') from exc
        except (error.URLError, OSError) as exc:
            raise HTTPException(502, 'body_converter_unavailable') from exc
        expected = {
            'X-Standin-Character-Id': character_id,
            'X-Standin-Character-SHA256': asset_sha256,
            'X-Standin-Source-BVH-SHA256': hashlib.sha256(bvh).hexdigest(),
            'X-Standin-Solver-Version': retarget_version,
            'X-Standin-Thumbnail-Renderer': renderer_version,
            'X-Standin-Thumbnail-View': view,
            'X-Standin-Thumbnail-SHA256': hashlib.sha256(data).hexdigest(),
        }
        if any(headers.get(key) != value for key, value in expected.items()):
            raise HTTPException(502, 'body_converter_identity_mismatch')
        if len(data) > MAX_IMAGE_BYTES or headers.get('Content-Type', '').split(';')[0] != 'image/png':
            raise HTTPException(502, 'invalid_body_preview')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'PNG' or image.size != (RENDER_SIZE, RENDER_SIZE):
                    raise ValueError('unexpected image')
                image.verify()
        except (OSError, ValueError) as exc:
            raise HTTPException(502, 'invalid_body_preview') from exc
        return data


def render_body_batch(batch: BodyRenderRequest, *, catalog_path, resolve_pose, renderer):
    try:
        catalog = load_catalog(catalog_path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(503, 'body_catalog_unavailable') from exc
    if catalog.sha256 != batch.catalog_sha256:
        raise HTTPException(409, 'body_catalog_changed')
    asset = next((a for a in catalog.eligible([p.pose_id for p in batch.poses])
                  if a['body_id'] == batch.body.body_id), None)
    if asset is None:
        raise HTTPException(409, 'body_not_eligible_for_poses')
    if any(asset.get(k) != v for k, v in batch.body.model_dump(include={
            'body_id', 'body_version', 'asset_sha256', 'rig_version', 'measurement_version'}).items()):
        raise HTTPException(409, 'body_asset_changed')
    character_id = asset.get('metadata', {}).get('converter_character_id')
    if not isinstance(character_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', character_id):
        raise HTTPException(409, 'body_converter_character_not_registered')
    # Validate the entire batch before the first converter call.
    sources = []
    for pose in batch.poses:
        if quarantine_record(pose.pose_id, CFG) is not None:
            raise HTTPException(409, 'pose_quarantined')
        if pose.kind == 'library':
            if pose.bvh_url != f'/pose/{quote(pose.pose_id, safe="")}/bvh':
                raise HTTPException(422, 'library_pose_url_mismatch')
            path = resolve_pose(pose.pose_id)
            if not path:
                raise HTTPException(404, 'body_pose_not_found')
            try:
                with Path(path).open('rb') as stream:
                    content = stream.read(MAX_BVH_BYTES + 1)
            except OSError as exc:
                raise HTTPException(404, 'body_pose_not_found') from exc
        else:
            content = pose.bvh.encode('utf-8')
        if len(content) > MAX_BVH_BYTES:
            raise HTTPException(413, 'body_pose_too_large')
        if hashlib.sha256(content).hexdigest() != pose.pose_sha256:
            raise HTTPException(409, 'body_pose_hash_mismatch')
        sources.append(content)
    previews = []
    for pose, content in zip(batch.poses, sources):
        png = renderer.render(content, character_id=character_id, asset_sha256=asset['asset_sha256'],
                              view=pose.view, retarget_version=batch.retarget_version,
                              renderer_version=batch.renderer_version)
        previews.append(BodyPreview(candidate_index=pose.candidate_index, pose_id=pose.pose_id,
                                    view=pose.view, source_pose_sha256=pose.pose_sha256,
                                    data=base64.b64encode(png).decode('ascii'),
                                    preview_sha256=hashlib.sha256(png).hexdigest()))
    return BodyPreviewResult(request_id=batch.request_id, person_id=batch.person_id,
                             selection_revision=batch.selection_revision, body=batch.body,
                             character_id=character_id, catalog_sha256=catalog.sha256,
                             retarget_version=batch.retarget_version, renderer_version=batch.renderer_version,
                             previews=previews)


def build_body_router(*, resolve_pose):
    router = APIRouter()

    @router.post('/body/render', response_model=BodyPreviewResult)
    def render_body(batch: BodyRenderRequest):
        url = os.getenv('BODY_RENDER_CONVERTER_URL', '')
        if not url:
            raise HTTPException(503, 'body_render_not_configured')
        try:
            renderer = ConverterBodyRenderer(url, float(os.getenv('BODY_RENDER_TIMEOUT_SECONDS', '60')))
        except ValueError as exc:
            raise HTTPException(503, 'body_render_configuration_invalid') from exc
        if not _RENDER_SLOT.acquire(blocking=False):
            raise HTTPException(503, 'body_renderer_busy', headers={'Retry-After': '2'})
        try:
            return render_body_batch(batch, catalog_path=CFG.body_catalog_path,
                                     resolve_pose=resolve_pose, renderer=renderer)
        finally:
            _RENDER_SLOT.release()

    return router


def preview_result_matches(current: BodyRenderRequest, result: BodyPreviewResult) -> bool:
    """BFF calls this against its CURRENT selection before replacing any cards."""
    return (
        result.request_id == current.request_id
        and result.person_id == current.person_id
        and result.selection_revision == current.selection_revision
        and result.body == current.body
        and result.catalog_sha256 == current.catalog_sha256
        and result.retarget_version == current.retarget_version
        and result.renderer_version == current.renderer_version
        and [(p.candidate_index, p.pose_id, p.view, p.source_pose_sha256) for p in result.previews]
            == [(p.candidate_index, p.pose_id, p.view, p.pose_sha256) for p in current.poses]
    )
