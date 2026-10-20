"""Low priority preview jobs accept only a bounded, hash-pinned final binary FBX."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from fastapi import File, Form, UploadFile
from fastapi.responses import Response
from converter.scene_contract import FBX_MAGIC, MAX_SCENE_BYTES
from converter_api.runner import BlenderUnavailableError, ConversionTimeoutError, RunnerError, WorkerIntegrityError


def render_final(runner, data, digest, aligned, scene_camera=False):
    runner.inspect_blender()
    with runner._process_slots, tempfile.TemporaryDirectory(prefix='standin-final-preview-', dir=runner.settings.temp_root) as directory:
        root = Path(directory).resolve()
        (root / 'final.fbx').write_bytes(data)
        job = json.dumps({'artifact_sha256': digest, 'aligned': aligned, 'scene_camera': scene_camera}, sort_keys=True).encode()
        path = root / 'job.json'
        path.write_bytes(job)
        command = [runner.settings.blender_binary, '--background', '--python-exit-code', '1',
                   '--python', str(Path(__file__).resolve().parents[1] / 'converter/final_preview_worker.py'),
                   '--', '--job', str(path)]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=(os.name == 'posix'), env=runner._environment())
        except OSError as exc:
            raise BlenderUnavailableError('preview Blender could not start') from exc
        try:
            runner._communicate(process, runner.settings.camera_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            runner._terminate_process_group(process, runner.settings.terminate_grace_seconds)
            raise ConversionTimeoutError('final preview timed out') from exc
        try:
            report_path, output = root / 'report.json', root / 'preview.png'
            if process.returncode or report_path.is_symlink() or output.is_symlink() or output.stat().st_size > 4 * 1024 * 1024:
                raise ValueError('preview worker failed')
            report = json.loads(report_path.read_text())
            preview = output.read_bytes()
            if (report.get('ok') is not True or report.get('artifact_sha256') != digest
                    or report.get('job_sha256') != hashlib.sha256(job).hexdigest()
                    or hashlib.sha256((root / 'final.fbx').read_bytes()).hexdigest() != digest
                    or report.get('preview_sha256') != hashlib.sha256(preview).hexdigest()
                    or report.get('preview_size') != len(preview)
                    or not preview.startswith(b'\x89PNG\r\n\x1a\n')):
                raise ValueError('preview lineage mismatch')
            return preview
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise WorkerIntegrityError('final preview integrity mismatch') from exc


def register_final_preview(app, runner):
    from converter_api.app import ApiProblem, SHA256_RE

    @app.post('/render-final-preview')
    def final_preview(fbx: UploadFile = File(...), artifact_sha256: str = Form(...), aligned: bool = Form(False), scene_camera: bool = Form(False)):
        data = fbx.file.read(MAX_SCENE_BYTES + 1)
        if len(data) > MAX_SCENE_BYTES:
            raise ApiProblem(413, 'INVALID_FBX', 'final FBX exceeds size limit')
        if not data.startswith(FBX_MAGIC):
            raise ApiProblem(422, 'INVALID_FBX', 'expected binary FBX')
        if not SHA256_RE.fullmatch(artifact_sha256) or hashlib.sha256(data).hexdigest() != artifact_sha256:
            raise ApiProblem(409, 'SOURCE_FBX_SHA256_MISMATCH', 'final FBX checksum mismatch')
        try:
            preview = render_final(runner, data, artifact_sha256, aligned, scene_camera)
        except BlenderUnavailableError as exc:
            raise ApiProblem(503, exc.code, 'preview unavailable') from exc
        except ConversionTimeoutError as exc:
            raise ApiProblem(504, exc.code, 'preview timed out') from exc
        except RunnerError as exc:
            raise ApiProblem(500, exc.code, 'preview failed') from exc
        return Response(preview, media_type='image/png', headers={
            'Cache-Control': 'private, no-store', 'X-Standin-Artifact-SHA256': artifact_sha256,
            'X-Standin-Preview-Artifact-SHA256': hashlib.sha256(preview).hexdigest(),
        })
