"""Final FBX readiness is independent of image generation and failures."""
from dataclasses import replace
import hashlib
import json
import pytest

from tests.converter.test_api_contract import FakeRunner, _client
from tests.converter.test_framing_contract import FramedRunner, post
from tests.converter.test_worker_contract import _job
from converter.worker import load_job
from converter.scene_contract import FBX_MAGIC


class ReadyRunner(FramedRunner):
    def convert(self, **kwargs):
        result = super().convert(**kwargs)
        data = FBX_MAGIC + b'final-fixture'
        digest = hashlib.sha256(data).hexdigest()
        return replace(result, artifact=data, artifact_sha256=digest, report={**result.report,
            'artifact_validation': {'version':'final-fbx-v1', 'artifact_sha256':digest, 'scope':kwargs['output_scope'],
                'bones':22, 'meshes':1, 'vertices':8, 'polygons':6, 'skin_verified':True}})


@pytest.mark.parametrize('scope', ['full', 'half', 'bust', 'shoulder_head', 'head'])
def test_ready_has_verified_fbx_and_never_requests_visual_work(tmp_path, scope):
    runner = ReadyRunner()
    response = post(_client(tmp_path, runner=runner), preview_format='none', output_scope=scope)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['artifact_contract'] == 'verified-fbx-v1'
    assert 'preview_base64' not in value and 'preview_sha256' not in value
    assert runner.calls[0]['preview_view'] is None
    assert not runner.calls[0].get('preview_model')
    assert value['output_scope'] == scope


def test_ready_still_rejects_invalid_fbx(tmp_path):
    runner = FramedRunner()
    assert post(_client(tmp_path, runner=runner), preview_format='none').status_code == 500


def test_aligned_artifact_job_without_preview_is_valid(tmp_path):
    path, job = _job(tmp_path)
    job.update(camera_rotation=[[1,0,0],[0,1,0],[0,0,1]], preview_view=None)
    path.write_text(json.dumps(job))
    assert load_job(path)['preview_view'] is None


def test_final_preview_rejects_forged_fbx_before_worker(tmp_path):
    runner = FakeRunner()
    client = _client(tmp_path, runner=runner)
    data = FBX_MAGIC + b'final-fixture'
    response = client.post('/render-final-preview', files={'fbx': ('final.fbx', data)},
                           data={'artifact_sha256': 'a'*64})
    assert response.status_code == 409
    assert runner.calls == []


def test_final_preview_uses_exact_persisted_bytes_and_hash(tmp_path, monkeypatch):
    data = FBX_MAGIC + b'final-fixture'
    digest = hashlib.sha256(data).hexdigest()
    png = b'\x89PNG\r\n\x1a\nfixture'
    calls = []
    def render(runner, received, sha, aligned, scene_camera=False):
        calls.append((received, sha, aligned))
        return png
    monkeypatch.setattr('converter_api.final_preview.render_final', render)
    client = _client(tmp_path, runner=FakeRunner())
    response = client.post('/render-final-preview', files={'fbx': ('final.fbx', data)},
                           data={'artifact_sha256': digest, 'aligned': 'true'})
    assert response.status_code == 200
    assert calls == [(data, digest, True)]
    assert response.headers['X-Standin-Artifact-SHA256'] == digest
    assert response.content == png


def test_artifact_ready_requires_structural_import_receipt(tmp_path):
    class MissingValidation(ReadyRunner):
        def convert(self, **kwargs):
            result = super().convert(**kwargs)
            report = dict(result.report); report.pop('artifact_validation')
            return replace(result, report=report)
    assert post(_client(tmp_path, runner=MissingValidation()), preview_format='none').status_code == 500


@pytest.mark.parametrize('change', [{'skin_verified':False}, {'bones':0}, {'artifact_sha256':'a'*64}, {'scope':'head'}])
def test_structural_receipt_fails_closed(change):
    from converter.final_artifact import valid_receipt
    value = {'version':'final-fbx-v1', 'artifact_sha256':'b'*64, 'scope':'half',
             'bones':22, 'meshes':1, 'vertices':8, 'polygons':6, 'skin_verified':True}
    assert valid_receipt(value, 'b'*64, 'half')
    assert not valid_receipt({**value, **change}, 'b'*64, 'half')
