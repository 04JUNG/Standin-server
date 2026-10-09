"""Real FastAPI routes; fixture assets/runner. No Blender invocation or real FBX QA."""
import hashlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from converter_api.app import create_app
from converter_api.body_preview import preview_contract
from tests.converter.test_api_contract import _client
from tests.converter.test_framing_contract import FramedRunner, post
from tests.converter.test_preview_models import publish


def test_preview_contract_is_read_only_and_stable():
    app = create_app(registry=object(), runner=object())
    with TestClient(app) as client:
        value = client.get('/preview-contract').json()
        assert value == preview_contract()
        assert value == client.get('/preview-contract').json()
        assert len(value['preview_revision']) == 64
        assert value['model_version'] == 'posed-mesh-v1'


@pytest.mark.parametrize('expected,status', [('b'*64, 200), ('c'*64, 409), ('invalid', 400)])
def test_glb_checks_expected_body_before_read(tmp_path, monkeypatch, expected, status):
    publish(tmp_path)
    monkeypatch.setenv('POSE_PREVIEW_URI', str(tmp_path))
    app = create_app(registry=SimpleNamespace(metadata=lambda _: SimpleNamespace(sha256='b'*64)), runner=object())
    calls = []
    original = app.state.preview_models.get
    def get(*args):
        calls.append(args)
        return original(*args)
    app.state.preview_models.get = get
    with TestClient(app) as client:
        contract = client.get('/preview-contract').json()
        response = client.get('/pose-preview/'+'a'*64, params={
            'character_id': 'some-registered-body', 'expected_character_sha256': expected,
            'expected_preview_revision': contract['preview_revision'],
        })
        assert response.status_code == status, response.text
        if status == 200:
            assert response.headers['x-standin-character-sha256'] == expected
            assert response.headers['x-standin-preview-revision'] == contract['preview_revision']
            assert response.headers['x-standin-model-revision'] == contract['model_revision']
            assert calls == [('a'*64, 'b'*64)]
        else:
            assert not calls
            assert response.json()['error']['code'] in ('CHARACTER_SHA256_MISMATCH', 'INVALID_CHARACTER_SHA256')


@pytest.mark.parametrize('expected,status', [('0'*64, 409), ('bad', 400)])
def test_glb_old_runtime_rejected_before_registry_access(expected, status):
    with TestClient(create_app(registry=object(), runner=object())) as client:
        assert client.get('/pose-preview/'+'a'*64, params={'expected_preview_revision': expected}).status_code == status


@pytest.mark.parametrize('expected,status', [(hashlib.sha256(b'character').hexdigest(), 200), ('a'*64, 409), ('bad', 400)])
def test_framed_checks_expected_body_before_blender(tmp_path, expected, status):
    runner = FramedRunner()
    client = _client(tmp_path, runner=runner)
    contract = client.get('/preview-contract').json()
    response = post(client, expected_character_sha256=expected, expected_preview_revision=contract['preview_revision'])
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()['character_sha256'] == expected
        assert response.json()['preview_revision'] == contract['preview_revision']
        assert runner.calls[0]['character_sha256'] == expected
    else:
        assert not runner.calls


def test_framed_old_runtime_does_not_start_blender(tmp_path):
    runner = FramedRunner()
    response = post(_client(tmp_path, runner=runner), expected_preview_revision='0'*64)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'PREVIEW_REVISION_MISMATCH'
    assert not runner.calls


def test_runtime_contract_imports_from_exact_docker_copy_allowlist(tmp_path):
    """Catch a module that works in the repo but is absent from the deployed image."""
    import json
    from pathlib import Path
    import re
    import shutil
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    for source, target in re.findall(r'^COPY (\S+) /app/(\S+)\s*$', (root / 'Dockerfile.converter').read_text(), re.MULTILINE):
        src = root / source
        if src.is_file():
            dst = tmp_path / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    process = subprocess.run([sys.executable, '-c',
        'import json; from converter_api.body_preview import preview_contract; print(json.dumps(preview_contract()))'],
        cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == preview_contract()

@pytest.mark.parametrize("expected,status", [(hashlib.sha256(b"character").hexdigest(), 200), ("a"*64, 409)])
def test_final_model_keeps_body_and_runtime_contract(tmp_path, expected, status):
    from tests.converter.test_review_model import ModelRunner
    runner = ModelRunner()
    client = _client(tmp_path, runner=runner)
    contract = client.get("/preview-contract").json()
    response = post(client, preview_format="model", expected_character_sha256=expected,
                    expected_preview_revision=contract["preview_revision"])
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["preview_revision"] == contract["preview_revision"]
        assert response.json()["character_sha256"] == expected
        assert response.json()["preview_format"] == "model"
    else:
        assert not runner.calls
