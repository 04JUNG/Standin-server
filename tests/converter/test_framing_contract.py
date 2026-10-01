"""Fail-closed framing lineage; Blender mesh QA is a separate real-runtime probe."""
from dataclasses import replace
import base64
import hashlib
import json

import pytest

from converter.framing import FRAMING_VERSION, retained_roles
from converter.worker import JobValidationError, load_job
from tests.converter.test_worker_contract import _job
from tests.converter.test_api_contract import FakeRunner, _client, VALID_BVH


class FramedRunner(FakeRunner):
    corrupt = None

    def convert(self, **kwargs):
        result = super().convert(**kwargs)
        preview = b"\x89PNG\r\n\x1a\nfixture"
        report = {**result.report, "output_scope": kwargs["output_scope"],
                  "framing_version": FRAMING_VERSION, "preview_view": kwargs["preview_view"],
                  "preview_sha256": hashlib.sha256(preview).hexdigest(), "preview_size": len(preview)}
        if self.corrupt:
            report[self.corrupt] = "wrong"
        return replace(result, preview=preview, report=report)


def post(client, **fields):
    return client.post('/convert-framed', files={"bvh": ("final.bvh", VALID_BVH, "application/octet-stream")},
                       data={"output_scope": "half", "preview_view": "front",
                             "expected_bvh_sha256": hashlib.sha256(VALID_BVH).hexdigest(), **fields})


@pytest.mark.parametrize('scope', ['full', 'half', 'bust', 'head'])
def test_framed_pair_hashes_and_requested_scope(tmp_path, scope):
    runner = FramedRunner()
    response = post(_client(tmp_path, runner=runner), output_scope=scope)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['output_scope'] == scope
    assert value['framing_version'] == FRAMING_VERSION
    for key in ('fbx', 'preview'):
        assert hashlib.sha256(base64.b64decode(value[key+'_base64'])).hexdigest() == value[key+'_sha256']
    assert runner.calls[0]['mirror'] is False


@pytest.mark.parametrize('field', ['output_scope', 'framing_version', 'preview_sha256', 'preview_view', 'preview_size'])
def test_framed_integrity_fails_closed(tmp_path, field):
    runner = FramedRunner()
    runner.corrupt = field
    assert post(_client(tmp_path, runner=runner)).status_code == 500


@pytest.mark.parametrize('fields', [{'output_scope': 'auto'}, {'preview_view': '../x'}, {'expected_bvh_sha256': '0'*64}])
def test_framed_invalid_request_never_calls_blender(tmp_path, fields):
    runner = FramedRunner()
    assert post(_client(tmp_path, runner=runner), **fields).status_code in (400, 409)
    assert runner.calls == []


def test_worker_preserves_exact_solver_switch(tmp_path):
    path, value = _job(tmp_path)
    value.update(output_scope='half', preview_view='front', force_exact_v324=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(JobValidationError):
        load_job(path)


def test_anatomical_regions_keep_hands_only_in_half():
    assert 'hand.L' in retained_roles('half')
    assert 'hand.L' not in retained_roles('bust')
    assert retained_roles('head') == {'head'}
    assert 'upperleg.L' not in retained_roles('half')
