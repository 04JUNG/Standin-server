"""Recovery/publication and identity contracts; geometry runs in real Blender."""
import json
from pathlib import Path

import pytest

from converter import body_scene_transfer as transfer
from converter import shape_bake


class Report:
    ok = True

    def as_dict(self):
        return {'ok': self.ok}


@pytest.mark.parametrize('value', [None, '0', 'true', 'invalid'])
def test_disabled_runs_only_legacy(monkeypatch, tmp_path, value):
    monkeypatch.delenv(transfer.FLAG, raising=False)
    if value is not None:
        monkeypatch.setenv(transfer.FLAG, value)
    report = Report()
    monkeypatch.setattr(shape_bake, 'convert_shaped', lambda **kw: report)
    monkeypatch.setattr(transfer, '_candidate', lambda **kw: pytest.fail('candidate with flag OFF'))
    assert transfer.convert_shaped(recipe={}, out_path=str(tmp_path/'out.fbx')) is report
    assert not list(tmp_path.iterdir())


def test_exact_parent_bypasses_experiment(monkeypatch, tmp_path):
    monkeypatch.setenv(transfer.FLAG, '1')
    report = Report()
    monkeypatch.setattr(shape_bake, 'convert_shaped', lambda **kw: report)
    monkeypatch.setattr(transfer, '_candidate', lambda **kw: pytest.fail('changed exact parent'))
    assert transfer.convert_shaped(recipe={}, out_path=str(tmp_path/'out.fbx'), force_exact_v324=True) is report


def test_candidate_publishes_only_after_completion(monkeypatch, tmp_path):
    monkeypatch.setenv(transfer.FLAG, '1')
    target = tmp_path/'out.fbx'

    def candidate(*, out_path, trace, checkpoint, **kw):
        assert not target.exists()
        checkpoint()
        Path(out_path).write_bytes(b'final candidate')
        Path(out_path).with_suffix('.shape.json').write_text('{}')
        trace.update(disk_full_mesh_imports_avoided=1, disk_full_mesh_exports_avoided=1)
        return Report()

    monkeypatch.setattr(transfer, '_candidate', candidate)
    monkeypatch.setattr(shape_bake, 'convert_shaped', lambda **kw: pytest.fail('unnecessary retry'))
    result = transfer.convert_shaped(recipe={}, out_path=str(target))
    assert target.read_bytes() == b'final candidate'
    assert result.as_dict()['scene_transfer']['status'] == 'transferred'
    assert result.trace['disk_full_mesh_imports_avoided'] == result.trace['disk_full_mesh_exports_avoided'] == 1
    assert not list(tmp_path.glob('body-scene-*'))


@pytest.mark.parametrize('error', [ValueError, TimeoutError, RuntimeError])
def test_failure_drops_candidate_and_retries_same_inputs(monkeypatch, tmp_path, error):
    monkeypatch.setenv(transfer.FLAG, '1')
    target = tmp_path/'out.fbx'
    recipe = {'asset_sha256': 'a'*64}

    def candidate(*, out_path, **kw):
        Path(out_path).write_bytes(b'unvalidated')
        raise error('private details must not be in trace')

    def legacy(**kw):
        assert kw == dict(recipe=recipe, out_path=str(target), bvh_path='source', mirror=True)
        assert not target.exists()
        assert not list(tmp_path.glob('body-scene-*'))
        target.write_bytes(b'legacy validated')
        return Report()

    monkeypatch.setattr(transfer, '_candidate', candidate)
    monkeypatch.setattr(shape_bake, 'convert_shaped', legacy)
    result = transfer.convert_shaped(recipe=recipe, out_path=str(target), bvh_path='source', mirror=True)
    assert target.read_bytes() == b'legacy validated'
    assert result.trace['status'] == 'legacy_recovery'
    assert result.trace['reason'] == error.__name__
    assert result.trace['recovery_seconds'] >= 0
    assert 'private details' not in json.dumps(result.trace)


def test_exhausted_budget_recovers_before_publication(monkeypatch, tmp_path):
    monkeypatch.setenv(transfer.FLAG, '1')
    monkeypatch.setenv('CONVERTER_EXPERIMENT_BODY_SCENE_SECONDS', '0.000000001')
    monkeypatch.setattr(transfer, '_candidate', lambda checkpoint, **kw: checkpoint())
    monkeypatch.setattr(shape_bake, 'convert_shaped', lambda **kw: Report())
    result = transfer.convert_shaped(recipe={}, out_path=str(tmp_path/'out.fbx'))
    assert result.trace['status'] == 'legacy_recovery'
    assert result.trace['reason'] == 'TimeoutError'


def test_recovery_failure_is_recorded_and_propagated(monkeypatch, tmp_path):
    monkeypatch.setenv(transfer.FLAG, '1')
    def fail(**kw):
        raise RuntimeError('failure')
    monkeypatch.setattr(transfer, '_candidate', fail)
    monkeypatch.setattr(shape_bake, 'convert_shaped', fail)
    with pytest.raises(RuntimeError):
        transfer.convert_shaped(recipe={}, out_path=str(tmp_path/'out.fbx'))
    trace = json.loads((tmp_path/'out.scene-transfer.json').read_text())
    assert trace['status'] == 'failed' and trace['recovery_reason'] == 'RuntimeError'


def test_legacy_rejection_cannot_be_reported_as_recovery_success(monkeypatch, tmp_path):
    monkeypatch.setenv(transfer.FLAG, '1')
    def fail(**kw):
        raise ValueError('candidate failed')
    rejected = Report()
    rejected.ok = False
    monkeypatch.setattr(transfer, '_candidate', fail)
    monkeypatch.setattr(shape_bake, 'convert_shaped', lambda **kw: rejected)
    result = transfer.convert_shaped(recipe={}, out_path=str(tmp_path/'out.fbx'))
    assert not result.ok and result.trace['status'] == 'failed'


def test_runtime_and_preview_identities_separate_flag_values(monkeypatch):
    from converter.interactive_adapter import runtime_revision
    from converter_api.body_preview import preview_contract
    monkeypatch.delenv(transfer.FLAG, raising=False)
    off = runtime_revision(), preview_contract()['preview_revision']
    monkeypatch.setenv(transfer.FLAG, '1')
    on = runtime_revision(), preview_contract()['preview_revision']
    assert all(a != b for a, b in zip(off, on))


def test_prepared_identity_sources_are_not_changed_by_experiment():
    # Prepared key stays on the prerequisite shape implementation, never adapter
    # or flag. The runtime snapshot identity above does include both.
    text = Path(transfer.__file__).with_name('prepared_body.py').read_text()
    assert 'body_scene_transfer.py' not in text
