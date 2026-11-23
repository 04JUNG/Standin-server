import copy
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from converter.plantar_mode import enabled, selected
from scripts.prepare_plantar_deployment import prepare


def entry_without_blender(monkeypatch):
    import converter
    entry = ModuleType('converter.convert')
    path = Path(__file__).resolve().parents[2]/'converter/convert.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'convert')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), entry.__dict__)
    entry._convert_v321_compat = lambda **kw: None
    monkeypatch.setattr(converter, 'convert', entry, raising=False)
    monkeypatch.setitem(sys.modules, 'converter.convert', entry)
    return entry


def test_default_keeps_existing_behavior_and_unknown_value_is_rejected(monkeypatch):
    monkeypatch.delenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', raising=False)
    assert enabled()
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', 'disabled-ish')
    with pytest.raises(ValueError):
        enabled()


def test_off_selects_base_with_defaults_and_restores_entry_after_failure(monkeypatch):
    entry = entry_without_blender(monkeypatch)
    original = entry.convert
    calls = []
    sentinel = object()
    monkeypatch.setattr(entry, '_convert_v321_compat', lambda **kw: calls.append(kw) or sentinel)
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '0')
    with pytest.raises(RuntimeError):
        with selected() as policy:
            assert policy['mode'] == 'v321-base-only'
            assert entry.convert(bvh_path='source', character_fbx='body', out_path='output') is sentinel
            assert calls[0]['frame'] == 0 and calls[0]['mirror'] is False
            assert calls[0]['src_profile'] is None and 'force_exact_v324' not in calls[0]
            with pytest.raises(ValueError, match='conflicts'):
                entry.convert(bvh_path='source', character_fbx='body', out_path='output', force_exact_v324=True)
            raise RuntimeError('test cleanup')
    assert entry.convert is original


def test_worker_runs_all_existing_checks_and_records_mode(monkeypatch):
    from converter import worker
    entry_without_blender(monkeypatch)
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '0')
    expected = dict(ok=True, artifact_validation=dict(skin_verified=True))
    calls = []
    monkeypatch.setattr(worker, '_execute_job', lambda job: calls.append(job) or dict(expected))
    job = {'owned': 'fixture'}
    result = worker.run_job(job)
    assert calls == [job] and result['artifact_validation'] == expected['artifact_validation']
    assert result['plantar_correction']['enabled'] is False
    # The frozen manifest is still checked in the real _execute_job.
    assert worker._frozen_lineage()['solver_manifest_sha256']


def test_frozen_code_is_verified_before_loading_disabled_adapter(monkeypatch):
    from converter import worker, plantar_mode
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '0')
    monkeypatch.setattr(worker, '_frozen_lineage', lambda: (_ for _ in ()).throw(worker.FrozenLineageError('changed')))
    monkeypatch.setattr(plantar_mode, 'selected', lambda: (_ for _ in ()).throw(AssertionError('must not load')))
    with pytest.raises(worker.FrozenLineageError, match='changed'):
        worker.run_job({})


def test_mode_changes_snapshot_and_preview_cache_identity(monkeypatch):
    from converter.interactive_adapter import runtime_revision
    from converter_api.body_preview import preview_contract
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '1')
    old = runtime_revision(), preview_contract()['preview_revision']
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '0')
    assert old[0] != runtime_revision()
    assert old[1] != preview_contract()['preview_revision']


def test_conflicting_emergency_mode_is_rejected_before_serving(monkeypatch):
    from converter_api.app import create_app
    monkeypatch.setenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '0')
    with pytest.raises(ValueError, match='conflicts'):
        create_app(registry=SimpleNamespace(default_character_id='fixture'),
                   runner=SimpleNamespace(settings=SimpleNamespace(force_exact_v324=True)))


def test_health_serialization_preserves_disabled_mode():
    from converter_api.schemas import HealthResponse
    value = HealthResponse(ok=True, solver_version='fixture', checks={}, plantar_correction_enabled=False)
    assert value.model_dump()['plantar_correction_enabled'] is False


def test_production_override_only_changes_converter_environment():
    task = dict(family='StandinAppConverterTask', cpu='2048', containerDefinitions=[
        dict(name='body-assets-init', environment=[dict(name='PIN', value='same')]),
        dict(name='converter', image='unchanged', secrets=[dict(name='TOKEN', valueFrom='same')],
             environment=[dict(name='OTHER', value='same')])])
    original = copy.deepcopy(task)
    prepare(task, '0')
    environment = task['containerDefinitions'][1]['environment']
    assert environment.pop() == dict(name='CONVERTER_PLANTAR_CORRECTION_ENABLED', value='0')
    assert task == original
    with pytest.raises(ValueError):
        prepare(dict(family='StandinStagingAppConverterTask'), '0')
    dockerfile = Path(__file__).resolve().parents[2]/'Dockerfile.converter'
    assert 'COPY converter/plantar_mode.py /app/converter/plantar_mode.py' in dockerfile.read_text()
