"""Operator switch for optional sole/ankle candidate search, after frozen QA.

The frozen entrypoint and its manifest remain byte-identical. A single-job
Blender child can select its existing V3.2.1 base path; body/shape/finger and
final-FBX validation still run in worker.run_job. Never install in the API.
"""
from contextlib import contextmanager
import inspect
import os


def enabled():
    value = os.getenv('CONVERTER_PLANTAR_CORRECTION_ENABLED', '1').strip().lower()
    if value not in ('0', '1', 'false', 'true'):
        raise ValueError('CONVERTER_PLANTAR_CORRECTION_ENABLED must be 0 or 1')
    return value in ('1', 'true')


@contextmanager
def selected():
    active = enabled()
    policy = dict(version='optional-plantar-correction-v1', enabled=active,
                  mode='v325-search' if active else 'v321-base-only')
    if active:
        yield policy
        return
    from converter import convert as entry
    original = entry.convert
    signature = inspect.signature(original)

    def base_only(**kwargs):
        arguments = signature.bind(**kwargs)
        arguments.apply_defaults()
        values = dict(arguments.arguments)
        if values.pop('force_exact_v324'):
            raise ValueError('plantar disabled conflicts with force_exact_v324')
        return entry._convert_v321_compat(**values)

    entry.convert = base_only
    try:
        yield policy
    finally:
        entry.convert = original
