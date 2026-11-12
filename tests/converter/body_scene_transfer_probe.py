"""External benchmark preload; no serving import or output without explicit dir."""
import atexit
import functools
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
directory = os.getenv('STANDIN_SCENE_PROBE_DIR')
if directory:
    started = time.monotonic()
    records = {}
    io_records = []

    def wrap(module, name):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            begin = time.monotonic()
            ok = False
            try:
                value = original(*args, **kwargs)
                ok = True
                if name == 'run_job':
                    (Path(directory)/'worker-report.json').write_text(json.dumps(value))
                return value
            finally:
                records[name] = dict(seconds=time.monotonic()-begin, ok=ok)
        setattr(module, name, measured)

    from converter import worker, interactive_adapter
    wrap(worker, 'run_job')
    wrap(interactive_adapter, 'prepare_body')
    wrap(interactive_adapter, 'solve_from_body')

    def io_wrap(module, name, operation):
        original = getattr(module, name)
        @functools.wraps(original)
        def measured(path, *args, **kwargs):
            before = time.monotonic()
            ok = False
            try:
                value = original(path, *args, **kwargs)
                ok = True
                return value
            finally:
                file = Path(path)
                memory = file.name.endswith('.memory-rig.fbx')
                io_records.append(dict(operation=operation, memory=memory,
                    file=file.name, bytes=file.stat().st_size if file.exists() else None,
                    seconds=time.monotonic()-before, ok=ok))
        setattr(module, name, measured)

    from io_scene_fbx import parse_fbx, encode_bin
    if os.getenv('STANDIN_SCENE_FAULT_MEMORY_CODEC') == '1':
        codec_write = encode_bin.write
        def fault(file, *args, **kwargs):
            if str(file).endswith('.memory-rig.fbx'):
                raise RuntimeError('synthetic memory codec failure')
            return codec_write(file, *args, **kwargs)
        encode_bin.write = fault
    io_wrap(parse_fbx, 'parse', 'parse')
    io_wrap(encode_bin, 'write', 'write')

    def finish():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        value = dict(stages=records, io=io_records, child_wall_seconds=time.monotonic()-started,
            peak_rss_bytes=usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss*1024,
            user_seconds=usage.ru_utime, system_seconds=usage.ru_stime)
        (Path(directory)/('process-'+str(os.getpid())+'.json')).write_text(json.dumps(value, indent=2))
    atexit.register(finish)
