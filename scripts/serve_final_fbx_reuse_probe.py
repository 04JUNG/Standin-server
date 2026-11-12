"""Local-only HTTP probe for precomputed final-byte reuse. Never deploy this app.

Use the same environment as the offline batch; --forbid-conversion makes any
unexpected runtime Blender conversion fail the proof. PNG rendering stays separate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--library-sha256')
    parser.add_argument('--run-log', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18971)
    parser.add_argument('--forbid-conversion', action='store_true')
    args = parser.parse_args()
    if args.env_file:
        os.environ.update(json.loads(args.env_file.read_text()))
    if args.library_sha256:
        os.environ['CONVERTER_FINAL_LIBRARY_SHA256'] = args.library_sha256
    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    from converter_api.app import create_app
    from converter_api.runner import BlenderRunner
    import uvicorn

    class ProbeRunner(BlenderRunner):
        def _convert_one(self, **options):
            with args.run_log.open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), character_id=options['character_id'])) + '\n')
            if args.forbid_conversion:
                raise RuntimeError('unexpected runtime conversion during prepared-asset proof')
            return super()._convert_one(**options)

    uvicorn.run(create_app(runner=ProbeRunner()), host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
