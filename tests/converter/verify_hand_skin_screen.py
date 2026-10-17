"""Opt-in real Blender + local asset HTTP verification; never downloads assets.

python -m tests.converter.verify_hand_skin_screen --blender /path/Blender \
  --character /path/standin-master-v2.fbx --intersect-bvh /path/bad.bvh \
  --clear-bvh /path/clear.bvh
"""
import argparse
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from converter_api.app import create_app
from converter_api.registry import CharacterRegistry
from converter_api.runner import BlenderRunner, RunnerSettings
from src.hand_skin_screen import HandSkinClient


def main():
    parser = argparse.ArgumentParser()
    for key in ('blender', 'character', 'intersect-bvh', 'clear-bvh'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    registry = CharacterRegistry(root/'config/characters.example.json',
        environ={'STANDIN_MASTER_V2_URI': str(args.character.resolve())})
    runner = BlenderRunner(RunnerSettings(blender_binary=str(args.blender.resolve()),
        worker_path=root/'converter/worker.py', timeout_seconds=30))
    app = TestClient(create_app(registry=registry, runner=runner))

    class Response:
        def __init__(self, response):
            self.response, self.status = response, response.status_code
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, limit): return self.response.content[:limit]

    def opener(request, timeout):
        response = app.post('/screen-hand-contact', content=request.data,
                            headers={'Content-Type': request.get_header('Content-type')})
        if response.status_code != 200:
            print(response.text)
        return Response(response)

    client = HandSkinClient('http://converter', 'standin-master-v2',
        hashlib.sha256(args.character.read_bytes()).hexdigest(), opener=opener)
    for label, path, hands in (
        ('known_intersection', args.intersect_bvh, None),
        ('known_clear', args.clear_bvh, None),
        ('hand_baked', args.clear_bvh, {'left': 'relaxed', 'right': 'relaxed'}),
    ):
        result = client.check(path.read_text(), hands, 30)
        print(json.dumps(dict(case=label, receipt=result), ensure_ascii=False))
        assert result['status'] != 'unavailable', result
        if label == 'known_intersection': assert result['status'] == 'intersection', result
        if label == 'known_clear': assert result['status'] == 'clear_screen', result
    print('PASS real converter HTTP, hand baking, artifact/receipt hashes, opposite-hand screening')


if __name__ == '__main__':
    main()
