"""Keep the staging body initializer on the inference commit being deployed."""
import argparse
import json
from pathlib import Path
import re

IMAGE = re.compile(r'136889124221\.dkr\.ecr\.ap-northeast-2\.amazonaws\.com/standin/inference:[0-9a-f]{40}\Z')


def refresh(task, image):
    if not IMAGE.fullmatch(image) or not task.get('family', '').startswith('StandinStagingAppInferenceTask'):
        raise ValueError('only a pinned staging inference deployment may refresh assets')
    containers = task.get('containerDefinitions', [])
    initializers = [c for c in containers if c.get('name') == 'body-assets-init']
    if not initializers:
        return task
    if len(initializers) != 1 or len([c for c in containers if c.get('name') == 'inference']) != 1:
        raise ValueError('ambiguous staging containers')
    initializer = initializers[0]
    command = initializer.get('command', [])
    if not command or not isinstance(command[-1], str) or 'body_assets_ready' not in command[-1]:
        raise ValueError('unknown body initializer contract')
    initializer['image'] = image
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=Path)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    value = refresh(json.loads(args.task.read_text()), args.image)
    args.task.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    main()
