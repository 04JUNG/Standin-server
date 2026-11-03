"""Keep staging's inline Python initializer on the image being deployed."""
import argparse
import json
from pathlib import Path
import re


def prepare(task, image):
    if not re.fullmatch(r'[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/standin/inference:[a-f0-9]{40}',image):
        raise ValueError('immutable inference image required')
    for container in task['containerDefinitions']:
        if container['name']=='body-assets-init':
            if container.get('command',[])[:2] != ['python','-c']:
                raise ValueError('unknown initializer contract; preserve rather than guess')
            container['image']=image
    return task


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task'); p.add_argument('--image',required=True)
    a=p.parse_args(); path=Path(a.task)
    path.write_text(json.dumps(prepare(json.loads(path.read_text()),a.image),indent=2)+'\n')
