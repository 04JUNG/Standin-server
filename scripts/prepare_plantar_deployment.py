"""Set only the production Converter's optional sole-correction override."""
import argparse
import json
from pathlib import Path


def prepare(task, value):
    if value not in ('0', '1'):
        raise ValueError('choose plantar correction 0 or 1')
    if not task.get('family', '').startswith('StandinApp'):
        raise ValueError('expected production task')
    containers = [c for c in task['containerDefinitions'] if c['name'] == 'converter']
    if len(containers) != 1:
        raise ValueError('expected exactly one Converter')
    container = containers[0]
    environment = container.setdefault('environment', [])
    if value == '0' and any(e['name'] == 'CONVERTER_FORCE_EXACT_V324'
                           and e['value'].lower() in ('1', 'true') for e in environment):
        raise ValueError('cannot combine force_exact_v324 and plantar disabled')
    key = 'CONVERTER_PLANTAR_CORRECTION_ENABLED'
    environment[:] = [e for e in environment if e['name'] != key]
    environment.append(dict(name=key, value=value))
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('--enabled', choices=['0', '1'], required=True)
    args = parser.parse_args()
    task = prepare(json.loads(args.task.read_text()), args.enabled)
    args.task.write_text(json.dumps(task, indent=2)+'\n')
    print('Production Converter plantar override prepared; no AWS operation performed')
