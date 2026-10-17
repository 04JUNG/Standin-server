"""Export only non-secret routing config from a rendered ECS task for Docker.

No secret values/ARNs are exported. AWS credentials are passed by name from the
GitHub job to Docker, never written to this file. Do not log the resulting file.
"""
import argparse
import json
from pathlib import Path


ALLOWED = {
    'APP_ENV', 'DATA_DIR', 'DB_PATH', 'POSE_LIBRARY_URI', 'POSE_LIBRARY_VERSION',
    'POSE_BACKEND', 'POSE_MODEL_VARIANT', 'POSE_CANARY_STAGE', 'POSE_MODEL_URI',
    'POSE_MODELS_ROOT', 'POSE_MODEL_DOWNLOAD_BUDGET_SECONDS', 'POSE_STRICT',
    'ROUTING_FULLBODY_ENABLED', 'ROUTING_PREPARE_ASSETS', 'ROUTING_BVH_ROOT',
    'ROUTING_INCLUDE_STORED_COMPOSITIONS', 'POSE_CALIBRATION_FROM_MANIFEST',
    'SEMANTIC_285_TAGS_PATH', 'SEMANTIC_285_TAXONOMY_PATH',
    'SEMANTIC_285_BODY_INDEX_PATH', 'SEMANTIC_285_SHAPE_INDEX_PATH',
    'SEMANTIC_EMBEDDING_DIR', 'POSE_GENERATION_ENABLED', 'POSE_GENERATION_RIG',
    'POSE_GENERATION_TAXONOMY', 'REFINE_POSE_QUARANTINE_PATH',
}


def export(task):
    [container] = [c for c in task['containerDefinitions'] if c['name'] == 'inference']
    rows = []
    for item in container.get('environment', []):
        name, value = item['name'], item['value']
        if name in ALLOWED:
            if not isinstance(value, str) or '\n' in value or '\r' in value:
                raise ValueError('invalid preflight environment entry: ' + name)
            rows.append(name + '=' + value)
    return '\n'.join(rows) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.write_text(export(json.loads(args.task.read_text())), encoding='utf-8')
