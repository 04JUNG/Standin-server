"""Automatic rough-only observations for the original-BVH validation cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.part_vlm_observe import SYSTEM, enum, object_schema, observe, schema


EXTRA = """
Before choosing anatomical leg labels, trace the leg from each hip, not from
the screen position of the foot. Independently report the screen side of the
hip leading to a clearly raised knee and the hip leading to a clear support foot.
Use image_left or image_right, or unknown if crossed, ambiguous, or neither
is clearly unique. A panel line or background line is not a support surface.
Report torso_view. Use facing_camera for front or front three-quarter,
facing_away for back or rear three-quarter, profile for side, or unknown.

Report head direction relative to the image camera, independently of the torso.
For yaw use left_profile, left_three_quarter, frontal, right_three_quarter,
right_profile or unknown. Left/right refer to the direction the face points
on the image, not the person's anatomy. For pitch use down, slightly_down,
level, slightly_up, up or unknown. Use level only with visible face evidence.
Report in-plane head lean separately as left, upright, right, or unknown.
For a small, hidden, ambiguous or rear-facing head use unknown on unresolved
axes. Do not infer the head solely from the neck or overall body gesture.
"""


def observation_schema():
    properties = schema()['properties']
    screen_side = enum(('image_left', 'image_right', 'unknown'))
    properties.update(
        torso_view=enum(('facing_camera', 'facing_away', 'profile', 'unknown')),
        raised_hip_screen=screen_side, support_hip_screen=screen_side,
        head=object_schema(dict(
            yaw=enum(('left_profile', 'left_three_quarter', 'frontal', 'right_three_quarter', 'right_profile', 'unknown')),
            pitch=enum(('down', 'slightly_down', 'level', 'slightly_up', 'up', 'unknown')),
            lean=enum(('left', 'upright', 'right', 'unknown')),
            evidence={'type': 'string'},
        )),
    )
    return object_schema(properties)


def setup(p1, out):
    manifest_path = p1 / 'manifest.json'
    cohort = load_json(p1 / 'p1-cohort.json')
    if digest(manifest_path) != cohort['manifest_sha256']:
        raise ValueError('P0 manifest changed')
    approved = set(cohort['case_ids'])
    rows = [row for row in load_json(manifest_path)['rows'] if row['case_id'] in approved]
    if len(rows) != len(approved) or any(row['split'] not in ('development', 'validation') for row in rows):
        raise ValueError('invalid training or validation split')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'observations').mkdir(exist_ok=True)
    save_json(out / 'cohort.json', dict(p1_manifest_sha256=cohort['manifest_sha256'],
        development=[r['case_id'] for r in rows if r['split'] == 'development'],
        validation=[r['case_id'] for r in rows if r['split'] == 'validation'],
        test_split_used=False))
    return [row for row in rows if row['split'] == 'validation']


def run(p1, out, env_file):
    rows = setup(p1, out)
    config = dotenv_values(env_file)
    key = os.environ.get('OPENAI_API_KEY') or config.get('OPENAI_API_KEY')
    model = os.environ.get('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'gpt-5-mini'
    if not key:
        raise ValueError('OpenAI credential unavailable')
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = []
        for row in rows:
            cid = row['case_id']
            path = p1 / row['rough']
            with Image.open(path) as image:
                size = list(image.size)
            item = dict(case_id=cid, image=str(path), image_sha256=row['rough_sha256'], dimensions=size)
            jobs.append(pool.submit(observe, item, out / 'observations' / f'{cid}.json', model, key,
                                    instructions=SYSTEM + EXTRA, response_schema=observation_schema))
        for job in jobs:
            job.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'out', 'env-file'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run(args.p1, args.out, args.env_file)
