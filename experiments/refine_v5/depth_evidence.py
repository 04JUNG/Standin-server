"""Bind reviewed relative-depth observations to a fixed B0 and its cameras."""

import argparse
from pathlib import Path

import numpy as np

KINDS = {
    'body_behind', 'camera_farther', 'sole_clearance', 'waist_side_contact', 'head_behind',
}
ARM = ('RightArm', 'RightForeArm', 'RightHand')
LIFTED_LEG = ('LeftUpLeg', 'LeftLeg', 'LeftFoot')



def has_cue(case, kind):
    return any(cue['kind'] == kind for cue in case['depth_evidence']['cues'])


def cue_value(kind, subject, reference, body_forward, camera_farther, torso):
    """Return signed distance in torso lengths; increasing values satisfy the cue."""
    delta = np.asarray(subject, float) - np.asarray(reference, float)
    if kind in ('body_behind', 'waist_side_contact', 'head_behind'):
        return float(-delta @ body_forward / torso)
    if kind == 'camera_farther':
        return float(delta @ camera_farther / torso)
    if kind == 'sole_clearance':
        return float(delta[2] / torso)
    raise ValueError('unknown depth cue')


def cue_minimum(cue):
    return -cue['margin'] if cue['kind'] == 'waist_side_contact' else cue['margin']


def bind(run, previous, output):
    from experiments.refine_v5.ab_common import load_json, save_json
    from experiments.refine_v5.learning_data import digest, frame

    repo = Path(__file__).resolve().parents[2]
    review = load_json(repo / 'evaluation/refine_v5/depth_review_20261108.json')
    old = {r['case_id']: r for r in load_json(run / 'manifest.json')['cases']}
    poses = {(r['case_id'], r['method']): r for r in load_json(run / 'fit-results.json')}
    seeds = {r['case_id']: r for r in load_json(previous / 'all-seeds/seeds.json')}
    cases = []
    for case_id, observed in review['cases'].items():
        case = old[case_id]
        if digest(case['source']) != case['source_sha256'] or digest(case['image']) != case['image_sha256']:
            raise ValueError('stale B0 or rough')
        if set(observed) - {'movable', 'cues', 'locked_support'}:
            raise ValueError('unexpected evidence keys')
        chain = tuple(observed['movable'])
        if chain not in (ARM, LIFTED_LEG):
            raise ValueError('unsupported editable chain')
        for cue in observed['cues']:
            if set(cue) != {'kind', 'subject', 'reference', 'margin'} or cue['kind'] not in KINDS:
                raise ValueError('unsupported depth cue')
            if not 0 < cue['margin'] < .5 or not np.isfinite(cue['margin']):
                raise ValueError('invalid solver margin')
        _, _, torso, basis = frame(case['bones'])
        camera = poses[case_id, 'perspective']['camera']
        case = dict(case, depth_evidence=observed, depth_authority=review['authority'],
                    body_forward=basis[:, 2].tolist(), torso_world=torso, camera=camera)
        if chain == LIFTED_LEG:
            if observed.get('locked_support') != 'Right':
                raise ValueError('explicit right support is required')
            seed = seeds[case_id]
            ids = [11, 13, 15]
            if min(seed['scores'][i] for i in ids) < .3:
                raise ValueError('lifted leg anchors not observed')
            case['lifted_leg_targets_px'] = [seed['keypoints'][i] for i in ids]
        elif 'locked_support' in observed:
            raise ValueError('support lock applies only to the lifted-leg trial')
        cases.append(case)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'manifest.json', dict(
        schema=review['version'], parent_manifest_sha256=digest(run / 'manifest.json'),
        depth_signs=dict(body='positive anterior', camera='positive farther', height='Blender world Z up'),
        margin_units=review['margin_units'], cases=cases,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'previous', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    bind(args.run, args.previous, args.out)


if __name__ == '__main__':
    main()
