"""Freeze rough observations separately from the legacy depth cues and the GT."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest


SIDES = {'Left', 'Right'}
RESOLVED = {'reviewed', 'confirmed'}
STATUSES = RESOLVED | {'unknown', 'ambiguous'}
REVIEW_FIELDS = {'side', 'side_status', 'support', 'support_status', 'contact', 'head'}
DEFAULT_REVIEW = Path(__file__).resolve().parents[2] / 'evaluation/refine_v5/part_observations_20261108.json'


def validate_review(review):
    if set(review) - REVIEW_FIELDS:
        raise ValueError('unsupported part observation')
    for name in ('side', 'support'):
        state = review.get(name + '_status', 'unknown')
        if state not in STATUSES:
            raise ValueError(f'unsupported {name} status')
        if (name in review) != (state in RESOLVED) or (name in review and review[name] not in SIDES):
            raise ValueError(f'inconsistent {name} observation')
    if 'contact' in review and review['contact'] not in {'waist', 'head'}:
        raise ValueError('unsupported contact surface')
    if 'head' in review:
        head = review['head']
        kind = head.get('source')
        if kind == 'reviewed' and set(head) == {'source', 'up_screen'}:
            if np.asarray(head['up_screen']).shape != (2,) or not np.isclose(
                np.linalg.norm(head['up_screen']), 1., atol=.002
            ):
                raise ValueError('invalid screen head axis')
        elif kind == 'synthetic' and set(head) == {'source', 'axes_body'}:
            axes = head['axes_body']
            if not axes or set(axes) - {'forward', 'up', 'neck'}:
                raise ValueError('unsupported head axes')
            for vector in axes.values():
                vector = np.asarray(vector, float)
                if vector.shape != (3,) or not np.isclose(np.linalg.norm(vector), 1., atol=.002):
                    raise ValueError('invalid head axis')
        else:
            raise ValueError('unsupported head observation')


def abstention_reason(case, review, leg):
    """Return an abstention reason before any image objective can move a limb."""
    if case['ownership_authority'] != 'offline_single_target_review':
        return 'unverified_owner'
    if review['side_status'] not in RESOLVED:
        return 'unresolved_side'
    if leg and review.get('support_status') != 'confirmed':
        return 'unresolved_support'
    if leg and review.get('support') == review.get('side'):
        return 'inconsistent_support'
    return None


def projected_evidence(points, pixel_scale, bone_lengths, camera_torso):
    """Observed bend and foreshortening proxies in the fixed rough camera."""
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if (lengths < 3.).any() or not np.isfinite(lengths).all():
        return dict(bend_cos=None, foreshortening=[None, None])
    cosine = float(segments[0] @ segments[1] / np.prod(lengths))
    ratios = lengths / (pixel_scale * bone_lengths / camera_torso)
    # A stretched sketch segment is not evidence for an impossible 3D ratio.
    return dict(bend_cos=float(np.clip(cosine, -1., 1.)),
                foreshortening=[float(r) if .04 <= r <= 1.05 else None for r in ratios])


def bind_case(case, review, camera):
    validate_review(review)
    limb = case['depth_evidence']
    chain = limb['movable']
    leg = 'locked_support' in limb
    bound = dict(side_status=review['side_status'], support_status=review.get('support_status', 'unknown'),
                 side=review.get('side'), support=review.get('support'), source=case['depth_authority'],
                 contact=None, head=review.get('head') if case['ownership_authority'] == 'offline_single_target_review' else None,
                 reason=abstention_reason(case, review, leg))
    if bound['reason'] is not None:
        return bound
    side = bound['side']
    expected = [side + suffix for suffix in (('UpLeg', 'Leg', 'Foot') if leg else ('Arm', 'ForeArm', 'Hand'))]
    if chain != expected or (leg and bound['support'] != limb['locked_support']):
        raise ValueError('review and legacy cue use different anatomical chains')
    if 'contact' in review:
        if [side, review['contact']] not in case['contact']:
            raise ValueError('contact has no verified subject and surface')
        bound['contact'] = review['contact']
    points = (np.asarray(case['lifted_leg_targets_px'], float) if leg
              else np.asarray(case['arms'][side], float) * case['dimensions'])
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError('invalid chain observations')
    heads = {bone['name']: np.asarray(bone['matrix'])[:3, 3] for bone in case['bones']}
    vectors = np.diff([heads[name] for name in chain], axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if (lengths < 1e-6).any():
        raise ValueError('degenerate B0 chain')
    # Depth magnitude is unknown. The B0 bend is a prior for the lifted leg,
    # distinct from the observed screen-space bend and explicit depth order.
    base_bend = float(vectors[0] @ vectors[1] / np.prod(lengths)) if leg else None
    bound.update(projected_evidence(points, camera['pixel_scale'], lengths, camera['torso_world']),
                 base_bend_cos=base_bend)
    return bound


def prepare_run(prior, review_file, out):
    manifest = load_json(prior / 'manifest.json')
    review = load_json(review_file)
    if set(review) != {'schema', 'cases'} or review['schema'] != 'v5-part-review-v1':
        raise ValueError('unexpected observation schema')
    cases = manifest['cases']
    if {case['case_id'] for case in cases} != set(review['cases']):
        raise ValueError('observation case inventory differs')
    for case in cases:
        if digest(case['source']) != case['source_sha256'] or digest(case['image']) != case['image_sha256']:
            raise ValueError('B0 or rough changed')
        # Compare the same camera and the same prebound legacy depth axes.
        case['camera'] = camera = case['cameras']['ortho_calibrated']
        case['part_goal'] = bind_case(case, review['cases'][case['case_id']], camera)
    out.mkdir(parents=True, exist_ok=False)
    for case in cases:
        name = f"{case['case_id']}-B0.npz"
        shutil.copyfile(prior / name, out / name)
    save_json(out / 'manifest.json', dict(schema='v5-part-target-pilot-v1', cases=cases,
              prior_manifest_sha256=digest(prior / 'manifest.json'), review_sha256=digest(review_file)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    prepare_run(args.prior, args.review, args.out)


if __name__ == '__main__':
    main()
