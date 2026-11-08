"""Run reviewed and detector-driven 029 inputs through the same offline solver."""

import argparse
from pathlib import Path

import numpy as np

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.part_single_fit import fit
from experiments.refine_v5.part_single_observe import detector_goals


FIXTURE = Path(__file__).resolve().parents[2] / 'evaluation/refine_v5/single_goal_029.json'


def prepare_plans(root, seeds, fixture=FIXTURE):
    case = load_json(root / 'input.json')
    review = load_json(fixture)
    if review['case_id'] != case['case_id']:
        raise ValueError('review fixture is for a different case')
    if digest(case['source']) != case['source_sha256'] or digest(case['image']) != case['image_sha256']:
        raise ValueError('source or rough changed')
    seed = next(row for row in load_json(seeds) if row['case_id'] == case['case_id'])
    automatic, evidence = detector_goals(case, seed, review['intent'])
    save_json(root / 'observation.json', dict(case_id=case['case_id'], intent_source='reviewed_request',
              intent=review['intent'], detector_evidence=evidence, automatic_goals=automatic))
    for name, goals in [('reviewed', review['reviewed']), ('detector', automatic)]:
        destination = root / name
        destination.mkdir(exist_ok=False)
        with np.load(root / 'base.npz') as mesh:
            plan = fit(case, mesh, goals)
        save_json(destination / 'plan.json', dict(input_sha256=digest(root / 'input.json'), result=plan))
        print('PART_SINGLE', name, None if plan is None else plan['after'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seeds', required=True, type=Path)
    parser.add_argument('--fixture', default=FIXTURE, type=Path)
    args = parser.parse_args()
    prepare_plans(args.root, args.seeds, args.fixture)


if __name__ == '__main__':
    main()
