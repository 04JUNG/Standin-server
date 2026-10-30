"""Generate offline failure/unknown reports using repository synthetic fixtures.

No user roughs, providers, Blender or installed apps are accessed. Generated
BVH/synthetic save bytes stay in the output directory. Public reports contain
only the evidence summaries. This is a regression demo, not a model benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-reports', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Reuse the known independent regression fixtures instead of private assets.
    from tests.test_eval_experiment import source_fixture
    from tests.test_smoke import _bvh_with_rotation
    from tests.test_eval_export_contracts import report_case
    from standin_eval.experiment_runner import FixtureAdapter, run_experiment
    from standin_eval.final_fbx_evaluator import seal_export_policy, evaluate_export_tasks
    from standin_eval.save_scenarios import run_save_scenarios
    from standin_eval.util import read_json, read_jsonl, write_json

    workspace = args.output / 'synthetic-source'
    workspace.mkdir()
    source, selection, base, _ = source_fixture(workspace)
    outcomes = {'F25': {'state':'abstained', 'refined':False,
                       'actual_backend':'none', 'bvh_bytes':base.read_bytes()}}
    reports = {}
    for name, outcome in [('unknown', outcomes)]:
        run = run_experiment(from_run=source, policy={'timeout_seconds':1},
            selections=[selection], adapters={'F25':FixtureAdapter(outcome)},
            output_root=args.output/'runs', run_id=name)
        reports['experiment-' + name] = read_json(run/'experiment_report.json')
        unknown_run = run
    changed = Path(_bvh_with_rotation(str(workspace), 'wrong-recovery.bvh', 'LeftArm', 12))
    outcomes['F25']['bvh_bytes'] = changed.read_bytes()
    run = run_experiment(from_run=source, policy={'timeout_seconds':1},
        selections=[selection], adapters={'F25':FixtureAdapter(outcomes)},
        output_root=args.output/'runs', run_id='failed-recovery')
    reports['experiment-failure'] = read_json(run/'experiment_report.json')
    export_workspace = args.output / 'synthetic-export'
    export_workspace.mkdir()
    task, policy, observation = report_case(export_workspace)
    observation['execution_kind'] = 'fixture'
    observation.pop('human_review')  # No human was evaluated in this demo.
    sealed = seal_export_policy([task, {**task,'task_id':'not-executed'}], policy)
    reports['export-unknown'] = evaluate_export_tasks(sealed, [observation])
    observation['artifacts'][-1]['sha256'] = 'b'*64
    reports['export-failure'] = evaluate_export_tasks(sealed, [observation])
    from standin_eval.pipeline_report import build_pipeline_report
    units = read_jsonl(unknown_run/'frozen_units.private.jsonl')
    linked_tasks = [{**task, 'task_id':'save-'+str(i),
        'selection_keys':[{'unit_id':unit['unit_id'],'arm':'F25'}]} for i,unit in enumerate(units)]
    reports['pipeline-unknown'] = build_pipeline_report(unknown_run, seal_export_policy(linked_tasks,policy), [])
    saves = run_save_scenarios(output_dir=args.output/'synthetic-saves')
    reports['save-contracts'] = {k:v for k,v in saves.items() if k!='scenarios'}
    for name, report in reports.items():
        write_json(args.output/(name+'.json'), report)
        if args.public_reports:
            args.public_reports.mkdir(parents=True, exist_ok=True)
            write_json(args.public_reports/(name+'.json'), report)
    print({name:r.get('decision') for name,r in reports.items()})
    print({'offline_save_contracts':saves['contract_tests']})


if __name__ == '__main__':
    main()
