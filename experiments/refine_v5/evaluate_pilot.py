"""Independent measurements from reimported actual FBX snapshots and GT."""
import argparse, json
from pathlib import Path
import numpy as np
from .model import ACTIVE, angles

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--run', default='learned')
    a = p.parse_args()
    r = Path(a.root)
    gt = {x['case_id']: x for x in json.loads((r / 'data/native_audit.json').read_text())}
    report = []
    for row in json.loads((r / a.run / 'records.json').read_text()):
        target = {b['name'].split(':')[-1]: b for b in gt[row['case_id']]['bones']}
        vec = np.array([np.array(target[n.split(':')[-1]]['tail']) - target[n.split(':')[-1]]['head'] for n in ACTIVE])
        center = np.array(target['Hips']['head'])
        scale = np.linalg.norm(np.array(target['Spine2']['head']) - center)
        metrics = {}
        for key in ['before', 'planned', 'actual']:
            state = row[key]
            dirs = np.array([state['arms'][n]['direction'] for n in ACTIVE])
            errors = []
            for name in ACTIVE:
                bone = state['arms'][name]
                g = target[name.split(':')[-1]]
                for point in ['head', 'tail']:
                    pred = (np.array(bone[point]) - state['center']) / state['scale']
                    truth = (np.array(g[point]) - center) / scale
                    errors.append(float(np.linalg.norm(pred - truth)))
            metrics[key] = dict(mean_angle_deg=float(angles(dirs, vec).mean()), max_angle_deg=float(angles(dirs, vec).max()), joint_nme=float(np.mean(errors)))
        drift = max((abs(row['before']['arms'][n]['length'] - row['actual']['arms'][n]['length']) for n in ACTIVE))
        report.append(dict(case_id=row['case_id'], metrics=metrics, max_arm_length_drift_m=drift, full_pose_complete=False, hand_orientation_trained=False, mesh_checked=False))
    path = r / (a.run + '-metrics.json')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
if __name__ == '__main__':
    main()
