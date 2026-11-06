"""GT is read only by this training driver; no GT fields in predictions."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from .model import ACTIVE, Corrector, feature, angles, VERSION

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    audit = {r['case_id']: r for r in json.loads((root / 'native_audit.json').read_text())}
    inv = {r['case_id']: r for r in json.loads((root / 'inventory.json').read_text())}
    cases = json.loads(Path(a.base).read_text())
    x = []
    y = []
    labels = []
    for r in cases:
        if inv[r['case_id']]['split'] != 'development':
            raise ValueError('only development training permitted')
        gt = audit[r['case_id']]
        bones = {b['name'].split(':')[-1]: b for b in gt['bones']}
        target = np.array([np.array(bones[n.split(':')[-1]]['tail']) - bones[n.split(':')[-1]]['head'] for n in ACTIVE])
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        f, base = feature(r['image'], r['state'])
        x.append(f)
        y.append((target - base).ravel())
        labels.append(target)
    model = Corrector.fit(x, y)
    checkpoint = out / 'checkpoint.npz'
    model.save(checkpoint)
    model = Corrector.load(checkpoint)
    predictions = []
    metrics = []
    for r, target in zip(cases, labels):
        pred = model.predict(r['image'], r['state'])
        _, base = feature(r['image'], r['state'])
        predictions.append(dict(case_id=r['case_id'], base_fbx=r['base_fbx'], prediction=pred.tolist()))
        metrics.append(dict(case_id=r['case_id'], baseline_angle_deg=angles(base, target).tolist(), predicted_angle_deg=angles(pred, target).tolist()))
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    report = dict(version=VERSION, experiment='training_reconstruction', cases=len(cases), output='four_arm_bone_directions_only', gt_runtime_input=False, independent_holdout=False, training_metadata=[dict(case_id=r['case_id'], image_sha256=inv[r['case_id']]['files']['image']['sha256'], gt_sha256=audit[r['case_id']]['sha256'], seed_kind=r['seed_kind']) for r in cases], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), metrics=metrics, full_pose_complete=False, mesh_checked=False)
    (out / 'training.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics))
if __name__ == '__main__':
    main()
