"""Native GT label validation; keeps raw world geometry and a fixed torso scale."""
import argparse, json
from pathlib import Path
import numpy as np

def build_label(record):
    if record.get('status') != 'imported':
        raise ValueError('native import unavailable')
    bones = record['bones']
    names = [b['name'].split(':')[-1] for b in bones]
    if len(set(names)) != len(names):
        raise ValueError('ambiguous anatomical mapping')
    byname = dict(zip(names, bones))
    heads = np.array([b['head'] for b in bones], float)
    tails = np.array([b['tail'] for b in bones], float)
    matrices = np.array([b['matrix'] for b in bones], float)
    if not all((np.isfinite(x).all() for x in (heads, tails, matrices))):
        raise ValueError('nonfinite native geometry')
    lengths = np.linalg.norm(tails - heads, axis=1)
    if (lengths <= 1e-08).any():
        raise ValueError('degenerate bone')
    center = np.array(byname['Hips']['head'])
    scale = float(np.linalg.norm(np.array(byname['Spine2']['head']) - center))
    if scale <= 1e-06:
        raise ValueError('degenerate torso anchor')
    return dict(case_id=record['case_id'], gt_sha256=record['sha256'], joint_names=names, world_heads=heads.tolist(), world_tails=tails.tolist(), directions=((tails - heads) / lengths[:, None]).tolist(), normalized_heads=((heads - center) / scale).tolist(), torso_scale_m=scale, root_center=center.tolist(), camera_status='missing', visibility_mask_status='pending_review', pose_label_available=True, mesh_checked=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    r = Path(a.root)
    labels = []
    for record in json.loads((r / 'native_audit.json').read_text()):
        try:
            labels.append(build_label(record))
        except (ValueError, KeyError) as e:
            labels.append(dict(case_id=record['case_id'], pose_label_available=False, reason=str(e)))
    (r / 'native_labels.json').write_text(json.dumps(labels, indent=2))
    print({'labels': len(labels), 'valid': sum((x['pose_label_available'] for x in labels))})
if __name__ == '__main__':
    main()
