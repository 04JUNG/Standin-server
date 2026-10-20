"""Audit canonical facing without editing BVH channels or search indexes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bvh import load_coco17
from src.pose_camera import canonical_yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bvh-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.bvh_dir.glob('*.bvh')):
        try:
            points, scores = load_coco17(str(path))
            if any(scores[i] < .3 for i in (5, 6, 11, 12)):
                raise ValueError('missing anchors')
            yaw = canonical_yaw(points)
            rows.append({'pose_id': path.stem, 'status': 'ok', 'canonical_yaw': round(yaw, 3),
                         'raw_front_differs': abs(yaw) > 30})
        except (ValueError, OSError, AssertionError, IndexError):
            rows.append({'pose_id': path.stem, 'status': 'ambiguous'})
    summary = dict(Counter(row['status'] for row in rows))
    summary['raw_front_differs'] = sum(row.get('raw_front_differs', False) for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'poses': rows}, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
