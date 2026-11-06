"""Diagnostic B0 selection from rough only. No GT opened by this program."""
import argparse, hashlib, json
from pathlib import Path
from src.pose import RTMPoseModel
from src.features import normalize_skeleton, pose_distance
from src.repo import load_entries
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', nargs='+', default=['001', '074'])
    a = p.parse_args()
    root = Path(a.data)
    lib = Path(a.library)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    entries = load_entries(str(lib / 'poses.db'))
    model = RTMPoseModel()
    rows = []
    selected = sorted((x.stem for x in (root / '러프').iterdir())) if a.cases == ['all'] else a.cases
    for case in selected:
        matches = list((root / '러프').glob(case + '.*'))
        image = matches[0]
        with Image.open(image) as im:
            w, h = im.size
        sks = model.estimate(str(image), [], w, h)
        if len(sks) != 1:
            rows.append(dict(case_id=case, status='unsupported', reason='person_correspondence_ambiguous', detected_people=len(sks)))
            continue
        sk = sks[0]
        feat = normalize_skeleton(sk.keypoints, sk.scores)
        hits = []
        for e in entries:
            path = Path(e.bvh_path or '')
            if not path.is_file():
                path = lib / 'bvh' / path.name
            if not path.is_file():
                continue
            hits.append((float(pose_distance(feat, e.feature)), e.pose_id, e.view.value, str(path)))
        hits.sort()
        best = hits[0]
        raw = Path(best[3]).read_bytes()
        seed = out / (case + '.bvh')
        seed.write_bytes(raw)
        rows.append(dict(status='ready', case_id=case, seed_kind='reconstructed_seed', seed_selection='diagnostic_geometry_top1', pose_id=best[1], view=best[2], distance=best[0], bvh=str(seed), bvh_sha256=hashlib.sha256(raw).hexdigest(), image=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), keypoints=sk.keypoints.tolist(), scores=sk.scores.tolist(), top5=hits[:5], service_eligibility=None, ownership_verified=False))
    (out / 'seeds.json').write_text(json.dumps(rows, indent=2))
    print({'inputs': len(rows), 'ready': sum((r['status'] == 'ready' for r in rows)), 'unsupported': sum((r['status'] != 'ready' for r in rows))})
if __name__ == '__main__':
    main()
