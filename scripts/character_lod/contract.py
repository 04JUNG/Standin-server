"""Pure data guards shared by the offline authoring and evidence workflow."""
import hashlib
import math
from pathlib import Path

APPROVED = {
    'male': {'blend': 'a03550749315592c543eeda66a183fe4116380f2068b412105f751c7b2e9c39a',
             'controller': 'bcdabb7d69534f08400dfbd8a91a52ddc01528428332d4f14d4ad6792a7af8da'},
    'female': {'blend': '5a62d2f3fb49d77fdc04d1c78a361edf9fbab55e71e0e47d3b6b393fae14d3a9',
               'controller': '1164dff9a4c6069d0da37b80dd26c61544bc6f9eb9c9f3469c33d457e893e9af'},
}


def require_hash(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'input checksum mismatch: {Path(path).name}')
    return actual


def require_subset(ids, protected, original_faces, candidate_faces):
    ids = list(map(int, ids))
    if len(set(ids)) != len(ids) or min(ids, default=0) < 0:
        raise ValueError('correspondence must be a unique nonnegative source subset')
    protected = set(map(int, protected))
    if not protected.issubset(ids):
        raise ValueError('protected source vertex removed')
    def oriented(face):
        values = tuple(face)
        return min(values[i:] + values[:i] for i in range(len(values)))
    expected = {oriented(face) for face in original_faces if protected.intersection(face)}
    actual = {oriented(ids[i] for i in face) for face in candidate_faces}
    if not expected.issubset(actual):
        raise ValueError('protected face connectivity changed')


def require_paired_runs(rows, repeats=3):
    """Reject incomplete/cache-mixed/body-mismatched evidence before any claim."""
    groups = {}
    for row in rows:
        key = (row['gender'], row['case'], row['skin'])
        groups.setdefault(key, []).append(row)
    for key, items in groups.items():
        variants = {v: [r for r in items if r['variant'] == v] for v in ('original', 'candidate')}
        for values in variants.values():
            if len(values) != repeats or {r['trial'] for r in values} != set(range(repeats)):
                raise ValueError(f'incomplete paired measurement: {key}')
            for row in values:
                if not row['ok'] or row['report']['shape_bake']['neutral_preparation']['status'] != 'prepared':
                    raise ValueError('failed conversion or mixed neutral preparation')
        if len({str(sorted(r['controls'].items())) for r in items}) != 1:
            raise ValueError('paired body controls differ')
        if len({r['bvh_sha256'] for r in items}) != 1:
            raise ValueError('paired BVH inputs differ')
    if not groups:
        raise ValueError('empty measurement')


def final_geometry_pass(metrics):
    """Same correspondence gates, plus explicit hand/body regional checks."""
    limits = {'retained_world_p99_mm': .5, 'protected_max_mm': .05,
              'hand_max_mm': .05, 'body_surface_p99_mm': .5}
    return all(key in metrics and math.isfinite(metrics[key])
               and metrics[key] <= limit for key, limit in limits.items())
