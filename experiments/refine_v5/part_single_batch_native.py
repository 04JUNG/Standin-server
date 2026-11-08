"""Export each single solve before loading the benchmark ground truths."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.refine_v5.ab_blender import bone_world, sha, weighted_mesh
from experiments.refine_v5.ab_native_audit import write
from experiments.refine_v5.audit_iterative_native import load
from experiments.refine_v5.part_single_native import export


def main(root, previous):
    cases = json.loads((root / 'manifest.json').read_text())['cases']
    for case in cases:
        export(root / case['case_id'], 'detector')
    gt_rows = {r['case_id']: r for r in json.loads((previous / 'data/native_audit.json').read_text())}
    for case in cases:
        cid = case['case_id']
        path = previous / 'data/정답_3d인형' / f'{cid}.fbx'
        if sha(path) != gt_rows[cid]['sha256']:
            raise ValueError('pinned GT changed')
        arm, meshes = load(path)
        world = bone_world(arm)
        vertices, faces, _ = weighted_mesh(meshes, list(world))
        np.savez_compressed(root / cid / 'gt.npz', vertices=vertices, faces=faces)
        write(root / cid / 'gt.json', dict(sha256=sha(path), bones=[dict(name=n, matrix=m.tolist()) for n, m in world.items()]))


if __name__ == '__main__':
    main(*map(Path, sys.argv[sys.argv.index('--') + 1:]))
