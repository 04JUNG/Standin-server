"""Blender entry point for evaluated skin on a frozen, label-free candidate pool."""
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector
import numpy as np


def main():
    root = Path(sys.argv[sys.argv.index('--') + 1])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from converter.mesh_validation import extract_evaluated_skin
    from experiments.refine_v5.audit_iterative_native import load

    cases = json.loads((root / 'pool.json').read_text())
    checks = []
    for case in cases:
        source = Path(case['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != case['source_sha256']:
            raise ValueError('candidate pool source changed')
        armature, meshes = load(source)
        bones = {bone.name.split(':')[-1]: bone for bone in armature.pose.bones}
        original = {name: bone.matrix_basis.copy() for name, bone in bones.items()}
        for candidate in case['candidates']:
            for name, bone in bones.items():
                bone.matrix_basis = original[name]
            for name, vector in zip(candidate['active_bones'], candidate['delta_rotvec'], strict=True):
                rotation = Vector(vector)
                if rotation.length > 1e-12:
                    delta = Quaternion(rotation.normalized(), rotation.length).to_matrix().to_4x4()
                    bones[name].matrix_basis = original[name] @ delta
            bpy.context.view_layer.update()
            if bones.keys() != candidate['expected_heads'].keys():
                raise ValueError('candidate bone inventory differs from P5')
            head_error = max(
                (armature.matrix_world @ bone.head - Vector(candidate['expected_heads'][name])).length
                for name, bone in bones.items()
            )
            if head_error > 1e-5:
                raise ValueError('candidate skin pose differs from P5')
            checks.append(dict(case_id=case['case_id'], candidate=candidate['name'], max_head_error_m=head_error))
            vertices, faces = extract_evaluated_skin(meshes)
            if not vertices or not faces:
                raise ValueError('candidate has no evaluated skin')
            np.savez_compressed(root / candidate['mesh'], vertices=vertices, faces=faces)
        print('SKIN', case['case_id'], len(case['candidates']), flush=True)
    (root / 'mesh-pose-checks.json').write_text(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
