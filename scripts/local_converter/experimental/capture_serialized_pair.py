"""Read actual posed FBX vertices for the offline geometry acceptance check."""
import argparse
import hashlib
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from converter.local import inspection


def capture(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _, meshes = inspection.import_fbx(path)
    if len(meshes) != 1:
        raise ValueError("This probe requires one mesh per FBX")
    obj = meshes[0]
    obj.data.calc_loop_triangles()
    points = np.asarray(inspection.surface(meshes)[obj.name])
    triangles = np.array([triangle.vertices[:] for triangle in obj.data.loop_triangles])
    return points, triangles, hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("reference", "candidate", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    before, faces, source_hash = capture(args.reference)
    after, after_faces, candidate_hash = capture(args.candidate)
    np.savez_compressed(args.output, before=before, after=after, faces=faces,
                        afterFaces=after_faces, sourceSHA=source_hash, candidateSHA=candidate_hash)
