"""Apply a staged world-space delta directly to one posed FBX control table."""
import argparse
import hashlib
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter.local import inspection
from converter.outline_fbx import one
from scripts.local_converter.experimental.patch_pose_vertices import read, write_vertices


def patch(source, stage, changes, output, subset=None, scale=1.):
    if output.resolve() in {p.resolve() for p in (source,stage,changes,subset) if p}:
        raise ValueError("Candidate must use a separate output path")
    root, version, geometry = read(source)
    field = one(geometry,b"Vertices")
    local = np.asarray(field.props[0]).reshape(-1,3)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _, meshes = inspection.import_fbx(source)
    if len(meshes)!=1:
        raise ValueError("Expected one posed mesh")
    world = np.asarray(inspection.surface(meshes)[meshes[0].name])
    with np.load(stage) as data:
        np.testing.assert_allclose(world,data["corrected"],atol=2e-5,rtol=0)
    # Baked pose jobs express their control vertices in a single affine frame.
    # Refuse a rank-deficient or spatially varying mapping.
    centered = local-local.mean(axis=0)
    centered_world = world-world.mean(axis=0)
    linear, _, rank, singular = np.linalg.lstsq(centered,centered_world,rcond=None)
    if rank!=3 or min(singular)<max(singular)*1e-8 or np.linalg.cond(linear)>1e6:
        raise ValueError("Unsupported control frame")
    if np.linalg.norm(centered@linear-centered_world,axis=1).max()>np.ptp(world,axis=0).max()*1e-6:
        raise ValueError("Control table does not match the evaluated posed mesh")
    with np.load(changes) as data:
        if "sourceSHA" in data and str(data["sourceSHA"])!=hashlib.sha256(stage.read_bytes()).hexdigest():
            raise ValueError("Patch belongs to another stage")
        ids, delta = data["ids"], data["delta"]
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(local)):
        raise ValueError("Invalid patch indices")
    if delta.shape!=(len(ids),3) or not np.isfinite(delta).all() or not np.isfinite(scale):
        raise ValueError("Invalid patch displacement")
    if subset:
        with np.load(subset) as data:
            selected = data["ids"]
        if not np.isin(selected,ids).all():
            raise ValueError("Subset outside original support")
        keep = np.isin(ids,selected)
        ids,delta = ids[keep],delta[keep]
    result = local.copy()
    result[ids] += np.linalg.solve(linear.T,(scale*delta).T).T
    write_vertices(root,version,field,result,output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source","stage","changes","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--subset",type=Path)
    parser.add_argument("--scale",type=float,default=1.)
    patch(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
