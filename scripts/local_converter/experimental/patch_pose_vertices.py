"""Transfer a measured local displacement into a copy of a single posed FBX."""
import argparse
import array
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from converter.outline_fbx import fingerprint, one, write
from converter.local.inspection import bounded_fbx_parser


def read(path):
    with bounded_fbx_parser() as parser:
        tree, version = parser.parse(str(path))
    objects = one(tree,b"Objects")
    meshes = [node for node in objects.elems if node.id == b"Geometry" and node.props[-1] == b"Mesh"]
    if len(meshes) != 1:
        raise ValueError("Expected one posed mesh")
    return tree, version, meshes[0]


def patch(source, baseline, proposed, changes, output, subset=None, scale=1.):
    inputs = [p for p in (source,baseline,proposed,changes,subset) if p is not None]
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Candidate needs a separate output path")
    root, version, geometry = read(source)
    vertices = one(geometry,b"Vertices")
    original = np.asarray(vertices.props[0]).reshape(-1,3)
    topology = fingerprint(one(geometry,b"PolygonVertexIndex"))
    values = []
    for path in (baseline,proposed):
        _, _, mesh = read(path)
        if topology != fingerprint(one(mesh,b"PolygonVertexIndex")):
            raise ValueError("Different FBX polygon indexing")
        values.append(np.asarray(one(mesh,b"Vertices").props[0]).reshape(-1,3))
    if not all(v.shape==original.shape for v in values):
        raise ValueError("Vertex tables differ")
    extent = np.ptp(original,axis=0).max()
    if np.linalg.norm(values[0]-original,axis=1).max() > extent*2e-5:
        raise ValueError("No-edit reexport changed the local coordinate frame")
    with np.load(changes) as data:
        ids = data["ids"]
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(original)):
        raise ValueError("Invalid patch indices")
    if not np.isfinite(scale):
        raise ValueError("Invalid displacement scale")
    diff = values[1]-values[0]
    fixed = np.ones(len(original),dtype=bool)
    fixed[ids] = False
    if np.any(diff[fixed]) or not np.isfinite(diff).all():
        raise ValueError("Displacement exceeds specified support")
    if subset:
        with np.load(subset) as chosen:
            selected = chosen["ids"]
        if not np.isin(selected, ids).all():
            raise ValueError("Subset outside verified support")
        ids = selected
    result = original.copy()
    result[ids] += scale*diff[ids]
    write_vertices(root, version, vertices, result, output)


def write_vertices(root, version, vertices, result, output):
    vertices.props[0] = array.array(vertices.props[0].typecode,result.ravel())
    expected = fingerprint(one(root,b"Objects"))
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        trial = Path(temp)/"trial.fbx"
        write(trial, root, version)
        reread, _, _ = read(trial)
        if fingerprint(one(reread,b"Objects"))!=expected or fingerprint(one(reread,b"Connections"))!=fingerprint(one(root,b"Connections")):
            raise ValueError("FBX object or connection changed during encoding")
        trial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source","baseline","proposed","changes","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--scale",type=float,default=1.)
    parser.add_argument("--subset",type=Path)
    patch(**vars(parser.parse_args(sys.argv[sys.argv.index("--")+1:])))
