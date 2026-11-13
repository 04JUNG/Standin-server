"""Track which contacts immobilize the output peak edge across recorded stages."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from converter.local import surface
from scripts.local_converter.experimental.contact import potential_pairs, static_relation
from scripts.local_converter.experimental.contact_features import crossing_edges


def inspect(source, output, trace=None):
    with np.load(source) as data:
        rest, faces = data["rest"], data["triangles"]
        graph = surface.Neighborhood(faces, rest)
        lengths = surface.lengths(rest[graph.b] - rest[graph.a])
        valid = lengths > np.ptp(rest, axis=0).max() * 1e-10
        posed = data["corrected"]
        ratios = np.divide(surface.lengths(posed[graph.b] - posed[graph.a]), lengths,
                           out=np.zeros_like(lengths), where=valid)
        worst = ratios.argmax()
        endpoints = np.array([graph.a[worst], graph.b[worst]])
        incident = np.isin(faces, endpoints).any(axis=1)
        rows = []
        previous = set()
        stages = {"rest": rest}
        if trace:
            snapshots = sorted(trace.glob("carry-*.npz"))
            if not snapshots:
                raise ValueError("Missing interpolation samples")
            for snapshot in snapshots:
                with np.load(snapshot) as sample:
                    stages[snapshot.stem] = sample["posed"]
            np.testing.assert_allclose(stages["carry-1.000"], data["solver"], atol=2e-5, rtol=0)
        stages.update({key: data[key] for key in ("solver", "arm", "baked", "corrected") if key in data})
        for stage, points in stages.items():
            triangles = points[faces]
            extent = np.ptp(points, axis=0).max()
            contacts = set()
            proper = [0, 0]
            fixed = [False, False]
            for i, j in potential_pairs(faces, incident, triangles.min(1), triangles.max(1)):
                a, b = triangles[i], triangles[j]
                if static_relation(a, b, extent) == "separated":
                    continue
                contacts.add(tuple(sorted((int(i), int(j)))))
                interior = crossing_edges(a, b, extent) is not None
                for k, endpoint in enumerate(endpoints):
                    if endpoint in faces[i] or endpoint in faces[j]:
                        fixed[k] = True
                        proper[k] += int(interior)
            rows.append(dict(stage=stage, targetRatio=float(np.linalg.norm(np.diff(points[endpoints], axis=0))/lengths[worst]),
                             contactPairs=len(contacts), newSincePrevious=len(contacts-previous) if rows else None,
                             fixedEndpoints=fixed, properPairsPerEndpoint=proper))
            previous = contacts
    output.write_text(json.dumps(dict(edge=endpoints.tolist(), stages=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trace", type=Path)
    inspect(**vars(parser.parse_args()))
