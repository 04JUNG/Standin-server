"""Rest-relative surface correction for a baked personal-character pose.

Only mesh connectivity and world-space positions enter the solve. Keep open
boundaries and split seams fixed. Accept a change only after measuring it.
"""
import numpy as np


def lengths(vectors):
    return np.linalg.norm(vectors, axis=-1)


class Neighborhood:
    def __init__(self, faces, rest):
        vertex_count = len(rest)
        raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges, inverse, counts = np.unique(np.sort(raw, axis=1), axis=0,
                                           return_inverse=True, return_counts=True)
        self.a, self.b = edges.T
        # Cotangent weights reduce tangential drift on irregular triangulations.
        opposite = np.concatenate((faces[:, 2], faces[:, 0], faces[:, 1]))
        left = rest[raw[:, 0]] - rest[opposite]
        right = rest[raw[:, 1]] - rest[opposite]
        cot = np.einsum('ij,ij->i', left, right) / np.maximum(lengths(np.cross(left, right)), 1e-30)
        self.weights = np.clip(np.bincount(inverse.ravel(), cot), 0., 50.)
        self.mass = self.total(self.weights[:, None], vertex_count)
        self.degree = np.bincount(edges.ravel(), minlength=vertex_count)
        self.fixed = self.degree < 3
        self.fixed[edges[counts != 2].ravel()] = True
        order = np.argsort(inverse.ravel(), kind="stable")
        starts = np.cumsum(counts) - counts
        shared = starts[counts == 2]
        self.face_pairs = np.column_stack((order[shared], order[shared + 1])) % len(faces)

    def total(self, values, size=None):
        size = len(self.degree) if size is None else size
        # Sum one per-edge value at both endpoints.
        return np.column_stack([
            np.bincount(self.a, column, size) +
            np.bincount(self.b, column, size) for column in values.T
        ])

    def smooth(self, points, steps):
        current = points.copy()
        for _ in range(steps):
            diff = (current[self.b] - current[self.a])*self.weights[:, None]
            total = np.column_stack([
                np.bincount(self.a, column, len(points)) -
                np.bincount(self.b, column, len(points)) for column in diff.T
            ])
            current += .5 * total / np.maximum(self.mass, 1e-12)
            current[self.fixed] = points[self.fixed]
        return current


def rotations(graph, reference, posed):
    a, b = graph.a, graph.b
    covariance = ((posed[b]-posed[a])[:, :, None] *
                  (reference[b]-reference[a])[:, None, :])
    covariance = graph.total(covariance.reshape(-1, 9)).reshape(-1, 3, 3)
    left, _, right = np.linalg.svd(covariance)
    left[:, :, 2] *= np.sign(np.linalg.det(left @ right))[:, None]
    return left @ right


def triangle_geometry(points, faces):
    tri = points[faces]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = lengths(normal)
    return area, normal / np.maximum(area[:, None], np.finfo(float).tiny)


def metrics(points, faces, graph, rest_lengths, rest_area, smooth_in_rest):
    ratio = lengths(points[graph.b]-points[graph.a]) / rest_lengths
    area, normals = triangle_geometry(points, faces)
    a, b = graph.face_pairs.T
    cos = np.einsum('ij,ij->i', normals[a], normals[b])
    return dict(maxStretch=float(ratio.max()), stretchP99=float(np.percentile(ratio, 99)),
                overDouble=int((ratio > 2).sum()), compressed=int((ratio < .2).sum()),
                collapsed=int((area < .01*rest_area).sum()),
                sharp=int(((cos < .5) & smooth_in_rest).sum()),
                folded=int(((cos < 0) & smooth_in_rest).sum()))


def limit_strain(candidate, posed, graph, rest_lengths, cap):
    a, b = graph.a, graph.b
    free = ~graph.fixed
    mobility = free[a].astype(float) + free[b]
    x = candidate.copy()
    for _ in range(70):
        edge = x[b]-x[a]
        length = lengths(edge)
        excess = length - np.clip(length, .27*rest_lengths, 1.85*rest_lengths)
        bad = (abs(excess) > rest_lengths*1e-5) & (mobility > 0)
        if not bad.any():
            break
        i, j = a[bad], b[bad]
        step = edge[bad] * (excess[bad]/(length[bad]+1e-30)/mobility[bad])[:, None]
        move = np.column_stack([
            np.bincount(i, col, len(x))-np.bincount(j, col, len(x)) for col in step.T
        ])
        count = np.bincount(np.concatenate((i, j)), minlength=len(x))
        x += .7*move / np.maximum(count[:, None], 1)
        delta = x-posed
        x = posed + delta * np.minimum(1., cap / np.maximum(lengths(delta), 1e-30))[:, None]
        x[graph.fixed] = posed[graph.fixed]
    return x


def _correct_part(rest, posed, faces, pinned):
    """Return a candidate in the same units, or the original on fallback."""
    rest, posed = np.asarray(rest, dtype=float), np.asarray(posed, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if not len(faces) or len(rest) > 150000 or len(faces) > 300000:
        return posed, {"applied": False, "reason": "mesh_budget"}
    graph = Neighborhood(faces, rest)
    edge_lengths = lengths(rest[graph.b]-rest[graph.a])
    scale = float(2 * lengths(rest - rest.mean(axis=0)).max())
    rest_area, normals = triangle_geometry(rest, faces)
    if scale == 0 or edge_lengths.min() <= scale*1e-9 or rest_area.min() <= scale**2*1e-14:
        return posed, {"applied": False, "reason": "degenerate_reference"}
    # Pin coincident vertices of UV splits even on a closed mesh.
    _, ids, counts = np.unique(np.round((rest-rest.min(0))/(scale*1e-7)).astype(int),
                               axis=0, return_inverse=True, return_counts=True)
    graph.fixed |= pinned | (counts[ids.ravel()] > 1)
    i, j = graph.face_pairs.T
    alignment = np.einsum("ij,ij->i", normals[i], normals[j])
    smooth = alignment > np.cos(np.pi/6)
    # Anchor both incident faces at already sharp features of the original.
    creases = graph.face_pairs[alignment < np.cos(np.pi/3)]
    graph.fixed[faces[creases].ravel()] = True
    before = metrics(posed, faces, graph, edge_lengths, rest_area, smooth)
    # An object's broad folds can span many rings. Use a radius proportional
    # to its extent, bounded in rings for dense or unevenly sampled meshes.
    steps = int(np.clip((.055*scale / np.median(edge_lengths))**2, 20, 140))
    reference = graph.smooth(rest, steps)
    low_pose = graph.smooth(posed, steps)
    rotation = rotations(graph, reference, low_pose)
    detail = np.einsum('nij,nj->ni', rotation, rest-reference)
    delta = low_pose + detail - posed
    cap = np.full(len(rest), .021*scale)
    cap[graph.fixed] = 0.
    correction = delta * np.minimum(1., cap/np.maximum(lengths(delta), 1e-30))[:, None]
    if lengths(correction).max() <= scale*1e-6:
        return posed, {"applied": False, "reason": "already_smooth"}
    activity = np.clip(graph.smooth(lengths(delta)[:, None], 8) / (.003*scale), 0, 1)
    for fraction in (1., .75, .5, .4, .3, .2):
        seed = posed + fraction*correction
        fair = seed + activity*(graph.smooth(seed, max(12, steps // 3))-seed)
        x = limit_strain(fair, posed, graph, edge_lengths, cap)
        after = metrics(x, faces, graph, edge_lengths, rest_area, smooth)
        if after["collapsed"] == 0 and all(after[k] <= before[k]+1e-6 for k in before):
            return x, {"applied": True, "before": before, "after": after,
                       "rings": steps, "strength": fraction, "maxMove": float(lengths(x-posed).max())}
    return posed, {"applied": False, "reason": "metrics_regressed", "before": before}


def correct(rest, posed, faces, pinned=None):
    """Solve connected pieces independently so a large prop cannot set body scale."""
    rest = np.asarray(rest, dtype=float)
    posed = np.asarray(posed, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if not len(faces) or len(rest) > 150000 or len(faces) > 300000:
        return posed, {"applied": False, "reason": "mesh_budget"}
    pinned = np.zeros(len(rest), dtype=bool) if pinned is None else pinned.copy()
    # Keep coincident vertices on separate islands together by pinning them.
    _, ids, counts = np.unique(rest, axis=0, return_inverse=True, return_counts=True)
    pinned |= counts[ids.ravel()] > 1
    roots = list(range(len(rest)))

    def root(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    for a, b, c in faces.tolist():
        parent = root(a)
        roots[root(b)] = parent
        roots[root(c)] = parent
    labels = np.fromiter((root(i) for i in faces[:, 0]), dtype=int, count=len(faces))
    order = np.argsort(labels)
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    if len(groups) > 256:
        return posed, {"applied": False, "reason": "component_budget"}
    result = posed.copy()
    reports = []
    for group in groups:
        vertices, indices = np.unique(faces[group], return_inverse=True)
        points, report = _correct_part(rest[vertices], posed[vertices],
                                       indices.reshape(-1, 3), pinned[vertices])
        result[vertices] = points
        reports.append(report)
    return result, {"applied": any(row["applied"] for row in reports), "parts": reports}


def within_budget(meshes):
    return sum(len(mesh.data.vertices) for mesh in meshes) <= 150000 and sum(
        max(0, len(poly.vertices)-2) for mesh in meshes for poly in mesh.data.polygons
    ) <= 300000


def apply_to_meshes(meshes, neutral, normal_policies):
    from mathutils import Vector

    if not within_budget(meshes):
        return {"applied": False, "reason": "job_budget"}
    # Shared points at clothing/accessory boundaries stay fixed across objects.
    all_rest = np.concatenate([neutral[mesh.name] for mesh in meshes])
    _, ids, counts = np.unique(all_rest, axis=0, return_inverse=True, return_counts=True)
    pin = counts[ids.ravel()] > 1
    offset = 0
    reports = []
    for obj in meshes:
        mesh = obj.data
        count = len(mesh.vertices)
        pinned = pin[offset:offset+count]
        offset += count
        if mesh.users > 1:
            reports.append({"applied": False, "reason": "instanced_mesh"})
            continue
        mesh.calc_loop_triangles()
        faces = np.asarray([tri.vertices[:] for tri in mesh.loop_triangles], dtype=int)
        posed = np.asarray([obj.matrix_world @ vertex.co for vertex in mesh.vertices])
        points, report = correct(neutral[obj.name], posed, faces, pinned)
        if report["applied"]:
            inverse = obj.matrix_world.inverted()
            moved = np.any(points != posed, axis=1)
            for index in np.flatnonzero(moved):
                mesh.vertices[int(index)].co = inverse @ Vector(points[index])
            mesh.update()
            report["updatedCorners"] = update_shading(mesh, normal_policies[obj.name], moved)
        reports.append(report)
    return {"applied": any(row["applied"] for row in reports), "meshes": reports}


def geometric_normals(vertices, faces):
    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals = np.zeros_like(vertices)
    for column in faces.T:
        np.add.at(normals, column, area_vectors)
    return normals / np.maximum(lengths(normals)[:, None], 1e-30)


def normal_policy(mesh):
    """Derive corner eligibility from the registered model's authored shading."""
    mesh.calc_loop_triangles()
    faces = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=int)
    vertex_ids = np.array([loop.vertex_index for loop in mesh.loops])
    neutral = np.array([v.co[:] for v in mesh.vertices])
    authored = np.array([n.vector[:] for n in mesh.corner_normals])
    if not len(faces):
        return np.zeros(len(vertex_ids))
    geometric = geometric_normals(neutral, faces)[vertex_ids]
    alignment = (authored * geometric).sum(axis=1)
    # Fade correction out at corners with an intentional custom direction.
    strength = np.clip((alignment-np.cos(np.radians(18))) /
                       (np.cos(np.radians(5))-np.cos(np.radians(18))), 0., 1.)
    for polygon in mesh.polygons:
        if not polygon.use_smooth:
            strength[polygon.loop_start:polygon.loop_start+polygon.loop_total] = 0.
    sharp = mesh.attributes.get("sharp_edge")
    if sharp:
        hard_vertices = {idx for edge in mesh.edges if sharp.data[edge.index].value
                         for idx in edge.vertices}
        strength[np.isin(vertex_ids, list(hard_vertices))] = 0.
    return strength


def update_shading(mesh, policy, moved):
    vertex_ids = np.array([loop.vertex_index for loop in mesh.loops])
    faces = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=int)
    active = np.zeros(len(moved), dtype=bool)
    active[faces[moved[faces].any(axis=1)].ravel()] = True
    weights = policy * active[vertex_ids]
    original = np.array([normal.vector[:] for normal in mesh.corner_normals])
    current = np.array([v.co[:] for v in mesh.vertices])
    target = geometric_normals(current, faces)[vertex_ids]
    use = (weights > 0) & (lengths(target) > .99)
    normals = original.copy()
    normals[use] += weights[use, None] * (target[use]-original[use])
    normals[use] /= np.maximum(lengths(normals[use])[:, None], 1e-30)
    mesh.normals_split_custom_set(normals.tolist())
    mesh.update()
    return int(use.sum())
