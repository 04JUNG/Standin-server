"""Pose-only contact policy for the separately versioned breast7.3 candidate.

Appended to the current controller. Authored neutral keys are never modified.
Withdraw only the new local breast residual at newly intersecting triangles,
using the old breast reference as the destination. Maximum pose shift: 0.1 mm.
The real armature's displacement Jacobian is measured, rather than assuming LBS.
"""

BREAST_CONTACT_POLICY = 'reference-local-pose-withdrawal-0.1mm-v1'


def breast_pose_contact_guard(scene, reference_surface):
    from mathutils.bvhtree import BVHTree
    mesh, arm = objects(scene)
    s = source(scene)
    faces = s['faces']
    original = evaluated(mesh)
    owned = np.linalg.norm(breast_delta(s, controls(scene)), axis=1) > 1e-10

    def crossings(points):
        tree = BVHTree.FromPolygons(points.tolist(), faces.tolist(), all_triangles=True)
        return {(i, j) for i, j in tree.overlap(tree)
                if i < j and not set(faces[i]).intersection(faces[j])}

    reference_pairs = crossings(reference_surface)
    added = crossings(original) - reference_pairs
    report = {'policy': BREAST_CONTACT_POLICY, 'active': False,
              'new_pairs_before': len(added), 'new_pairs_after': len(added),
              'max_shift_mm': 0., 'neutral_keys_changed': False, 'bones_changed': False}
    if not added:
        mesh['breast_contact_qa'] = canonical(report)
        return
    if any(not owned[faces[[i, j]]].any() for i, j in added):
        raise ValueError('Breast pose guard cannot own an unrelated new intersection')
    zone = np.zeros(len(original), dtype=bool)
    for i, j in added:
        vertices = faces[[i, j]].ravel()
        zone[vertices[owned[vertices]]] = True
    # One contiguous ring avoids an isolated vertex pull; support remains inside
    # the authored breast residual and cannot reach the arm, back or hands.
    edges = s['edges_raw']
    expanded = zone.copy()
    expanded[edges[zone[edges[:, 0]], 1]] = True
    expanded[edges[zone[edges[:, 1]], 0]] = True
    zone = expanded & owned
    selected = np.flatnonzero(zone)
    wanted = reference_surface[selected] - original[selected]
    wanted *= np.minimum(1., .0001 / np.maximum(np.linalg.norm(wanted, axis=1), 1e-15))[:, None]
    block = mesh.data.shape_keys.key_blocks.get('AUTO_BreastContact')
    if block is None:
        block = mesh.shape_key_add(name='AUTO_BreastContact', from_mix=False)
    basis = s['co']
    block.value = 1.
    jacobian = np.empty((len(selected), 3, 3))
    for axis in range(3):
        probe = basis.copy()
        probe[selected, axis] += .0001
        block.data.foreach_set('co', probe.ravel())
        mesh.data.update()
        actual_probe = arr(block.data)[selected, axis] - basis[selected, axis]
        jacobian[:, :, axis] = (evaluated(mesh)[selected] - original[selected]) / actual_probe[:, None]
    if np.any(abs(np.linalg.det(jacobian)) < 1e-8):
        raise ValueError('Pose contact displacement Jacobian is singular')
    local = np.linalg.solve(jacobian, wanted[..., None])[..., 0]
    before_normals = np.cross(original[faces[:, 1]]-original[faces[:, 0]], original[faces[:, 2]]-original[faces[:, 0]])

    def trial(amount):
        points = basis.copy()
        points[selected] += local * amount
        block.data.foreach_set('co', points.ravel())
        mesh.data.update()
        q = evaluated(mesh)
        normals = np.cross(q[faces[:, 1]]-q[faces[:, 0]], q[faces[:, 2]]-q[faces[:, 0]])
        shift = float(np.linalg.norm(q-original, axis=1).max())
        safe = (shift <= .0001001
                and not np.any(np.linalg.norm(normals, axis=1) < 2e-12)
                and np.all(np.einsum('ij,ij->i', normals, before_normals) > 0)
                and not (crossings(q)-reference_pairs))
        return safe, shift

    ok, _ = trial(1.)
    if not ok:
        block.value = 0.
        mesh.data.update()
        raise ValueError('New breast contact cannot be resolved within the 0.1 mm pose-only bound')
    low, high = 0., 1.
    accepted_points = arr(block.data).copy()
    for _ in range(12):
        mid = (low+high)/2
        if trial(mid)[0]:
            high = mid
            accepted_points = arr(block.data).copy()
        else:
            low = mid
    # Float32 triangle contact is not strictly monotone. Keep the exact accepted
    # mesh, rather than nudging across another contact boundary after the search.
    block.data.foreach_set('co', accepted_points.ravel())
    mesh.data.update()
    final = evaluated(mesh)
    shift = float(np.linalg.norm(final-original, axis=1).max())
    normals = np.cross(final[faces[:, 1]]-final[faces[:, 0]], final[faces[:, 2]]-final[faces[:, 0]])
    if (crossings(final)-reference_pairs or shift > .0001001
            or np.any(np.linalg.norm(normals, axis=1) < 2e-12)
            or np.any(np.einsum('ij,ij->i', normals, before_normals) <= 0)):
        raise ValueError('Final breast pose contact gate failed')
    report.update(active=True, new_pairs_after=0, max_shift_mm=shift*1000,
                  affected_vertices=len(selected), residual_withdrawal=high)
    mesh['breast_contact_qa'] = canonical(report)
