"""Anatomical folded-forearm support, enabled only by inferred contact action.

Targets derive from rig lengths and skin thickness. No image/person/pose IDs,
hand-authored landmarks or per-cut angles enter the solver. The forearms form
two opposing horizontal segments, with one layered above the other. Head
orientation and support height are solved jointly instead of minimizing a
single head/arm distance (which can fold the neck through the crossed arms).
"""
from __future__ import annotations
import math
import numpy as np


def solve_folded_forearm_rest(arm, meshes):
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.kdtree import KDTree
    from . import retarget as rt

    def pb(name): return arm.pose.bones['mixamorig:' + name]
    def joint(name): return arm.matrix_world @ pb(name).head
    def tail(name): return arm.matrix_world @ pb(name).tail
    up = Vector((0, 0, 1))
    right = joint('LeftArm') - joint('RightArm'); right.z = 0; right.normalize()
    forward = right.cross(up).normalized()
    rests = {b.name: b.matrix_basis.copy() for b in arm.pose.bones}
    lengths = {b.name: b.length for b in arm.data.bones}
    initial_head_frame = arm.matrix_world @ pb('Head').matrix
    head_points = []; radial = {'Left': [], 'Right': []}
    arm_clouds = {'Left': [], 'Right': []}
    for mesh in meshes:
        groups = {g.index: g.name.rsplit(':', 1)[-1] for g in mesh.vertex_groups}
        for v in mesh.data.vertices:
            weights = {groups.get(g.group, ''): g.weight for g in v.groups}
            point = mesh.matrix_world @ v.co
            if weights.get('Head', 0) >= .8: head_points.append(point)
            for side in radial:
                if weights.get(side + 'ForeArm', 0) < .8: continue
                frame = arm.matrix_world @ pb(side+'ForeArm').matrix
                normal = mesh.matrix_world.to_3x3().inverted().transposed() @ v.normal
                arm_clouds[side].append((list(frame.inverted() @ point)+[1.], list(frame.to_3x3().inverted() @ normal)))
                a, b = joint(side + 'ForeArm'), joint(side + 'Hand')
                axis = (b-a).normalized(); projection = (point-a).dot(axis)
                if .15 * (b-a).length < projection < .85 * (b-a).length:
                    radial[side].append((point-a-axis*projection).length)
    if len(head_points) < 30 or any(len(v) < 20 for v in radial.values()):
        raise ValueError('head and forearm skin landmarks required')
    radii = {side: float(np.quantile(v, .70)) for side, v in radial.items()}
    unit = (joint('LeftArm') - joint('RightArm')).length
    gap = unit * .012
    head_local = np.array([list(initial_head_frame.inverted() @ p) + [1.] for p in head_points])
    # Bounded deterministic sample for optimization; final checks use all skin.
    head_local = head_local[::max(1, len(head_local)//1600)]
    arm_clouds = {side: (np.array([p[0] for p in cloud])[::max(1,len(cloud)//1800)],
                         np.array([p[1] for p in cloud])[::max(1,len(cloud)//1800)])
                  for side, cloud in arm_clouds.items()}

    def transform(bone, rotation):
        pivot = arm.matrix_world @ bone.head
        world = arm.matrix_world @ bone.matrix
        bone.matrix = arm.matrix_world.inverted() @ Matrix.Translation(pivot) @ rotation.to_4x4() @ Matrix.Translation(-pivot) @ world
        bpy.context.view_layer.update()

    def rotate(name, angle):
        transform(pb(name), Matrix.Rotation(math.radians(angle), 3, right))

    def aim(name, direction):
        axis = tail(name) - joint(name)
        transform(pb(name), axis.rotation_difference(direction).to_matrix())

    def roll_towards(name, current, desired):
        axis = (tail(name) - joint(name)).normalized()
        current = current - axis*current.dot(axis)
        desired = desired - axis*desired.dot(axis)
        if min(current.length, desired.length) < 1e-8: return
        current.normalize(); desired.normalize()
        angle = math.atan2(axis.dot(current.cross(desired)), current.dot(desired))
        transform(pb(name), Matrix.Rotation(angle, 3, axis))

    def palm_normal(side):
        wrist = joint(side + 'Hand')
        normal = (joint(side + 'HandIndex1') - wrist).cross(joint(side + 'HandPinky1') - wrist)
        return normal if side == 'Left' else -normal

    def flatten_hand(side, direction):
        wrist = joint(side + 'Hand')
        longitudinal = joint(side + 'HandMiddle1') - wrist; longitudinal.normalize()
        normal = palm_normal(side)
        normal -= longitudinal * normal.dot(longitudinal); normal.normalize()
        old = Matrix((longitudinal, normal.cross(longitudinal), normal)).transposed()
        longitudinal_target = direction.normalized()
        new = Matrix((longitudinal_target, up.cross(longitudinal_target), up)).transposed()
        transform(pb(side + 'Hand'), new @ old.transposed())

    def head_skin():
        matrix = np.array(arm.matrix_world @ pb('Head').matrix)
        return (head_local @ matrix.T)[:, :3]

    def apply(values):
        spine_extra, head_pitch, height_offset, flare, layer_scale = values
        for b in arm.pose.bones: b.matrix_basis = rests[b.name].copy()
        bpy.context.view_layer.update()
        rotate('Spine1', spine_extra * .5); rotate('Spine2', spine_extra * .5)
        torso = joint('Neck') - joint('Hips')
        torso_pitch = math.degrees(math.atan2(torso.dot(forward), torso.dot(up)))
        neck_pitch = (torso_pitch + head_pitch) * .5
        aim('Neck', forward * math.sin(math.radians(neck_pitch)) + up * math.cos(math.radians(neck_pitch)))
        aim('Head', forward * math.sin(math.radians(head_pitch)) + up * math.cos(math.radians(head_pitch)))
        skin = head_skin()
        # Anatomical layer order, fixed for this action class, not this image.
        upper_z = float(np.quantile(skin[:, 2], .02)) - radii['Right'] - gap + height_offset
        # Vertical stacking alone leaves a large air gap when the two elbows
        # have different depths. Solve their full Y/Z separation instead.
        def elbow_at(side, z):
            shoulder = joint(side + 'Arm')
            length = (tail(side + 'Arm') - shoulder).length
            depth = math.sqrt(max(length**2-(unit*flare)**2-(z-shoulder.z)**2, .000001))
            return shoulder.dot(forward)+depth
        upper_depth = elbow_at('Right', upper_z)
        wanted = (radii['Left'] + radii['Right']) * layer_scale + gap
        low, high = 0., wanted
        for _ in range(20):
            delta = (low+high)*.5
            separation = math.hypot(delta, elbow_at('Left', upper_z-delta)-upper_depth)
            if separation < wanted: low = delta
            else: high = delta
        levels = {'Right': upper_z, 'Left': upper_z - (low+high)*.5}
        targets = {}; reach_error = 0.
        for side, sign in (('Left', 1.), ('Right', -1.)):
            shoulder = joint(side + 'Arm')
            upper_length = (tail(side + 'Arm') - shoulder).length
            lower_length = (tail(side + 'ForeArm') - joint(side + 'ForeArm')).length
            lateral = unit * flare
            vertical = levels[side] - shoulder.z
            residual = upper_length**2 - lateral**2 - vertical**2
            reach_error += max(0., -residual)
            depth = math.sqrt(max(residual, .000001))
            elbow = shoulder + right * (sign*lateral) + forward*depth + up*vertical
            wrist = elbow - right * (sign*lower_length)
            aim(side + 'Arm', elbow - shoulder)
            roll_towards(side+'Arm', tail(side+'ForeArm')-joint(side+'ForeArm'), wrist-elbow)
            aim(side + 'ForeArm', wrist - joint(side + 'ForeArm'))
            roll_towards(side+'ForeArm', palm_normal(side), up)
            flatten_hand(side, -right * sign)
            targets[side] = dict(elbow=list(elbow), wrist=list(wrist))
        return skin, targets, reach_error, torso_pitch

    def point_segment_dist(points, a, b):
        a, b = np.array(a), np.array(b); direction = b-a
        t = np.clip(((points-a) @ direction) / max(float(direction @ direction), 1e-10), 0., 1.)
        return np.linalg.norm(points - a - t[:, None]*direction, axis=1)

    def evaluate(values):
        skin, targets, reach_error, torso_pitch = apply(values)
        center = skin.mean(0); arms = {s: (joint(s+'ForeArm'), joint(s+'Hand')) for s in radii}
        clearances = {s: float(np.min(point_segment_dist(skin, *arms[s]))) - radii[s] for s in radii}
        a, b = arms['Right']; axis = b-a
        projection = max(0., min(1., (Vector(center)-a).dot(axis)/axis.length_squared))
        offset = Vector(center) - (a+axis*projection); offset.z = 0
        clouds = {}
        for side, (points, normals) in arm_clouds.items():
            matrix = np.array(arm.matrix_world @ pb(side+'ForeArm').matrix)
            clouds[side] = ((points @ matrix.T)[:, :3], normals @ np.linalg.inv(matrix[:3,:3]))
        points, normals = clouds['Right']; tree = KDTree(len(points))
        for i, p in enumerate(points): tree.insert(Vector(p), i)
        tree.balance()
        def skin_contact(query):
            minimum = float('inf'); penetration = 0.
            for p in query:
                closest, i, distance = tree.find(Vector(p)); minimum = min(minimum, distance)
                if distance < unit*.08:
                    signed = float(np.dot(p-np.array(closest), normals[i]))
                    penetration = max(penetration, -signed)
            return minimum, penetration
        head_distance, head_penetration = skin_contact(skin)
        arm_distance, arm_penetration = skin_contact(clouds['Left'][0])
        # Skin cross-sections are tapered and elliptical; a circular bone
        # capsule alone leaves visible air between supposedly stacked arms.
        support_clearance = clearances['Right']
        cost = 150*(head_distance-gap*.25)**2 + 150*(arm_distance-gap*.25)**2 + 3*offset.length_squared
        cost += 1000*(head_penetration**2 + arm_penetration**2) + 200*reach_error
        cost += .000006*values[0]**2 + .000003*(values[1]-90.)**2 + .8*values[2]**2
        cost += .02*(values[3]-.04)**2
        # For folded table rest, preserve a moderate torso and a forward crown.
        cost += .002*(max(0., 30.-torso_pitch)**2 + max(0., torso_pitch-65.)**2)
        vertical = max(abs((b-a).normalized().dot(up)) for a, b in arms.values())
        directions = [(b-a).normalized() for a, b in arms.values()]
        endpoints = {s: sorted([a.dot(right), b.dot(right)]) for s, (a, b) in arms.items()}
        overlap = min(v[1] for v in endpoints.values()) - max(v[0] for v in endpoints.values())
        if overlap < unit*.30: cost += 100*(unit*.30-overlap)**2
        report = dict(head_forearm_capsule_clearance_m=clearances, head_support_horizontal_offset_m=offset.length,
            forearm_vertical_fraction=vertical, forearm_opposition_dot=directions[0].dot(directions[1]),
            forearm_horizontal_overlap_m=overlap, torso_forward_degrees=torso_pitch,
            head_axis_pitch_degrees=values[1], targets=targets, reach_error=reach_error,
            predicted_head_skin_gap_m=head_distance, predicted_forearm_skin_gap_m=arm_distance,
            predicted_head_skin_penetration_m=head_penetration, predicted_forearm_skin_penetration_m=arm_penetration)
        return float(cost), report

    start = [0., 90., 0., .04, 1.]
    best = start[:]; best_cost, before = evaluate(best); evaluations = 1
    limits = [(-12., 30.), (75., 105.), (-unit*.12, unit*.12), (0., .08), (.45, 1.15)]
    for steps in ((10., 10., unit*.04, .04, .15), (5., 5., unit*.02, .02, .075), (2., 2., unit*.008, .008, .03), (.5, .5, unit*.002, .002, .01)):
        for sweep in range(4):
            changed = False
            for axis in range(5):
                base = best[:]
                for sign in (-1., 1.):
                    candidate = base[:]; candidate[axis] = max(limits[axis][0], min(limits[axis][1], base[axis]+sign*steps[axis]))
                    score, _ = evaluate(candidate); evaluations += 1
                    if score < best_cost: best, best_cost, changed = candidate, score, True
            if not changed: break
    _, after = evaluate(best)
    if after['forearm_vertical_fraction'] > .025 or after['forearm_opposition_dot'] > -.99 or after['reach_error'] > 1e-6:
        raise ValueError('folded-forearm anatomical constraints failed: ' + str(after))
    rt.apply_output_mode(arm, meshes, 'rigged_rest')
    skin = {'Head': [], 'LeftForeArm': [], 'RightForeArm': []}
    for mesh in meshes:
        groups = {g.index: g.name.rsplit(':', 1)[-1] for g in mesh.vertex_groups}
        for v in mesh.data.vertices:
            weights = {groups.get(g.group, ''): g.weight for g in v.groups}
            point = mesh.matrix_world @ v.co
            for part in skin:
                if weights.get(part, 0) > .8: skin[part].append(point)
    def nearest(a, b):
        tree = KDTree(len(b))
        for i, point in enumerate(b): tree.insert(point, i)
        tree.balance()
        return min(tree.find(p)[2] for p in a)
    length_error = max(abs(b.length-lengths[b.name]) for b in arm.data.bones)
    if length_error > 1e-5: raise ValueError('bone length changed')
    return dict(status='automatic_folded_forearm_support_v3', before=before, after=after,
        forearm_skin_radii_m=radii, minimum_layer_clearance_m=gap,
        head_forearm_skin_vertex_clearance_m=nearest(skin['Head'], skin['RightForeArm']),
        forearm_forearm_skin_vertex_clearance_m=nearest(skin['LeftForeArm'], skin['RightForeArm']),
        max_bone_length_change_m=length_error, parameters=dict(zip(('spine_extra_degrees','head_axis_pitch_degrees','support_height_offset_m','elbow_flare_shoulder_ratio','forearm_layer_scale'),best)),
        evaluations=evaluations, manual_targets=False)
