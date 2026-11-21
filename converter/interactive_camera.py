"""Exact search or explicit presentation projection; fixed torso framing v2.

Search looks along +Z and uses image-down Y. WebGL looks along -Z; the
proper camera rotation and negative X projection preserve the same screen
coordinates without rounding yaw or rotating/baking the character.
"""
import math
from converter.camera import validate_rotation
from converter.solved_state import digest, canonical

FRAME = dict(target=[0., 0., 0.], referenceAspect=1., projectionIncludesCenterScaleCrop=True)


def _validate_descriptor(descriptor):
    """Accept both server camera contracts without calling a fixed view search-fit."""
    version = descriptor.get('version')
    if version == 'candidate-camera-v2':
        if descriptor.get('reference') != 'search-fit':
            raise ValueError('complete search projection required')
    elif version == 'pose-preview-camera-v1':
        required = dict(basis='right-handed-Y-up', rotation_order='Rz@Rx@Ry_column_vectors',
                        origin='hip_midpoint', scale='hip_to_shoulder_midpoint_length', screen_y='down')
        if (any(descriptor.get(key) != value for key, value in required.items())
                or descriptor.get('fit_to_rough_pixels') is not False
                or descriptor.get('bvh_camera_rotation_applied') is not False
                or type(descriptor.get('projection_matches_search')) is not bool):
            raise ValueError('invalid presentation camera contract')
        angles = [descriptor.get(key) for key in ('yaw_degrees', 'pitch_degrees', 'roll_degrees')]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in angles):
            raise ValueError('invalid presentation camera angles')
        projection = descriptor.get('projection')
        rule = 'xy=(x,-y)' if projection == 'orthographic' else 'xy=(x,-y)/(z+distance_torso)'
        if descriptor.get('projection_rule') != rule:
            raise ValueError('invalid presentation projection rule')
    else:
        raise ValueError('complete camera projection required')
    rotation = validate_rotation(descriptor.get('rotation'))
    if rotation is None:
        raise ValueError('camera rotation required')
    return rotation


def camera_for_model(descriptor, model, character_sha, scope):
    import numpy as np
    from converter.bone_map import PROFILES, resolve_profile
    rotation = np.asarray(_validate_descriptor(descriptor), float)
    projection = descriptor.get('projection')
    distance_torso = descriptor.get('distance_torso')
    if projection not in ('orthographic', 'perspective'):
        raise ValueError('unknown search projection')
    minimum = .8 if descriptor['version'] == 'candidate-camera-v2' else 0.
    if projection == 'perspective' and (type(distance_torso) not in (int,float) or not math.isfinite(distance_torso)
                                      or distance_torso < minimum or distance_torso <= 0):
        raise ValueError('invalid perspective distance')
    if projection == 'orthographic' and distance_torso is not None:
        raise ValueError('orthographic distance must be absent')
    bones = model['bones']; profile = PROFILES[resolve_profile([b['name'] for b in bones])]
    world = {}; positions = {}
    for b in bones:
        matrix = np.asarray(b['restLocal']).reshape(4,4)
        world[b['name']] = matrix if b['parent'] < 0 else world[bones[b['parent']]['name']] @ matrix
        positions[b['name']] = world[b['name']][:3,3]
    hip = (positions[profile['upleg.L']] + positions[profile['upleg.R']])/2
    shoulders = (positions[profile['upperarm.L']] + positions[profile['upperarm.R']])/2
    torso = float(np.linalg.norm(shoulders-hip))
    if not math.isfinite(torso) or torso < 1e-6:
        raise ValueError('invalid torso calibration')
    d = (distance_torso if projection == 'perspective' else 12.) * torso
    view = np.eye(4); view[:3,:3] = np.diag([-1.,1.,-1.]) @ rotation
    view[:3,3] = -view[:3,:3] @ hip + [0.,0.,-d]
    # Scope framing is a protocol constant, not a fit to a changed mesh/hand.
    span, center = {'full':(5.,.25), 'half':(3.,.9), 'bust':(1.9,1.35),
                    'shoulder_head':(1.65,1.5), 'head':(1.05,1.85)}[scope]
    scale = 2./(span*torso)
    near, far = .1*torso, max(100.*torso,d+10.*torso)
    p = np.zeros((4,4))
    if projection == 'perspective':
        # Search xy/depth, with focal length d to keep the same torso frame.
        p[0,0] = -scale*d; p[1,1] = scale*d
        p[1,2] = 2.*center/span
        p[2,2] = -(far+near)/(far-near); p[2,3] = -2.*far*near/(far-near); p[3,2] = -1.
    else:
        p[0,0] = -scale; p[1,1] = scale; p[1,3] = -2.*center/span
        p[2,2] = -2./(far-near); p[2,3] = -(far+near)/(far-near); p[3,3] = 1.
    return dict(projection=projection,viewMatrix=view.ravel().tolist(),projectionMatrix=p.ravel().tolist(),
        distanceTorso=float(distance_torso or 12.),torsoReferenceSha256=character_sha,baked=False)
