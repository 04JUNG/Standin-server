"""Exact continuous search projection; fixed, server-defined torso framing v2.

Search looks along +Z and uses image-down Y. WebGL looks along -Z; the
proper camera rotation and negative X projection preserve the same screen
coordinates without rounding yaw or rotating/baking the character.
"""
import math
from converter.camera import validate_rotation
from converter.solved_state import digest, canonical

FRAME = dict(target=[0., 0., 0.], referenceAspect=1., projectionIncludesCenterScaleCrop=True)


def camera_for_model(descriptor, model, character_sha, scope):
    import numpy as np
    from converter.bone_map import PROFILES, resolve_profile
    if descriptor.get('version') != 'candidate-camera-v2' or descriptor.get('reference') != 'search-fit':
        raise ValueError('complete search projection required')
    rotation = np.asarray(validate_rotation(descriptor['rotation']), float)
    projection = descriptor.get('projection')
    distance_torso = descriptor.get('distance_torso')
    if projection not in ('orthographic', 'perspective'):
        raise ValueError('unknown search projection')
    if projection == 'perspective' and (type(distance_torso) not in (int,float) or not math.isfinite(distance_torso) or distance_torso < .8):
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
