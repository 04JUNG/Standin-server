"""Frozen FK and rasterization helpers for the manual 004 reproduction."""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from PIL import Image

def prepare(rig):
    bones = rig['bones']
    names = [b['name'] for b in bones]
    index = {n: i for i, n in enumerate(names)}
    worlds = np.array([b['matrix'] for b in bones])
    parents = []
    local = []
    for i, b in enumerate(bones):
        parent = index[b['parent']] if b['parent'] else -1
        if parent >= i:
            raise ValueError('hierarchy not topologically sorted')
        parents.append(parent)
        local.append(np.linalg.solve(worlds[parent], worlds[i]) if parent >= 0 else worlds[i])
    return (names, index, np.array(parents), np.array(local), worlds)

def fk(local, parents, active, deltas):
    rotations = Rotation.from_rotvec(np.asarray(deltas).reshape(-1, 3)).as_matrix()
    changes = dict(zip(active, rotations))
    world = []
    for i, m in enumerate(local):
        m = m.copy()
        if i in changes:
            m[:3, :3] = m[:3, :3] @ changes[i]
        world.append(world[parents[i]] @ m if parents[i] >= 0 else m)
    return np.array(world)

def frame(bones):
    by={b['name'].split(':')[-1]:b for b in bones}
    def head(n):return np.array(by[n].get('head',np.array(by[n]['matrix'])[:3,3]),float)
    center=head('Hips');up=head('Spine2')-center;scale=np.linalg.norm(up)
    if scale<1e-6:raise ValueError('degenerate torso')
    up/=scale;right=head('LeftArm')-head('RightArm');right-=up*np.dot(right,up)
    if np.linalg.norm(right)<1e-6:raise ValueError('degenerate shoulders')
    right/=np.linalg.norm(right);forward=np.cross(right,up)
    return by,center,scale,np.column_stack([right,up,forward])

def render_projected(vertices, faces, xy, size):
    image = np.full((size[1], size[0], 3), 248, np.uint8)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    light = np.array([-.3, .5, -.8])
    shade = np.clip(abs(normal @ light), 0, 1) * 75 + 145
    pixels = xy.astype('int32')
    for i in np.argsort(triangles[:, :, 2].mean(1)):
        color = int(shade[i])
        cv2.fillConvexPoly(image, pixels[faces[i]], (color, color + 8, color + 12))
    return Image.fromarray(image)

