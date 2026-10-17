"""Internal contour evidence renderer; not a displayed character.
Frozen research rendering retained byte-for-byte for parity.
"""
import numpy as np
from PIL import Image, ImageDraw

def rotate_y(joints: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate world joints into a virtual camera yaw, matching library.py."""
    theta = np.deg2rad(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )
    points = np.asarray(joints, dtype=np.float32).reshape(17, 3)
    return points @ rotation.T

def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-8)

def render_pose(joints, yaw, size=512, style="solid", seed=0, *, pitch=0., distance=None):
    """Orthographic 3D ellipsoid mannequin with z-buffer and surface lighting.

    Screen X right, Y down; world Y up; camera looks along +Z (larger Z
    means farther). Unlike 2D sticks, overlapping limbs have depth occlusion.
    Faces/feet are approximated from body axes, not a rigged character mesh.
    """
    if style not in ("solid", "rough"):
        raise ValueError("unknown rendering style")
    points = rotate_y(joints, yaw).astype(float)
    hip = (points[11] + points[12]) / 2
    shoulder = (points[5] + points[6]) / 2
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 1e-6:
        raise ValueError("degenerate torso")
    points = (points - hip) / torso
    if pitch:
        angle = np.deg2rad(pitch)
        rotation = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],
                             [0,np.sin(angle),np.cos(angle)]])
        points = points @ rotation.T
    hip = (points[11] + points[12]) / 2
    shoulder = (points[5] + points[6]) / 2
    up = unit(shoulder - hip)
    lateral = unit(points[5] - points[6])
    # COCO left means the subject's anatomical left. In a frontal image it
    # points screen-right, so front points toward -Z for this camera.
    forward = unit(np.cross(up, lateral))
    lateral = unit(np.cross(forward, up))
    frame = np.column_stack((lateral, up, forward))
    objects = []
    skin = np.array([218, 178, 132.])
    shirt = np.array([95, 133, 170.])
    pants = np.array([70, 79, 99.])

    def ellipsoid(center, radii, basis, color):
        objects.append((np.asarray(center), np.asarray(radii), basis, color))

    def limb(a, b, radius, color):
        axis = unit(b - a)
        helper = np.array([1., 0, 0]) if abs(axis[0]) < .9 else np.array([0., 1, 0])
        cross = unit(np.cross(axis, helper))
        basis = np.column_stack((cross, axis, np.cross(cross, axis)))
        ellipsoid((a+b)/2, (radius, np.linalg.norm(b-a)/2+radius*.45, radius), basis, color)

    width = np.linalg.norm(points[5]-points[6]) / 2
    ellipsoid(hip+up*.56, (max(.25, width), .57, .22), frame, shirt)
    ellipsoid(hip+up*.02, (.30, .23, .21), frame, pants)
    for a, b in ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)):
        leg = a >= 11
        radius = (.13 if a in (11,12) else .10) if leg else .085
        color = pants if leg else skin
        limb(points[a], points[b], radius, color)
        ellipsoid(points[b], (radius, radius, radius), np.eye(3), color)
    for wrist in (9,10):
        ellipsoid(points[wrist], (.105,.13,.075), frame, skin)
    for ankle in (15,16):
        ellipsoid(points[ankle]+forward*.10, (.105,.09,.22), frame, pants)
    head = points[0]
    limb(shoulder, head, .11, skin)
    ellipsoid(head, (.22,.29,.21), frame, skin)
    # Small facial geometry provides an orientation cue without identity text.
    ellipsoid(head+forward*.215, (.045,.065,.07), frame, skin)
    for side in (-1,1):
        ellipsoid(head+forward*.19+up*.07+lateral*side*.083,
                  (.028,.018,.022), frame, np.array([48.,43,39]))

    # Conservative bounds include the actual rendered volume, not only joints.
    extent = [(c-np.abs(b)@r, c+np.abs(b)@r) for c,r,b,_ in objects]
    lo = np.min([v[0] for v in extent], axis=0)
    hi = np.max([v[1] for v in extent], axis=0)
    if distance is not None:
        # Project conservative 3D bounding-box corners, not an existing image.
        if distance <= 0 or lo[2]+distance <= .1:
            raise ValueError("camera intersects rendered volume")
        corners = np.array([[x,y,z] for x in (lo[0],hi[0])
                            for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        bounds = corners[:,:2]/(corners[:,2,None]+distance)
        lo[:2],hi[:2] = bounds.min(axis=0),bounds.max(axis=0)
    scale = (size-80)/max(hi[0]-lo[0], hi[1]-lo[1])
    cx, cy = (lo[:2]+hi[:2])/2
    yy, xx = np.mgrid[:size, :size]
    wx = (xx-size/2)/scale+cx
    wy = -(yy-size/2)/scale+cy
    depth = np.full((size,size), np.inf)
    surface_ids = np.full((size,size), -1, dtype=np.int16)
    canvas = np.full((size,size,3), 246., dtype=float)
    light = unit(np.array([-.5,.7,-1.]))
    for object_id, (center, radii, basis, color) in enumerate(objects):
        extent = np.abs(basis)@radii
        if distance is None:
            x0 = max(0, int((center[0]-extent[0]-cx)*scale+size/2)-2)
            x1 = min(size, int((center[0]+extent[0]-cx)*scale+size/2)+3)
            y0 = max(0, int(-(center[1]+extent[1]-cy)*scale+size/2)-2)
            y1 = min(size, int(-(center[1]-extent[1]-cy)*scale+size/2)+3)
        else:
            corners = np.array([[x,y,z] for x in (center[0]-extent[0],center[0]+extent[0])
                                for y in (center[1]-extent[1],center[1]+extent[1])
                                for z in (center[2]-extent[2],center[2]+extent[2])])
            bounds = corners[:,:2]/(corners[:,2,None]+distance)
            left,bottom = bounds.min(axis=0); right,top = bounds.max(axis=0)
            x0=max(0,int((left-cx)*scale+size/2)-2)
            x1=min(size,int((right-cx)*scale+size/2)+3)
            y0=max(0,int(-(top-cy)*scale+size/2)-2)
            y1=min(size,int(-(bottom-cy)*scale+size/2)+3)
        if x0>=x1 or y0>=y1:
            continue
        q = basis@np.diag(1/radii**2)@basis.T
        if distance is None:
            dx = wx[y0:y1,x0:x1]-center[0]
            dy = wy[y0:y1,x0:x1]-center[1]
            linear = q[2,0]*dx+q[2,1]*dy
            constant = q[0,0]*dx**2+2*q[0,1]*dx*dy+q[1,1]*dy**2-1
            discriminant = linear**2-q[2,2]*constant
            dz = (-linear-np.sqrt(np.maximum(discriminant,0)))/q[2,2]
            z = dz+center[2]
            visible = (discriminant>=0)&(z<depth[y0:y1,x0:x1])
            normals = np.stack((dx,dy,dz),axis=-1)@q.T
        else:
            rays = np.stack((wx[y0:y1,x0:x1],wy[y0:y1,x0:x1],
                             np.ones((y1-y0,x1-x0))),axis=-1)
            origin = np.array([0.,0.,-distance])-center
            quadratic = np.einsum('...i,ij,...j->...',rays,q,rays)
            linear = rays @ q @ origin
            discriminant = linear**2-quadratic*(origin @ q @ origin-1)
            ray_depth = (-linear-np.sqrt(np.maximum(discriminant,0)))/quadratic
            z=ray_depth-distance
            visible=(discriminant>=0)&(ray_depth>.1)&(z<depth[y0:y1,x0:x1])
            normals=(origin+rays*ray_depth[...,None])@q.T
        normals /= np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-8)
        diffuse = np.maximum(normals@light,0)
        rgb = np.clip(color*(.48+.52*diffuse[...,None]),0,255)
        canvas[y0:y1,x0:x1][visible] = rgb[visible]
        depth[y0:y1,x0:x1][visible] = z[visible]
        surface_ids[y0:y1,x0:x1][visible] = object_id
    plane=points[:,:2] if distance is None else points[:,:2]/(points[:,2,None]+distance)
    projected = np.column_stack(((plane[:,0]-cx)*scale+size/2,
                                  -(plane[:,1]-cy)*scale+size/2))
    if style == "rough":
        # Generate vector pen strokes directly from BVH-derived surface geometry.
        # No existing image is filtered, traced by a model, or used as input.
        # Depth-buffer boundaries retain occlusion; only the stroke varies.
        import cv2
        drawing = Image.new("RGB", (size*2,size*2), "white")
        pen = ImageDraw.Draw(drawing)
        rng = np.random.default_rng(seed)
        for object_id in range(len(objects)):
            contours, _ = cv2.findContours(
                (surface_ids==object_id).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour)<3:
                    continue
                path = cv2.approxPolyDP(contour,.65,True).reshape(-1,2).astype(float)
                path = np.concatenate((path,path[:1]))
                pen.line([tuple(v*2) for v in path],fill=(80,80,80),width=2)
                # Subpixel duplicate strokes imitate the inspected roughs;
                # the underlying projected joint coordinates never change.
                if len(path)>8:
                    jitter = rng.normal(0,.65,path.shape)
                    pen.line([tuple(v*2) for v in path+jitter],fill=(175,175,175),width=1)
        return drawing.resize((size,size),Image.Resampling.LANCZOS), projected
    return Image.fromarray(canvas.astype(np.uint8)), projected
