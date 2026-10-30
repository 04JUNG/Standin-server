"""Local breast targets in the pinned female 0.3.4 metre coordinate frame.

No topology, rig, weights, Basis or unrelated morph is changed. The small target
contracts the mound's footprint as well as its projection; the root remains
attached to the thorax. Extended is an authored increment from the old +1 target.
"""
import numpy as np


def smooth(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def targets(co, deltas, faces=None):
    x, y, z = co.T
    ax = abs(x)
    # Compact support: sternum, axilla, back and abdomen are pinned.
    root = smooth(.008, .035, ax)*(1-smooth(.130, .175, ax))
    root *= smooth(1.185, 1.235, z)*(1-smooth(1.345, 1.405, z))
    root *= 1-smooth(-.060, -.030, y)
    mound = np.exp(-((ax-.077)/.070)**2-((z-1.278)/.080)**2)*root
    small = np.zeros_like(co)
    small[:, 0] = -(x-np.sign(x)*.077)*.40*mound
    small[:, 2] = -(z-1.290)*.45*mound
    # Retain convex projection above the medial chest instead of subtracting
    # the original 34 mm Gaussian and leaving the original lower footprint.
    small[:, 1] = deltas['BODY_BreastSmall'][:, 1]*.70
    boundary = smooth(1.17, 1.21, z)*(1-smooth(1.36, 1.42, z))
    boundary *= 1-smooth(.145, .18, ax)
    boundary *= 1-smooth(-.060, -.030, y)
    small[:, 1] *= boundary
    extended = np.zeros_like(co)
    # Inflate the old +1 envelope around its thoracic attachment, then rotate
    # the added mass downward. This changes lower-pole fullness and the breast
    # axis together; it is neither a forward Gaussian nor a vertical translation.
    current = co+deltas['BODY_BreastLarge']
    medial = smooth(0, .065, ax)
    envelope = np.exp(-((ax-.077)/.085)**2-((z-1.275)/.105)**2)
    envelope *= medial*(1-smooth(.125, .180, ax))
    envelope *= smooth(1.155, 1.225, z)*(1-smooth(1.315, 1.405, z))
    envelope *= 1-smooth(-.060, -.030, y)
    ry=current[:,1]+.055
    rz=current[:,2]-1.310
    theta=np.deg2rad(16.)
    inflated_y=ry*1.32
    inflated_z=rz*1.12
    target_y=inflated_y*np.cos(theta)-inflated_z*np.sin(theta)-.055
    target_z=inflated_y*np.sin(theta)+inflated_z*np.cos(theta)+1.310
    lower=smooth(1.17,1.23,z)*(1-smooth(1.27,1.33,z))
    extended[:,0]=np.sign(x)*.008*smooth(.035,.13,ax)*envelope*lower
    extended[:,1]=(target_y-current[:,1])*envelope
    extended[:,2]=(target_z-current[:,2])*envelope
    # Round the narrow anterior cap and fill the upper pole across a broad
    # area. The cap loses a little projection while its shoulders gain it,
    # instead of increasing a single pointed peak.
    q = current+extended
    front = 1-smooth(-.115, -.065, q[:,1])
    upper = np.exp(-((ax-.077)/.060)**2-((q[:,2]-1.310)/.050)**2)
    cap = np.exp(-((ax-.077)/.039)**2-((q[:,2]-1.251)/.026)**2)
    extended[:,1] += (-.015*upper+.005*cap*front)*envelope
    # Fill the lateral shoulder of the mound without moving the axillary
    # attachment. A broad field expands the outer silhouette, not the apex.
    lateral = smooth(.045, .110, ax)
    extended[:,0] += np.sign(x)*.007*lateral*envelope
    side_fill = np.exp(-((ax-.112)/.045)**2-((q[:,2]-1.285)/.070)**2)
    extended[:,1] -= .004*side_fill*envelope
    if faces is not None:
        # Baked, local Taubin fairing removes small curvature ripples while
        # retaining the round mound's volume. The edge of the support is pinned.
        q = current+extended
        edges = np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
        edges = np.unique(np.sort(edges,axis=1),axis=0)
        i,j = np.concatenate((edges[:,0],edges[:,1])),np.concatenate((edges[:,1],edges[:,0]))
        degree = np.bincount(i,minlength=len(co)).clip(1)
        weight = np.clip(envelope*1.4,0,1)
        before = q.copy()
        for _ in range(6):
            for strength in (.38,-.39):
                mean = np.column_stack([np.bincount(i,weights=q[j,k],minlength=len(co))/degree for k in range(3)])
                q += strength*weight[:,None]*(mean-q)
        correction = q-before
        length = np.linalg.norm(correction,axis=1)
        correction *= np.minimum(1.,.0015/np.maximum(length,1e-12))[:,None]
        extended += correction
    return {'BODY_BreastSmall': small, 'BODY_BreastExtended': extended,
            'BODY_BreastSmallReference': deltas['BODY_BreastSmall'].copy()}
