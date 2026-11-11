"""Local Korean leg relationship parsing; separate intent from measured pose.

This does not infer anatomy from pixels. Side/overlap inferred from a supplied
sentence remains language evidence until a reviewer/geometry confirms it.
"""
import re
import numpy as np


def parse_leg_language(text):
    s=re.sub(r'\s+','',str(text)).lower()
    negative=bool(re.search(r'(꼬지않|안꼬|교차하지않|겹치지않)',s))
    cross=bool(re.search(r'(꼬|꼰|꽈|교차|겹)',s)) and not negative
    region='ankle' if '발목' in s else 'knee' if re.search(r'무릎(?:위|에서|끼리)',s) else 'unspecified'
    posture='airborne' if any(t in s for t in ['공중','날아','떠있는']) else 'standing' if any(t in s for t in ['서서','서있는']) else 'seated' if any(t in s for t in ['앉','책상','의자','벤치']) else 'unspecified'
    top=None
    # X를 Y 위에: first side is above, not the side next to '위'.
    m=re.search(r'(왼|오른)(?:쪽)?(?:다리|발목|발)(?:을|를)(왼|오른)(?:쪽)?(?:다리|발목|발)위',s)
    if m:
        if m[1]!=m[2]:top='left' if m[1]=='왼' else 'right'
    else:
        m=re.search(r'(왼|오른)(?:쪽)?(?:다리|발목|발)(?:을|를|이|가)?위(?:로|에|쪽)?',s)
        if m:top='left' if m[1]=='왼' else 'right'
    m=re.search(r'(왼|오른)(?:쪽)?(?:다리|발목|발)(?:을|를|이|가)?아래',s)
    bottom=('left' if m[1]=='왼' else 'right') if m else None
    if top and bottom==top:raise ValueError('same leg cannot be both above and below')
    if top and not bottom:bottom='right' if top=='left' else 'left'
    if bottom and not top:top='right' if bottom=='left' else 'left'
    return {'crossing':False if negative else True if cross else None,
            'cross_region':region if cross else 'unspecified','posture':posture,
            'forward_extended':bool(re.search(r'(앞으로|앞쪽으로|길게|쭉).*(뻗|편다리|펴)',s)),
            'elevated':any(t in s for t in ['책상','높게','위로올','들어올']),
            'top_side':top,'bottom_side':bottom,'side_reference':'subject_anatomical',
            'cross_style':'interlaced_x' if re.search(r'(감아|감기는|휘감|x자|엑스자)',s) else 'unspecified',
            'side_evidence':'explicit_sentence_only' if top else 'unspecified_not_guessed'}


def projected_crossing_order(positions,camera_matrix,depth_tolerance_m=.003):
    """Determine occluding anatomical leg at projected centerline crossings.

    Camera local +Z is toward the viewer. Image height alone cannot establish
    which shin covers the other in a rotated airborne pose.
    """
    camera=np.asarray(camera_matrix,float);rotation=camera[:3,:3]
    points={side:[rotation.T@(np.asarray(positions[role+'.'+side])-camera[:3,3])
                  for role in ('upleg','leg','foot')] for side in ('L','R')}
    crossings=[]
    for i in range(2):
        for j in range(2):
            a,b=points['L'][i:i+2];c,d=points['R'][j:j+2]
            mat=np.column_stack([(b-a)[:2],-(d-c)[:2]])
            if abs(np.linalg.det(mat))<1e-10:continue
            u,v=np.linalg.solve(mat,(c-a)[:2])
            if not (0<u<1 and 0<v<1):continue
            gap=float((c+v*(d-c))[2]-(a+u*(b-a))[2])
            crossings.append({'left_segment':i,'right_segment':j,'right_toward_viewer_m':gap,
                              'top_side':'right' if gap>depth_tolerance_m else 'left' if gap < -depth_tolerance_m else None,
                              'left_parameter':float(u),'right_parameter':float(v)})
    sides={x['top_side'] for x in crossings}
    return {'crossings':crossings,'top_side':next(iter(sides)) if len(sides)==1 else None,
            'basis':'reference_camera_depth_at_projected_crossing'}


def require_leg_order(intent,measurement):
    """Reject wrong or unverified order before accepting the pose's label."""
    expected=intent.get('top_side')
    if expected not in ('left','right'):raise ValueError('explicit anatomical top side required')
    bottom='left' if expected=='right' else 'right'
    if intent.get('bottom_side',bottom)!=bottom:raise ValueError('invalid anatomical lower side')
    if measurement.get('top_side')!=expected:raise ValueError('crossed leg order does not match reference')
    return {'requested_top_side':expected,'measured_top_side':measurement['top_side'],
            'requested_bottom_side':bottom,'measured_bottom_side':bottom,
            'side_reference':'subject_anatomical','passed':True,'measurement':measurement}


def relation_matches(query,metadata):
    """Hard relationship constraints prevent ankle-cross queries matching parallel legs."""
    q=parse_leg_language(query) if isinstance(query,str) else query
    for key in ['crossing','top_side']:
        if q.get(key) is not None and metadata.get(key)!=q[key]:return False
    for key in ['cross_region','posture','cross_style']:
        if q.get(key,'unspecified')!='unspecified' and metadata.get(key)!=q[key]:return False
    for key in ['forward_extended','elevated']:
        if q.get(key) and not metadata.get(key):return False
    return True
