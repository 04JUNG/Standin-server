"""Output a known neutral bust for explicit close-ups, not a pose match.

No hidden limbs are inferred. Coarse view is a camera recommendation; 2D eye
roll is an optional bounded neck adjustment, never a 3D reconstruction claim.
"""
import math

VIEWS={'front','profile_left','profile_right','three_quarter_left','three_quarter_right','back'}

# Temporarily paused pending further experiments. Do not enable via deployment env.
BASIC_CLOSEUP_ENABLED = False


def basic_closeup_plan(evidence, description):
    if not BASIC_CLOSEUP_ENABLED:
        return None
    desc=description or {}
    scope=evidence.get('output_scope')
    if scope not in ('head','shoulder_head'):
        return None
    # Tight framing must not silently remove a visibly active hand or arm.
    if desc.get('visible_scope') in ('half','full') or desc.get('pose_relations') or desc.get('held_object_hands'):
        return None
    view=desc.get('head_view');view_known=view in VIEWS
    roll=None
    points=evidence.get('keypoints');mask=evidence.get('joint_mask')
    points=[] if points is None else points
    mask=[] if mask is None else mask
    if len(points)==17 and len(mask)==17 and mask[1] and mask[2]:
        a,b=sorted((points[1],points[2]),key=lambda p:p[0])
        dx,dy=float(b[0]-a[0]),float(b[1]-a[1])
        if math.isfinite(dx) and math.isfinite(dy) and dx>=8:
            angle=math.degrees(math.atan2(dy,dx))
            if abs(angle)<=25:roll=angle
    return dict(kind='basic_shoulder_head',template='standin-master-v2',output_scope='shoulder_head',
        status='planned',artifact_status='not_rendered',search_match=False,generated_pose=False,
        visible_scope_source=evidence.get('output_scope_source','explicit_output_scope'),
        camera_view=view if view_known else 'front',camera_source='vlm_coarse_direction' if view_known else 'neutral_default',
        head_roll_degrees=roll if roll is not None else 0.,head_roll_source='observed_eye_line_2d' if roll is not None else 'neutral_default',
        head_pitch_degrees=0.,head_pitch_source='neutral_default',
        exact_pose_matched=False,refine_allowed=False,release_allowed=False,
        remaining_uncertainty=['Coarse camera and bounded eye-line roll only; neck yaw/pitch and shoulder slope are not reconstructed.'])
