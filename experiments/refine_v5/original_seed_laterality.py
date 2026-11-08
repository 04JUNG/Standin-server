"""Ground VLM observations to neutral detector guides before binding anatomical sides."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.learning_data import digest
from experiments.refine_v5.original_seed_observe import observation_schema, setup
from experiments.refine_v5.original_seed_prior import checked_seed
from experiments.refine_v5.part_vlm_observe import CHAINS, enum, object_schema, observe, schema


IDS = ('A', 'B', 'C', 'D')
COLORS = {'A': '#bd2658', 'B': '#006dce', 'C': '#20723d', 'D': '#8e43a2'}
OBSERVATION_PROTOCOL = 'attachment-visibility-hand-relations-v5'
DEFAULT_OBSERVATION_STAGE = 2
ARM_REGIONS = ('shoulder', 'upper_arm', 'elbow', 'forearm', 'wrist', 'hand', 'fingers')
LEG_REGIONS = ('hip', 'thigh', 'knee', 'shin', 'calf', 'ankle', 'foot', 'toes')
BODY_REGIONS = ('head', 'face', 'ear', 'neck', 'chest', 'upper_back', 'abdomen',
                'waist', 'lower_back', 'pelvis', 'buttocks')
RELATION_DIRECTIONS = ('front', 'side', 'back', 'unknown')
PROMPT = """Observe the unmarked rough on the left. The right panel shows detector guides
on the same image. Report A, B, C and D separately even if some are ambiguous.
A/B originate at shoulders and C/D at hips. Use these IDs throughout. The app
assigns anatomy. For attachment_screen, use only positions within the unmarked
drawing. For each shoulder, compare its x coordinate with the other shoulder.
For each hip, compare its x coordinate with the other hip. Report smaller_x
(viewer's left) or larger_x (viewer's right) for each verified attachment.
Two distinct attachments of the same kind cannot have the same x ordering.
If overlap prevents comparison, use unknown. Ignore wrist and ankle positions.

Trace each actual limb from its attachment. Detector dots may be wrong.
Confirm only when the root and the elbow/knee can be followed on one chain.
Use conflict when the guide jumps chains. For joint_marker and endpoint_marker
report the letter near the matching elbow/knee and wrist/ankle respectively.
Use unknown if the marker cannot be verified. Classify each joint and endpoint
independently as visible (direct contour or stroke), inferable (hidden but its
position can be localized from the visible segments), or unknown. Confidence
describes the visual support for the reported pose cues: strong for clear
evidence, moderate for usable but indirect evidence, unknown if none is usable.
Do not use clear joint visibility as evidence of depth.

Give true elbow/knee flexion bounds in degrees (0 straight, 90 right angle),
or two nulls if unresolved. For each segment report forward/backward from
proximal to distal, relative to the chest's forward axis for arms and the
pelvis's forward axis for legs. Proximal is shoulder to elbow or hip to knee.
Distal is elbow to wrist or knee to ankle. Use aligned only with evidence
of little front/back change; use unknown when the axis or direction is
unclear. Report proximal_depth_evidence and distal_depth_evidence separately:
clear (a readable depth cue such as occlusion or foreshortening supports the
direction), indirect (a weaker cue), or unknown (no usable depth cue). A
projected overlap alone supplies no direction. For thigh_height
use gravity/world up relative to the hip: down is nearly vertical toward
ground, diagonal_down is slanted below hip, level is near horizontal, above
rises from hip. Use unknown when the world vertical or thigh is unclear.

For each leg report foot support independently as planted (clearly bearing
on a support surface), lifted (clearly off support), or unknown. Independently
name raised_guide if exactly one knee is visibly raised in the pose. A raised
knee and an unsupported foot are separate observations. Name support_guide
if exactly one foot bears weight. Identify the guide by tracing from its hip.
Use unknown when the role is ambiguous, both qualify or neither qualifies.
A background line is not a support surface.

For each arm guide report hand_relations, one entry per relevant body region.
Include regions the hand touches or plausibly approaches, overlaps or is
occluded by. Use [] if none is relevant. Put the best-supported relation
first. For a paired limb region name its attachment guide in target_guide
(A/B for arm regions, C/D for leg regions). Use unknown if the attachment
cannot be traced. For head and torso regions use none. Use other only for
a body region not in the list and name it in target_detail; otherwise use "".

For each entry, report position as front, side, back or unknown. Position
describes the hand relative to the target, even when they do not touch.
Use the face direction as forward for the head, chest direction for the
upper torso and arms, and pelvis direction for the lower torso and legs.
Use side for a lateral placement, not just overlap on the page. The hand's
position relative to a body region and the direction of the forearm are
separate observations. For contact use touching, separated or unknown. If
touching, report contact_surface as front (anterior), side (lateral), back
(posterior) or unknown, based on where the hand itself meets the surface.
A hand on the waist flank may have fingers curling over the abdomen; the
palm's contact there is still side. Use unknown for contact_surface when
contact or its surface cannot be seen. Overlap alone proves neither depth
nor contact. Describe the contour, occlusion or gap that supports each entry.

Report torso_view and head yaw, pitch and lean independently. Head left/right
means the direction in the image. Use unknown for unreadable axes.
"""

def experiment_prompt(stage):
    prompt = PROMPT.replace(
        'clear (a readable depth cue such as occlusion or foreshortening supports the\ndirection), indirect (a weaker cue), or unknown (no usable depth cue).',
        'clear only if the relevant body forward axis and the signed direction between\nthe endpoints are both readable; indirect for weaker evidence that still fixes\nthe sign; unknown otherwise. Foreshortening alone cannot choose forward or backward.')
    prompt = prompt.replace(
        'or two nulls if unresolved.',
        'Widen the interval when overlap or foreshortening hides the true angle.\nReturn two nulls if the range cannot be narrowed meaningfully.')
    prompt = prompt.replace(
        'Report torso_view and head yaw, pitch and lean independently. Head left/right\nmeans the direction in the image. Use unknown for unreadable axes.',
        'Report torso_view and head yaw, pitch and lean independently. Yaw is the\nscreen direction the face points. Lean is where the crown tilts relative to\nthe chin in the image. Pitch comes from the face itself. Use level or upright\nonly with visible head evidence; do not infer from the neck or torso. Use\nunknown for unreadable axes.')
    if stage >= 2:
        start = prompt.index('A/B originate at shoulders')
        end = prompt.index('Trace each actual limb', start)
        prompt = prompt[:start] + (
            'A/B originate at shoulders and C/D at hips. Use only guide IDs for limb\n'
            'identity. Trace each limb from the printed guide root in the overlay.\n\n'
        ) + prompt[end:]
        prompt = prompt.replace(
            'Use conflict when the guide jumps chains.',
            'Use conflict when the guide cannot be traced to a single actual limb. '
            'For a misplaced dot, mark that point unknown and keep the trace if readable.')
        start = prompt.index('For each leg report foot support')
        end = prompt.index('For each arm guide', start)
        prompt = prompt[:start] + """For each leg report ground_contact as touching, off_ground or unknown, and
weight_bearing independently as bearing, not_bearing or unknown. A flat foot
does not establish load. Use a visible support surface and evidence of the
body loading the leg. A background line is not a support surface. Report
unknown when load or ground contact is unclear. The app derives unique support.
Independently report raised_guide only for a uniquely raised knee and identify
it by tracing from the hip. Use unknown when neither or both qualify.

""" + prompt[end:]
    if stage >= 3:
        start = prompt.index('separate observations. For contact') + len('separate observations. ')
        end = prompt.index('Report torso_view', start)
        prompt = prompt[:start] + """Report spacing independently of position:
- palm_touching: the drawing clearly shows the palm meeting the target.
- near: traced anatomy, surface contours or occlusion order support a small
  3D gap or possible contact, but palm contact is not established.
- separated: visible geometry supports a distinct gap in space.
- unknown: the gap cannot be inferred.
Projected overlap or a hidden hand alone does not establish near or contact.
A hand hidden behind the head can still have back position with unknown spacing.
Only for palm_touching, give contact_surface (front, side, back or unknown)
on the target itself. Fingers curling toward the abdomen do not change a
visible palm contact on the waist flank from side to front. Otherwise set
contact_surface to unknown. Describe the actual visual evidence.

""" + prompt[end:]
    return prompt



def guide_digest(seed):
    fields = [seed['keypoints'], seed['scores']]
    return hashlib.sha256(json.dumps(fields, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def neutral_layout(seed):
    if seed is None:
        return {}
    kp = np.asarray(seed['keypoints'], float)
    layout = {}
    for kind, ids in (('arm', IDS[:2]), ('leg', IDS[2:])):
        names = sorted((f'left_{kind}', f'right_{kind}'),
                       key=lambda name: (kp[CHAINS[name][1][0], 0], name))
        layout.update(zip(ids, names, strict=True))
    return layout


def harness_schema(stage=0):
    limb = dict(schema()['properties']['left_arm']['properties'])
    del limb['contact']
    marker = enum((*IDS, 'unknown'))
    point_visibility = enum(('visible', 'inferable', 'unknown'))
    limb.update(trace=enum(('confirmed', 'conflict', 'unknown')),
                attachment_screen=enum(('smaller_x', 'larger_x', 'unknown')),
                proximal_depth_evidence=enum(('clear', 'indirect', 'unknown')),
                distal_depth_evidence=enum(('clear', 'indirect', 'unknown')),
                joint=point_visibility, endpoint=point_visibility,
                endpoint_marker=marker, joint_marker=marker)
    relation = object_schema(dict(
        target_region=enum((*BODY_REGIONS, *ARM_REGIONS, *LEG_REGIONS, 'other')),
        target_detail={'type': 'string'},
        target_guide=enum((*IDS, 'none', 'unknown')),
        position=enum(RELATION_DIRECTIONS),
        contact=enum(('touching', 'separated', 'unknown')),
        contact_surface=enum(RELATION_DIRECTIONS),
        evidence={'type': 'string'},
    ))
    if stage >= 3:
        del relation['properties']['contact']
        relation['required'].remove('contact')
        relation['properties']['spacing'] = enum(('palm_touching', 'near', 'separated', 'unknown'))
        relation['required'].append('spacing')
    if stage >= 2:
        del limb['support']
        del limb['attachment_screen']
        leg = object_schema(dict(limb, ground_contact=enum(('touching', 'off_ground', 'unknown')),
                                 weight_bearing=enum(('bearing', 'not_bearing', 'unknown'))))
    else:
        leg = object_schema(limb)
    arm = object_schema(dict(limb, hand_relations={'type': 'array', 'items': relation}))
    prior = observation_schema()['properties']
    result = dict(guides=object_schema(dict(A=arm, B=arm, C=leg, D=leg)),
                  raised_guide=enum(('C', 'D', 'unknown')))
    if stage < 2:
        result['support_guide'] = enum(('C', 'D', 'unknown'))
    result.update(torso_view=prior['torso_view'], head=prior['head'])
    return object_schema(result)


def guide_image(image, seed, layout, original_size):
    width, height = image.size
    title = 29
    canvas = Image.new('RGB', (2 * width + 10, height + title), 'white')
    canvas.paste(image, (0, title))
    canvas.paste(image, (width + 10, title))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 16)
    draw.text((6, 4), 'Rough', font=font, fill='#202938')
    draw.text((width + 16, 4), 'Detector guides: verify against rough', font=font, fill='#202938')
    for code, limb in layout.items():
        indices = list(CHAINS[limb][1])
        pixels = np.asarray(seed['keypoints'])[indices] * (np.asarray([width, height]) / original_size)
        points = [tuple(p + [width + 10, title]) for p in pixels]
        color = COLORS[code]
        draw.line(points, fill=color, width=3)
        for index, (x, y) in enumerate(points):
            if not width + 10 <= x < 2 * width + 10 or not title <= y < height + title:
                continue
            radius = 5 if index == 0 else 3
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline='white', width=1)
            draw.text((x + 5, y - 9), code, font=font, fill=color, stroke_width=2, stroke_fill='white')
    return canvas


def bound_goals(raw, seed, recorded_layout, *, stage=0):
    layout = neutral_layout(seed)
    if layout != recorded_layout:
        raise ValueError('detector guide assignment changed')
    view = raw['torso_view']
    cues = dict(sides='resolved', torso_view=view, head=raw['head'])
    points = np.asarray(seed['keypoints'], float)
    scores = np.asarray(seed['scores'], float)
    torso_px = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    conflicting_roots, verified_roots = set(), set()
    if view in ('facing_camera', 'facing_away') and torso_px > 1.:
        for kind, left, right in (('arm', 5, 6), ('leg', 11, 12)):
            dx = points[left, 0] - points[right, 0]
            if min(scores[left], scores[right]) >= .40 and abs(dx) > .12 * torso_px:
                if (dx > 0) != (view == 'facing_camera'):
                    conflicting_roots.add(kind)
                else:
                    verified_roots.add(kind)
    statuses = {}
    for code, name in layout.items():
        cue = dict(raw['guides'][code])
        _, indices = CHAINS[name]
        score = scores[list(indices)]
        marker = cue['endpoint_marker']
        joint_marker = cue['joint_marker']
        status = 'bound'
        anatomical_side, kind = name.split('_')
        screen_side = 'smaller_x' if code in ('A', 'C') else 'larger_x'
        if kind in conflicting_roots:
            status = 'anatomical_anchor_conflict'
        elif stage >= 2 and kind not in verified_roots:
            status = 'anatomical_side_unverified'
        elif stage < 2 and cue.get('attachment_screen', 'unknown') not in ('unknown', screen_side):
            status = 'attachment_screen_conflict'
        elif cue['trace'] != 'confirmed' or cue['confidence'] == 'unknown':
            status = 'trace_unverified'
        elif stage < 2 and any(mark not in (code, 'unknown') for mark in (marker, joint_marker)):
            status = 'guide_identity_conflict'
        elif score[0] < .40 or not (
            (marker == code and cue['endpoint'] == 'visible' and score[2] >= .40)
            or (joint_marker == code and cue['joint'] == 'visible' and score[1] >= .42)
        ):
            status = 'detector_anchor_unverified'
        statuses[name] = status
        cue['anatomical_side'] = anatomical_side if status == 'bound' else 'uncertain'
        if stage >= 2 and status == 'bound':
            cue['attachment_screen'] = screen_side
        if status != 'bound':
            cue['confidence'] = 'unknown'
        else:
            if marker != code:
                cue['endpoint'] = 'unknown'
            if joint_marker != code:
                cue['joint'] = 'unknown'
        if stage >= 1:
            low, high = cue['bend_low_deg'], cue['bend_high_deg']
            if low is None or high is None or not 0 <= low < high <= 180:
                cue['bend_low_deg'] = cue['bend_high_deg'] = None
        if stage >= 2:
            ground, load = cue.get('ground_contact'), cue.get('weight_bearing')
            if ground == 'off_ground' and load == 'bearing':
                cue['weight_bearing'] = load = 'unknown'
            cue['support'] = 'planted' if load == 'bearing' else 'lifted' if ground == 'off_ground' else 'unknown'
        cue.setdefault('contact', 'unknown')
        cues[name] = cue
    for code in ('A', 'B'):
        name = layout[code]
        cue = cues[name]
        if 'hand_relations' in cue:
            cue['hand_relations'] = bind_hand_relations(cue['hand_relations'], layout, statuses, stage=stage)
            set_legacy_contact(cue)
    return cues, statuses


def bind_hand_relations(relations, layout, statuses, *, stage=0):
    result = []
    for relation in relations:
        target = relation['target_region']
        guide = relation['target_guide']
        paired_guides = IDS[:2] if target in ARM_REGIONS else IDS[2:] if target in LEG_REGIONS else ()
        target_limb = None
        if paired_guides:
            target_limb = layout[guide] if guide in paired_guides else 'unknown'
            if statuses.get(target_limb) != 'bound':
                target_limb = 'unknown'
        if 'spacing' in relation:
            spacing = relation['spacing']
            contact = 'touching' if spacing == 'palm_touching' else spacing
        else:
            contact = relation['contact']
            if stage >= 3 and contact == 'touching' and relation.get('palm_evidence') != 'visible_palm_contact':
                contact = 'unknown'
        surface = relation['contact_surface'] if contact == 'touching' else 'unknown'
        normalized = dict(relation, target_limb=target_limb, contact=contact, contact_surface=surface)
        if stage >= 1 and target != 'other':
            normalized['target_detail'] = ''
        if stage >= 1 and not paired_guides:
            normalized['target_guide'] = 'none'
        result.append(normalized)
    return result


def set_legacy_contact(cue):
    relations = cue['hand_relations']
    for relation in relations:
        if relation['contact'] != 'touching':
            continue
        region = relation['target_region']
        if region in ('head', 'face', 'ear'):
            cue['contact'] = 'head'
            return
        if region == 'waist' and relation['contact_surface'] != 'back':
            cue['contact'] = 'waist'
            cue['waist_surface'] = relation['contact_surface']
            return
    if any(r['target_region'] == 'waist' and r['contact'] == 'separated'
           and 'spacing' not in r for r in relations):
        cue['contact'] = 'clear_of_waist'


def leg_roles(raw, layout, cues, statuses, seed):
    raised = raw['raised_guide']
    if all('weight_bearing' in raw['guides'][code] for code in ('C', 'D')):
        names = [layout[code] for code in ('C', 'D')]
        if all(statuses[name] == 'bound' for name in names):
            loads = [cues[name]['weight_bearing'] for name in names]
            if sorted(loads) == ['bearing', 'not_bearing']:
                return 'independent_roles', names[loads.index('bearing')].split('_')[0]
        return 'independent_roles', None
    support = raw['support_guide']
    if all('attachment_screen' in raw['guides'][code] for code in ('C', 'D')):
        support_name = layout.get(support)
        if support_name and statuses[support_name] == 'bound' and cues[support_name]['support'] == 'planted':
            other = layout['D' if support == 'C' else 'C']
            if cues[other]['support'] != 'planted' or statuses[other] != 'bound':
                return 'independent_roles', support_name.split('_')[0]
        return 'independent_roles', None
    if raised == support == 'unknown':
        if any(cues[name]['support'] == 'lifted' for name in ('left_leg', 'right_leg')):
            return 'raise_unverified', None
        return 'no_unique_raise', None
    if raised == 'unknown' and support in layout and cues[layout[support]]['support'] == 'planted':
        return 'unique_support_unconfirmed_raise', None
    if raised not in layout or support not in layout or raised == support:
        return 'screen_roles_ambiguous', None
    raised_name, support_name = layout[raised], layout[support]
    if (statuses[raised_name] != 'bound' or statuses[support_name] != 'bound'
            or cues[raised_name]['support'] != 'lifted' or cues[support_name]['support'] != 'planted'):
        return 'guide_roles_disagree', None
    # A unique support assignment needs independent endpoint geometry. The
    # overlay can make two almost coincident ankles look like distinct evidence.
    points = np.asarray(seed['keypoints'], float)
    scores = np.asarray(seed['scores'], float)
    raised_ankle = CHAINS[raised_name][1][-1]
    support_ankle = CHAINS[support_name][1][-1]
    torso_px = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    separation = points[support_ankle, 1] - points[raised_ankle, 1]
    if min(scores[raised_ankle], scores[support_ankle]) < .40 or separation < max(2., .05 * torso_px):
        return 'ankle_order_unverified', None
    return 'screen_roles_agree', support_name.split('_')[0]


def run(p1, out, seeds_path, env_file, case_id=None, *, stage=DEFAULT_OBSERVATION_STAGE):
    protocol = f'{OBSERVATION_PROTOCOL}-stage-{stage}' if stage else OBSERVATION_PROTOCOL
    if stage >= 2:
        protocol += '-guide-identity'
    if stage >= 3:
        protocol += '-hand-spacing-v1'
    instructions = experiment_prompt(stage) if stage else PROMPT
    rows = setup(p1, out)
    if case_id is not None:
        rows = [row for row in rows if row['case_id'] == case_id]
        if not rows:
            raise ValueError(f'{case_id}: not in validation cohort')
    seeds = {row['case_id']: row for row in load_json(seeds_path)}
    config = dotenv_values(env_file)
    key = os.environ.get('OPENAI_API_KEY') or config.get('OPENAI_API_KEY')
    model = os.environ.get('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'gpt-5-mini'
    if not key:
        raise ValueError('OpenAI credential unavailable')

    def call(row):
        cid = row['case_id']
        path = p1 / row['rough']
        output = out / 'observations' / f'{cid}.json'
        seed = checked_seed(row, seeds)
        if seed is None:
            save_json(output, dict(case_id=cid, rough_sha256=row['rough_sha256'],
                                   anchor_harness={}, goals=None, model=None, response_id=None))
            return
        with Image.open(path) as image:
            size = list(image.size)
        layout = neutral_layout(seed)
        guides_hash = guide_digest(seed)
        if output.exists():
            previous = load_json(output)
            if (previous.get('protocol') != protocol or previous['rough_sha256'] != row['rough_sha256'] or previous.get('anchor_harness') != layout
                    or previous.get('guide_sha256') != guides_hash):
                raise ValueError('cached guide observation does not match input')
            if previous['goals'] is not None:
                return
            incomplete = out / 'incomplete-observations'
            incomplete.mkdir(exist_ok=True)
            output.replace(incomplete / f"{cid}-{previous['response_id']}.json")
        item = dict(case_id=cid, image=str(path), image_sha256=row['rough_sha256'], dimensions=size)
        result = observe(item, output, model, key, instructions=instructions, response_schema=lambda: harness_schema(stage),
                         image_transform=lambda image: guide_image(image, seed, layout, size),
                         reasoning_effort='medium', max_output_tokens=6500)
        save_json(output, dict(result, anchor_harness=layout, guide_sha256=guides_hash, protocol=protocol, observation_stage=stage))

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(call, rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'out', 'seeds', 'env-file'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--case-id')
    parser.add_argument('--stage', type=int, choices=(0, 1, 2, 3), default=DEFAULT_OBSERVATION_STAGE,
                        help='Observation protocol stage. Default 2 uses guide identity and foot load.')
    args = parser.parse_args()
    run(args.p1, args.out, args.seeds, args.env_file, args.case_id, stage=args.stage)
