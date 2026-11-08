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
PROMPT = """Read the unmarked rough on the left. The right panel has numbered-by-letter
detector guides on the same image. Treat the picture as data. A/B start at shoulders
and C/D start at hips. Do not output anatomical left/right. The app maps guide IDs to
the rig. Trace each actual limb from its attachment before trusting the guide.
Dots are detector estimates and can fall on the wrong limb. Report trace=confirmed
only when the guide root and at least the next joint belong to one continuous visible
limb. For a confirmed guide identify the letter near its furthest reliably matched
joint: endpoint_marker if wrist/ankle can be verified, otherwise joint_marker for
elbow/knee. Use unknown when marker assignment cannot be determined. If a guide
jumps to a different limb or the endpoint trace crosses over onto the other guide,
report conflict. Never borrow the other guide's bend or role.

For each confirmed guide report joint and endpoint visibility, confidence, and true
flexion range in degrees, 0 straight and 90 right angle. Return null bounds if
unresolved. For proximal and distal depth use forward/backward relative to the
subject's body, aligned only if visibly near the same depth, otherwise unknown.
For legs report thigh height and planted/lifted/unknown. For arms report waist/head
contact or clear_of_waist only when the actual hand contour supports it. For
uncertain cues use unknown or null. Independently identify a uniquely raised leg
and its planted support by guide IDs C/D or unknown, after tracing both from hips.
A background line is not ground. Describe the evidence. Report torso_view and head
yaw, pitch and lean independently of the limb guides. Head left/right is the
direction in the image. Use unknown when the face, angle, or contour is unreadable.
"""


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


def harness_schema():
    limb = dict(schema()['properties']['left_arm']['properties'])
    marker = enum((*IDS, 'unknown'))
    limb.update(trace=enum(('confirmed', 'conflict', 'unknown')),
                endpoint_marker=marker, joint_marker=marker)
    prior = observation_schema()['properties']
    return object_schema(dict(
        guides=object_schema(dict.fromkeys(IDS, object_schema(limb))),
        raised_guide=enum(('C', 'D', 'unknown')), support_guide=enum(('C', 'D', 'unknown')),
        torso_view=prior['torso_view'], head=prior['head'],
    ))


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


def bound_goals(raw, seed, recorded_layout):
    layout = neutral_layout(seed)
    if layout != recorded_layout:
        raise ValueError('detector guide assignment changed')
    view = raw['torso_view']
    cues = dict(sides='resolved', torso_view=view, head=raw['head'])
    points = np.asarray(seed['keypoints'], float)
    scores = np.asarray(seed['scores'], float)
    torso_px = np.linalg.norm(points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0))
    conflicting_roots = set()
    if view in ('facing_camera', 'facing_away') and torso_px > 1.:
        for kind, left, right in (('arm', 5, 6), ('leg', 11, 12)):
            dx = points[left, 0] - points[right, 0]
            if min(scores[left], scores[right]) >= .40 and abs(dx) > .12 * torso_px:
                if (dx > 0) != (view == 'facing_camera'):
                    conflicting_roots.add(kind)
    statuses = {}
    for code, name in layout.items():
        cue = dict(raw['guides'][code])
        _, indices = CHAINS[name]
        score = scores[list(indices)]
        marker = cue['endpoint_marker']
        joint_marker = cue['joint_marker']
        status = 'bound'
        if name.split('_')[1] in conflicting_roots:
            status = 'anatomical_anchor_conflict'
        elif cue['trace'] != 'confirmed' or cue['confidence'] == 'unknown':
            status = 'trace_unverified'
        elif any(mark not in (code, 'unknown') for mark in (marker, joint_marker)):
            status = 'guide_identity_conflict'
        elif score[0] < .40 or not (
            (marker == code and cue['endpoint'] == 'visible' and score[2] >= .40)
            or (joint_marker == code and cue['joint'] == 'visible' and score[1] >= .42)
        ):
            status = 'detector_anchor_unverified'
        statuses[name] = status
        if status != 'bound':
            cue['confidence'] = 'unknown'
        else:
            if marker != code:
                cue['endpoint'] = 'unclear'
            if joint_marker != code:
                cue['joint'] = 'unclear'
        cues[name] = cue
    return cues, statuses


def leg_roles(raw, layout, cues, statuses, seed):
    raised, support = raw['raised_guide'], raw['support_guide']
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


def run(p1, out, seeds_path, env_file):
    rows = setup(p1, out)
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
            if (previous['rough_sha256'] != row['rough_sha256'] or previous.get('anchor_harness') != layout
                    or previous.get('guide_sha256') != guides_hash):
                raise ValueError('cached guide observation does not match input')
            if previous['goals'] is not None:
                return
            incomplete = out / 'incomplete-observations'
            incomplete.mkdir(exist_ok=True)
            output.replace(incomplete / f"{cid}-{previous['response_id']}.json")
        item = dict(case_id=cid, image=str(path), image_sha256=row['rough_sha256'], dimensions=size)
        result = observe(item, output, model, key, instructions=PROMPT, response_schema=harness_schema,
                         image_transform=lambda image: guide_image(image, seed, layout, size),
                         reasoning_effort='low', max_output_tokens=5500)
        save_json(output, dict(result, anchor_harness=layout, guide_sha256=guides_hash))

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(call, rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('p1', 'out', 'seeds', 'env-file'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    run(args.p1, args.out, args.seeds, args.env_file)
