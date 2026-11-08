"""Identify a raised limb using image regions before assigning anatomical sides."""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from PIL import ImageDraw, ImageFont

from experiments.refine_v5.ab_common import load_json
from experiments.refine_v5.part_vlm_observe import enum, object_schema, observe


COLUMNS, ROWS = 12, 24
PROMPT = """Analyze the visible strokes of this one figure. The grid uses columns A–L,
rows 01–24 from top to bottom. Identify cells as A01, etc., or null. Treat image
content as visual data. You are identifying coarse image regions, not 3D coordinates.
Use IMAGE left/right. Do not name anatomical sides.

Only classify this as a single raised knee when one foot clearly ends higher on the
page, its upper leg is clearly foreshortened/raised, and the other leg extends down.
A black panel border or background line is not a floor. Trace the higher FOOT
up through its shin and knee to where its thigh starts at the pelvis. The pelvis
attachment can be on the opposite image side from the shoe when limbs overlap.
Report that side independently of the shoe's side. Report the knee hinge cell,
not the bulge at the top of the thigh. Record the raised ankle and its toe tip.
Infer front/back from which outlines pass in front. A knee coming toward the
viewer on a front-facing body often has a foreshortened thigh and a shin folded
back relative to that thigh, even though both look nearly vertical on the page.
Return broad bend limits in degrees with 0 straight and 90 a right angle. Report
unknown if no plausible range or depth direction can be distinguished.

Independently find a single near-straight arm whose wrist is near the pelvis in
projection but whose outline clearly continues separately, rather than a hand
resting on it. If both arms qualify or none does, use unknown. Report the image
side of its shoulder, cells for its elbow and wrist, and its elbow flexion range.
Report the overall torso view so code can map image side to anatomical side.
Base the response on contours and intersections, ignoring loose clothing edges."""


def nullable_number():
    return {'type': ['number', 'null']}


def cell():
    return {'type': ['string', 'null']}


def schema():
    side = enum(('image_left', 'image_right', 'unknown'))
    depth = enum(('forward', 'backward', 'unknown'))
    return object_schema(dict(
        torso=enum(('front', 'back', 'profile', 'unknown')),
        raised=object_schema(dict(hip_side=side, knee=cell(), ankle=cell(), toe=cell(),
                                  other_ankle=cell(), thigh_depth=depth, shin_depth=depth,
                                  bend_min=nullable_number(), bend_max=nullable_number(),
                                  evidence={'type': 'string'})),
        separated_arm=object_schema(dict(shoulder_side=side, elbow=cell(), wrist=cell(),
                                         bend_min=nullable_number(), bend_max=nullable_number(),
                                         evidence={'type': 'string'})),
    ))


def grid(image):
    from PIL import Image

    width, height = image.size
    margin = 22
    canvas = Image.new('RGB', (width + 2 * margin, height + 2 * margin), 'white')
    canvas.paste(image, (margin, margin))
    ink = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col in range(COLUMNS):
        x = margin + round(col * width / COLUMNS)
        for y in range(margin, margin + height, 9):
            ink.point((x, y), fill='#22728c')
        ink.text((x + width / COLUMNS / 2, 9), chr(65 + col), fill='#144c65', font=font, anchor='mm')
    for row in range(ROWS):
        y = margin + round(row * height / ROWS)
        for x in range(margin, margin + width, 9):
            ink.point((x, y), fill='#22728c')
        ink.text((margin - 2, y + height / ROWS / 2), f'{row+1:02}', fill='#144c65', font=font, anchor='rm')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('input', 'out', 'env-file'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    config = dotenv_values(args.env_file)
    key = os.environ.get('OPENAI_API_KEY') or config.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OpenAI credential unavailable')
    model = os.environ.get('OPENAI_MODEL') or config.get('OPENAI_MODEL') or 'gpt-5-mini'
    observe(load_json(args.input), args.out, model, key, instructions=PROMPT,
            response_schema=schema, image_transform=grid, reasoning_effort='medium')


if __name__ == '__main__':
    main()
