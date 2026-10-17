"""Shared live/eval inputs: preserve head context and identify the intended person.

Boxes come from existing descriptors, never from a new VLM coordinate prediction.
The unmarked detail crop and a marked crop locator serve different purposes.
"""

from dataclasses import dataclass
import math

INPUT_VERSION = "body-coarse-input.v1.head-context"


@dataclass
class CoarseInput:
    crop: object
    context: object
    target_box: tuple
    crop_box: tuple


def prepare_input(image, box):
    from PIL import ImageDraw

    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("invalid_coarse_target_box")
    width, height = image.size
    x1, y1, x2, y2 = box
    target = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )
    x1, y1, x2, y2 = target
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        raise ValueError("invalid_coarse_target_box")
    # Pose-derived boxes often start at the eyes or shoulders, cutting off the head.
    # Keep the detector box as the identity anchor; never recenter on a neighbor.
    side = math.ceil(w * 0.2)
    top = math.ceil(max(h * 0.3, w * 0.35))
    bottom = math.ceil(h * 0.05)
    crop_box = (
        max(0, x1 - side),
        max(0, y1 - top),
        min(width, x2 + side),
        min(height, y2 + bottom),
    )
    clean = image.convert("RGB")
    crop = clean.crop(crop_box)
    crop.thumbnail((768, 768))
    # The remote locator uses exactly the same pixels as the authorized person
    # crop. Never send the full source cut as additional remote context.
    context = clean.crop(crop_box)
    draw = ImageDraw.Draw(context)
    ox, oy = crop_box[:2]
    stroke = max(2, math.ceil(max(context.size) / 350))
    draw.rectangle(
        (x1 - ox, y1 - oy, x2 - ox - 1, y2 - oy - 1),
        outline=(255, 0, 180),
        width=stroke,
    )
    context.thumbnail((768, 768))
    return CoarseInput(crop, context, target, crop_box)
