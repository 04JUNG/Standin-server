"""Portable, in-memory rendering of the exact full-image RTM proposal."""

import io

from PIL import ImageDraw, ImageFont

from .schema import COCO17

EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overlay(image, prediction):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(11, round(width / 65)))
    except OSError:
        font = ImageFont.load_default()
    for person in prediction["people"]:
        points = [
            (
                None
                if p["state"] == "not_present"
                else (p["x"] * width / 1000, p["y"] * height / 1000)
            )
            for p in person["keypoints"]
        ]
        for a, b in EDGES:
            if points[a] is not None and points[b] is not None:
                draw.line(
                    (*points[a], *points[b]),
                    fill="#7851a9",
                    width=max(2, round(width / 350)),
                )
        for index, xy in enumerate(points):
            if xy is None:
                continue
            x, y = xy
            radius = max(3, round(width / 150))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill="#087f8c"
            )
            draw.text(
                (max(0, x - 8), max(0, y - 16)),
                f"P{person['person_index']}:{index}",
                font=font,
                fill="#17252f",
                stroke_width=1,
                stroke_fill="white",
            )
    return image


def context(prediction, width, height):
    return {
        "image_width": width,
        "image_height": height,
        "coordinates": "full-image 0..1000",
        "joint_order": COCO17,
        "overlay_labels": "P<RTM person index>:<COCO joint index>",
        "people": [
            {
                "person_index": p["person_index"],
                "bbox": p["bbox"],
                "keypoints": [
                    {
                        **{k: v for k, v in joint.items() if k != "score"},
                        "rtm_score": (
                            round(joint["score"], 4)
                            if joint.get("score") is not None
                            else None
                        ),
                    }
                    for joint in p["keypoints"]
                ],
            }
            for p in prediction["people"]
        ],
    }
