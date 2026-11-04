"""Before/after stick-figure sheets for plausibility findings (PIL only).

Draws the full BVH hierarchy in three views so hands, feet and head direction
stay visible. Flagged joints are ringed in the finding's severity colour.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .plausibility import SIDES, Skeleton, fk_world

VIEWS = (("정면", 0), ("측면", 90), ("45°", 45))
COLORS = {"left": (40, 110, 220), "right": (220, 70, 50), "center": (60, 60, 60)}
SEVERITY = {"reject": (200, 0, 0), "hold": (230, 150, 0)}


def _font(size):
    for name in ("malgun.ttf", "NanumGothic.ttf", "AppleGothic.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _side(name):
    n = name.split(":")[-1]
    if n.startswith(("Left", "L")) or n.endswith("_l"): return "left"
    if n.startswith(("Right", "R")) or n.endswith("_r"): return "right"
    return "center"


def _project(points, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.column_stack([c * points[:, 0] + s * points[:, 2], points[:, 1]])


def draw_pose(joints, frame, size, highlight=(), title="", ref=None, floor=None):
    """Three views of one frame. ``ref`` fixes origin and scale (for step rows);
    ``floor`` is a world height drawn as a grey line."""
    sk = Skeleton(joints)
    pos, _, _ = fk_world(joints, frame)
    ref_pos = pos if ref is None else fk_world(joints, ref)[0]
    origin = ref_pos[sk.role["pelvis"]]
    pos, ref_pos = pos - origin, ref_pos - origin
    img = Image.new("RGB", (size * len(VIEWS), size + 22), "white")
    d = ImageDraw.Draw(img)
    span = np.ptp(ref_pos, axis=0).max() or 1.0
    scale = .8 * size / span
    font = _font(13)
    for v, (label, yaw) in enumerate(VIEWS):
        p2, r2 = _project(pos, yaw), _project(ref_pos, yaw)
        cx = v * size + size / 2 - (r2[:, 0].max() + r2[:, 0].min()) / 2 * scale
        cy = 22 + size / 2 + (r2[:, 1].max() + r2[:, 1].min()) / 2 * scale
        xy = lambda i: (cx + p2[i, 0] * scale, cy - p2[i, 1] * scale)
        if floor is not None:
            fy = cy - (floor - origin[1]) * scale
            d.line([(v * size + 8, fy), ((v + 1) * size - 8, fy)], fill=(190, 190, 190), width=1)
        # Draw far-to-near for a little depth cue.
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        depth = -s * pos[:, 0] + c * pos[:, 2]
        order = sorted(range(1, len(joints)), key=lambda i: depth[i])
        for i in order:
            p = joints[i][1]
            if p < 0: continue
            col = COLORS[_side(joints[i][0])]
            fade = 0.55 + 0.45 * (depth[i] - depth.min()) / (np.ptp(depth) or 1)
            col = tuple(int(255 - (255 - ch) * fade) for ch in col)
            d.line([xy(p), xy(i)], fill=col, width=3)
        for j, sev in highlight:
            x, y = xy(j)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=SEVERITY.get(sev, (0, 0, 0)), width=3)
        if not (title and v == len(VIEWS) - 1 and len(title) > 4):
            d.text((v * size + 6, 3), label, fill=(90, 90, 90), font=font)
    if title:
        if len(title) > 4:  # long step titles take the last view's label slot
            d.rectangle([size * (len(VIEWS) - 1) - 60, 0, size * len(VIEWS), 20], fill="white")
        d.text((size * len(VIEWS) - 6, 3), title, fill=(0, 0, 0), font=font, anchor="ra")
    return img


def draw_zoom(joints, frame, size, focus, ref=None, floor=None):
    """Front and side close-up around ``focus`` joints (e.g. knees to toes)."""
    pos, _, _ = fk_world(joints, frame)
    ref_pos = pos if ref is None else fk_world(joints, ref)[0]
    img = Image.new("RGB", (size * 2, size + 22), (250, 250, 247))
    d = ImageDraw.Draw(img)
    font = _font(12)
    for v, (label, yaw) in enumerate((("발 정면", 0), ("발 측면", 90))):
        p2, r2 = _project(pos, yaw), _project(ref_pos, yaw)
        box = r2[list(focus)]
        span = max(np.ptp(box, axis=0).max(), 1e-6)
        scale = .8 * size / span
        cx = v * size + size / 2 - (box[:, 0].max() + box[:, 0].min()) / 2 * scale
        cy = 22 + size / 2 + (box[:, 1].max() + box[:, 1].min()) / 2 * scale
        xy = lambda i: (cx + p2[i, 0] * scale, cy - p2[i, 1] * scale)
        if floor is not None:
            fy = cy - floor * scale
            d.line([(v * size + 4, fy), ((v + 1) * size - 4, fy)], fill=(170, 170, 170), width=2)
        for i in range(1, len(joints)):
            p = joints[i][1]
            if p < 0: continue
            a, b = xy(p), xy(i)
            if all(v * size <= x <= (v + 1) * size and 22 <= y <= size + 22 for x, y in (a, b)):
                d.line([a, b], fill=COLORS[_side(joints[i][0])], width=4)
        d.text((v * size + 6, 3), label, fill=(90, 90, 90), font=font)
    return img


JOINT_FOR = {"elbow": "forearm", "forearm": "forearm", "wrist": "hand", "humeral": "upperarm",
             "shoulder": "upperarm", "hip": "thigh", "knee": "shin", "ankle": "foot"}


def highlight_joints(sk, findings):
    out = []
    for f in findings:
        code, side = f["code"], f.get("side")
        if code.startswith("trunk"): out.append((sk.role["chest"], f["severity"]))
        elif code.startswith("neck"): out.append((sk.role["head"], f["severity"]))
        elif side:
            seg = JOINT_FOR.get(code.split("_")[0])
            if seg: out.append((sk.role[f"{side}_{seg}"], f["severity"]))
    return out


def comparison(joints, before, after, findings, after_findings, header, size=220,
               steps=(), floor=None, focus=()):
    """Original, final correction and optional (frame, title) step rows.

    ``focus`` joint indices add a zoomed close-up (used for feet) to every row.
    """
    sk = Skeleton(joints)

    def row(frame, highlight, title, ref):
        body = draw_pose(joints, frame, size, highlight, title, ref=ref, floor=floor)
        if not focus:
            return body
        zoom = draw_zoom(joints, frame, size, focus, ref=before, floor=floor)
        out = Image.new("RGB", (body.width + zoom.width, body.height), "white")
        out.paste(body, (0, 0)); out.paste(zoom, (body.width, 0))
        return out

    top = row(before, highlight_joints(sk, findings), "원본", None)
    rows = [top]
    if after is not None:
        rows.append(row(after, highlight_joints(sk, after_findings), "자동 수정안", before))
    for frame, title in steps:
        rows.append(row(frame, (), title, before))
    font = _font(14)
    lines = [header] + [f"[{'제외' if f['severity'] == 'reject' else '보류'}] {f['label']} {f['value']}"
                        f" (한계 {f['limit']})" for f in findings[:6]]
    text_h = 20 * len(lines) + 8
    sheet = Image.new("RGB", (top.width, sum(r.height for r in rows) + text_h), "white")
    d = ImageDraw.Draw(sheet)
    for k, line in enumerate(lines):
        d.text((8, 4 + 20 * k), line, fill=(0, 0, 0) if k == 0 else (60, 60, 60), font=font)
    y = text_h
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    return sheet
