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


def draw_pose(joints, frame, size, highlight=(), title=""):
    sk = Skeleton(joints)
    pos, _, _ = fk_world(joints, frame)
    pos = pos - pos[sk.role["pelvis"]]
    img = Image.new("RGB", (size * len(VIEWS), size + 22), "white")
    d = ImageDraw.Draw(img)
    span = np.ptp(pos, axis=0).max() or 1.0
    scale = .8 * size / span
    font = _font(13)
    for v, (label, yaw) in enumerate(VIEWS):
        p2 = _project(pos, yaw)
        cx = v * size + size / 2 - (p2[:, 0].max() + p2[:, 0].min()) / 2 * scale
        cy = 22 + size / 2 + (p2[:, 1].max() + p2[:, 1].min()) / 2 * scale
        xy = lambda i: (cx + p2[i, 0] * scale, cy - p2[i, 1] * scale)
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
        d.text((v * size + 6, 3), label, fill=(90, 90, 90), font=font)
    if title:
        d.text((size * len(VIEWS) - 6, 3), title, fill=(0, 0, 0), font=font, anchor="ra")
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


def comparison(joints, before, after, findings, after_findings, header, size=220):
    sk = Skeleton(joints)
    top = draw_pose(joints, before, size, highlight_joints(sk, findings), "원본")
    rows = [top]
    if after is not None:
        rows.append(draw_pose(joints, after, size, highlight_joints(sk, after_findings), "자동 수정안"))
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
