"""Read-only scenario index using the same revision-bound review decisions."""

from collections import defaultdict, Counter
from html import escape
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from .selection import decision


def scenario_router(catalog, store):
    router = APIRouter()

    @router.get("/scenarios", response_class=HTMLResponse)
    def scenarios():
        groups = defaultdict(list)
        reviews = store.all()
        for pose in catalog.all():
            if pose.metadata.get("scenario"):
                groups[pose.metadata["category"]].append(pose)
        labels = {
            "accepted": "채택",
            "rejected": "제외",
            "pending": "검수 중",
            "hold": "보류",
        }
        html = [
            '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>웹툰 상황별 포즈</title>',
            "<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:30px auto;padding:0 20px;color:#24382f}a{color:#25634e}nav{display:flex;flex-wrap:wrap;gap:15px}section{margin:35px 0}ul{padding:0;list-style:none;columns:2;column-gap:28px}li{break-inside:avoid;border-bottom:1px solid #ddd;padding:10px 0}small{color:#62736b;margin-left:12px}h1{line-height:1.3}@media(max-width:680px){ul{columns:1}}</style></head><body>",
            '<a href="/">← 전체 포즈 라이브러리</a><h1>웹툰 상황별 포즈</h1>',
            "<p>웹툰 상황별 포즈 · 단일 프레임 BVH · 양손 30관절 · 4방향 캐릭터 검수</p>",
            "<p>소품은 파지·배치를 확인하는 단순 도형이며 BVH에는 인물의 관절만 포함됩니다. 상호작용은 인물 한 명의 자세로, 상대의 위치는 장면에서 맞춰야 합니다.</p><nav>",
        ]
        for category, poses in groups.items():
            label = escape(poses[0].metadata["category_label"])
            html.append(f'<a href="#{category}">{label} {len(poses)}개</a>')
        html.append("</nav>")
        for category, poses in groups.items():
            label = escape(poses[0].metadata["category_label"])
            counts = Counter(decision(p, reviews)["status"] for p in poses)
            stats = " · ".join(f"{labels[k]} {v}" for k, v in counts.items())
            html.append(
                f'<section id="{category}"><h2>{label} · {len(poses)}개</h2><p>{stats} · <a href="/?category={category}">미리보기 모아 보기 ↗</a></p><ul>'
            )
            for i, pose in enumerate(poses, 1):
                title = escape(pose.metadata["style"])
                status = labels[decision(pose, reviews)["status"]]
                html.append(
                    f'<li><a href="/?category={category}&amp;pose={pose.key}">{i:02d}. {title}</a><small>{status}</small></li>'
                )
            html.append("</ul></section>")
        html.append("</body></html>")
        return HTMLResponse("\n".join(html), headers={"Cache-Control": "no-store"})

    return router
