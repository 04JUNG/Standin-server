"""Create an isolated local review gallery with matching FBX/GLB downloads."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil

WEAPONS = set("sword greatsword dagger spear shield pistol rifle crossbow bow axe staff wand".split())


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise ValueError("gallery destination already contains a different asset")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--thumbnails", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.catalog.read_text("utf-8"))["poses"]
    a.output.mkdir(parents=True, exist_ok=True)
    cards = []
    for r in rows:
        source = a.base / "scene-recovery-r7/poses" / r["pose_id"]
        digest = r["source_bvh_sha256"]
        destination = a.output / "assets" / digest
        images = []
        for angle, label in ((0, "정면"), (45, "45°"), (90, "측면"), (180, "후면")):
            name = f"full__yaw_{angle:03d}.jpg"
            link(a.thumbnails / digest / name, destination / name)
            images.append(f'<figure><img loading="lazy" src="assets/{digest}/{name}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        downloads = []
        for scope, label in (("full", "전신"), ("half", "반신"), ("bust", "흉상"), ("shoulder_head", "두상")):
            for extension in ("fbx", "glb"):
                link(source / f"{scope}.{extension}", destination / f"{scope}.{extension}")
            downloads.append(f'<a download="{html.escape(r["pose_id"])}-{scope}.fbx" href="assets/{digest}/{scope}.fbx">{label} FBX</a>')
        category = "weapon" if set(r["geometry_kinds"]) & WEAPONS else "daily"
        searchable = html.escape(" ".join([r["pose_id"], *r["objects"]]), quote=True)
        cards.append(f'<article data-category="{category}" data-search="{searchable}"><h2>{html.escape(r["pose_id"])}</h2><p>{html.escape(" · ".join(r["objects"]))}</p><div class="views">{"".join(images)}</div><nav>{"".join(downloads)}</nav><a href="assets/{digest}/full.glb" download>전신 GLB</a></article>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>소품 포함 포즈</title>
<style>body{font:15px system-ui;margin:0;background:#f5f6f7;color:#17252b}header{position:sticky;top:0;background:white;padding:20px 4%;border-bottom:1px solid #dfe5e4;z-index:1}h1{margin:0 0 8px;font-size:24px}p{color:#53656b}button,input{font:inherit;padding:10px;border:1px solid #c7d1ce;border-radius:7px;background:white;margin-right:6px}button.active{background:#286453;color:white}main{padding:24px 4%;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{padding:18px;background:white;border:1px solid #dfe5e4;border-radius:12px}h2{font-size:15px;overflow-wrap:anywhere}.views{display:grid;grid-template-columns:repeat(2,1fr)}figure{margin:0;text-align:center}img{width:100%;aspect-ratio:1}figcaption{font-size:12px;color:#687777}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 10px}a{color:#286453}article[hidden]{display:none}input{min-width:260px}</style>
<header><h1>소품 포함 포즈</h1><p>무기와 일상 소품의 형태·배치를 확인하고, 소품이 함께 들어 있는 FBX를 내려받을 수 있습니다. 소품은 기본 형태 모델입니다.</p><button data-filter="all" class="active">전체</button><button data-filter="weapon">무기</button><button data-filter="daily">일상·기타</button><input placeholder="포즈 이름 또는 소품 검색 (sword, cup, book 등)"><span id="count"></span></header><main>''' + "".join(cards) + '''</main><script>
let category='all';const cards=[...document.querySelectorAll('article')],input=document.querySelector('input');function filter(){let n=0;for(const c of cards){c.hidden=!((category==='all'||c.dataset.category===category)&&c.dataset.search.toLowerCase().includes(input.value.toLowerCase()));if(!c.hidden)n++;}document.querySelector('#count').textContent=n+'개';}for(const b of document.querySelectorAll('button'))b.onclick=()=>{category=b.dataset.filter;for(const x of document.querySelectorAll('button'))x.classList.toggle('active',x===b);filter();};input.oninput=filter;filter();</script></html>'''
    (a.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"poses": len(rows), "gallery": str(a.output / "index.html")}))


if __name__ == "__main__":
    main()
