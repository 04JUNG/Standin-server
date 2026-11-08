"""Render paired rough, B0, saved candidate, and GT for the single-solve replay."""

import argparse
from html import escape
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.refine_v5.ab_common import load_json, save_json
from experiments.refine_v5.ab_evaluate import error_deg
from experiments.refine_v5.depth_ten_report import review_images
from experiments.refine_v5.learning_data import axes, digest


LABELS = {'B0': 'B0', 'single': 'Single solve', 'GT': 'Ground truth'}


def bundle(root, out):
    manifest = load_json(root / 'manifest.json')
    cases, native, labels, results = [], {}, [], []
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    (out / 'fbx').mkdir()
    with TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'native'
        stage.mkdir()
        for original in manifest['cases']:
            cid = original['case_id']
            folder = root / cid
            case = load_json(folder / 'input.json')
            evidence = load_json(folder / 'evidence.json')
            audit = load_json(folder / 'detector/audit.json')
            plan = load_json(folder / 'detector/plan.json')['result']
            gt = load_json(folder / 'gt.json')
            saved = folder / 'detector' / f'{cid}-part-goals-head-limbs.fbx'
            if digest(saved) != audit['output_sha256'] or digest(case['source']) != case['source_sha256']:
                raise ValueError('comparison artifact changed')
            if not audit['accepted'] and audit['output_sha256'] != case['source_sha256']:
                raise ValueError('fallback did not preserve B0 bytes')
            shutil.copyfile(saved, out / 'fbx' / f'{cid}.fbx')
            cases.append(dict(case, cameras={name: case['camera'] for name in ('ortho_fixed', 'single')}))
            native.update({(cid, 'B0'): case, (cid, 'single'): audit, (cid, 'GT'): gt})
            for method, path in [('B0', root / f'{cid}-B0.npz'),
                                 ('single', folder / 'detector/final.npz'), ('GT', folder / 'gt.npz')]:
                (stage / f'{cid}-{method}.npz').symlink_to(path.resolve())
            visible = np.flatnonzero(np.asarray(case['part_mask'][:4]))
            ground_truth = axes(gt['bones'])
            def score(bones):
                if not len(visible):
                    return None
                actual = axes(bones)
                return float(np.mean([error_deg(actual[i, 1], ground_truth[i, 1]) for i in visible]))
            before, after = score(case['bones']), score(audit['bones'])
            for method, value in [('B0', before), ('single', after)]:
                labels.append(dict(case_id=cid, method=method, status='reference', camera_status=None,
                                   chain_error_deg=value if value is not None else 0.))
            results.append(dict(case_id=cid, split=case['split'], representation=case['representation'],
                                visible_arm_bones=len(visible), before_deg=before, after_deg=after,
                                applied=audit['accepted'], edited_bones=list(plan['edits_deg']) if plan else [],
                                full_skin_certified=audit.get('full_skin_certified'), evidence=evidence,
                                native_reasons=audit.get('new_or_worsened_skin', []),
                                source_sha256=case['source_sha256'], output_sha256=audit['output_sha256'],
                                gt_sha256=gt['sha256'], fbx=f'fbx/{cid}.fbx'))
        review_images(stage.parent, out, cases, labels, native, labels=LABELS, short=LABELS)
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>v5 단일 보정 10컷</title>',
            '<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem}img{width:100%}section{border-top:1px solid #ddd}</style>',
            '<h1>v5 단일 보정 10컷</h1>',
            '<p>위는 전신 정면, 아래는 측면입니다. 각도는 관측 가능한 팔 뼈의 정답 방향 평균 오차입니다.</p>',
            '<p>시작점은 재검색 B0입니다. 팔은 관측이 충분한 쪽을 한 번 보정했습니다. 다리·머리와 표면 접촉은 검증된 별도 목표가 없었습니다.</p>',
            '<p><a href="overview.png">전체 이미지</a> · <a href="comparison.json">수치 JSON</a></p>']
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    grid = Image.new('RGB', (980, 500 * len(results)), 'white')
    for index, row in enumerate(results):
        cid = row['case_id']
        path = out / 'images' / f'{cid}.png'
        with Image.open(path) as image:
            cell = image.convert('RGB')
        draw = ImageDraw.Draw(cell)
        for col, key in [(1, 'before_deg'), (2, 'after_deg')]:
            x = col * 245
            draw.rectangle((x + 4, 24, x + 241, 41), fill='white')
            value = row[key]
            text = f'{value:.1f}°' if value is not None else 'Not scored'
            if col == 2:
                text += ' | Applied' if row['applied'] else ' | B0 retained'
            draw.text((x + 8, 25), text, fill='#333333', font=font)
        cell.save(path)
        grid.paste(cell, (0, 500 * index))
        reasons = ', '.join(f'{side} {details["status"]}' for side, details in row['evidence']['arms'].items())
        page.append(f'<section><h2>{cid} · {"보정 적용" if row["applied"] else "B0 유지"}</h2>'
                    f'<a href="images/{cid}.png"><img loading="lazy" src="images/{cid}.png" alt="{cid}"></a>'
                    f'<p>{escape(reasons)} · <a href="{row["fbx"]}">저장 FBX</a></p></section>')
    grid.save(out / 'overview.png')
    applied = [row for row in results if row['applied']]
    scored = [row for row in applied if row['before_deg'] is not None]
    summary = dict(cases=len(results), applied=len(applied), retained=len(results)-len(applied),
                   applied_mean_before_deg=float(np.mean([r['before_deg'] for r in scored])) if scored else None,
                   applied_mean_after_deg=float(np.mean([r['after_deg'] for r in scored])) if scored else None)
    save_json(out / 'comparison.json', dict(summary=summary, cases=results))
    (out / 'review.html').write_text('\n'.join(page), encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    bundle(**vars(parser.parse_args()))
