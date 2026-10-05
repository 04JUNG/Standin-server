"""A reviewable removal ledger, with retained representatives and scene labels."""

import argparse
from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import urlencode

from ..storage import read_json


def render(result):
    def pose_card(key, pose_id, label):
        if not key:
            return '<p>품질 문제로 제외 · 대체 포즈를 지정하지 않았습니다.</p>'
        link = '/?' + urlencode({'pose': key})
        images = ''.join(
            f'<img loading="lazy" src="/api/poses/{escape(key)}/thumbnail?view={v}" alt="{label} {v}">'
            for v in ('front', 'three_quarter', 'side', 'back')
        )
        return f'<h3>{label} · <a href="{escape(link)}">{escape(pose_id)}</a></h3><div class="views">{images}</div>'

    rows = []
    for row in result['removals']:
        alias = row.get('scene_alias', {})
        scene = ' / '.join(str(alias[k]) for k in ('category', 'style') if alias.get(k))
        rows.append('<article>' + pose_card(row['key'], row['pose_id'], '제외')
                    + pose_card(row.get('keeper'), row.get('keeper_pose_id'), '유지')
                    + f'<p>{escape(row["reason"])}</p>'
                    + (f'<p class="muted">통합한 상황 이름: {escape(scene)}</p>' if scene else '')
                    + '</article>')
    counts = Counter(r['group'] for r in result['removals'])
    quality = sum(r['kind'] == 'quality' for r in result['removals'])
    coverage = result.get('coverage', {})
    comparison = (f'<p>고정 러프 비교: 유효 인물 {coverage["eligible"]}명 중 '
                  f'검색 거리 동일 {coverage["same"]}명, 증가 {coverage["worse"]}명. '
                  f'최대 증가 {coverage["max_increase"]:.6f}. 검색 거리는 시각적 성공률이 아닙니다.</p>') if coverage else ''
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>라이브러리 정리 결과</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f4f6f2;color:#243c34;margin:0}}
main{{max-width:1120px;margin:auto;padding:32px}}a{{color:#276952}}h1{{font-size:clamp(22px,3vw,30px);word-break:keep-all}}
article{{background:white;border:1px solid #dce3da;border-radius:14px;padding:20px;margin:24px 0}}
h3{{overflow-wrap:anywhere;font-size:16px}}.views{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
img{{width:100%;border-radius:8px}}.muted{{color:#627469}}.stats{{font-size:22px;font-weight:600}}
@media(max-width:650px){{main{{padding:16px}}.views{{grid-template-columns:repeat(2,1fr)}}}}</style>
<main><a href="/">← 포즈 라이브러리</a><h1>라이브러리 정리 결과</h1>
<p class="stats">검색 포즈 {result['before']['published']:,} → {result['after']['published']:,}</p>
<p>중복 통합 {len(result['removals']) - quality}개 · 품질 제외 {quality}개
 / 기존 {counts['existing']}개 · 신규 {counts['new']}개</p>
<p>4방향을 확인하고 제외했습니다. 원본 BVH·미리보기·이전 검수 기록은 보관하며,
라이브러리의 ‘제외한 포즈’에서 사유를 확인하거나 복구할 수 있습니다.</p>
<p class="muted">몸통·팔다리뿐 아니라 목, 손목, 발 방향과 손가락을 비교합니다.
손이 없는 예전 파일은 자동으로 같은 손 모양이라고 가정하지 않고 별도로 비교했습니다.</p>
{comparison}{''.join(rows)}</main></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(read_json(args.result)), encoding='utf-8')
