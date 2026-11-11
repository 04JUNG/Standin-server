"""Read-only rough preparation. Inventory is not person GT or consent evidence."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS
from .util import hash_jsonl, image_dimensions, sha256_file, write_json, write_jsonl

INTAKE_FIELDS = ['컷ID', '파일명', '작가그룹ID', '작품ID', '원본페이지장면그룹ID', '평가세트',
                 '대표셋_어려운셋', '인물수', '태그', '평가사용허용', '팀내공유허용', '외부VLM전송허용', '비고',
                 'expected_route']
PERSON_FIELDS = ['컷ID', '인물별칭', '위치설명', '3D보조대상', '핵심동작', '보존할부위',
                 '인물기준좌우', '보이는방향', '가린부위', '접촉_지지의도', '작가가아는숨은자세',
                 '희망체형', '왼손', '오른손', '출력범위', '판단불가사항', 'person_id',
                 'bbox_xyxy', 'bbox_source', 'bbox_confirmed', 'out_of_scope', 'scale_class']


def write_csv(path: Path, fields: list[str], rows) -> None:
    # Excel/Sheets must never execute a source filename as a formula.
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')) else value
                             for key, value in row.items()})


def prepare_inventory(source: str | Path, output: str | Path) -> dict:
    root, destination = Path(source).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError('source must be an existing directory')
    if destination == root or root in destination.parents:
        raise ValueError('output must be outside the read-only source tree')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('inventory output already exists; use a new version')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    rows, by_hash = [], defaultdict(list)
    for path in paths:
        # Symlinks outside the supplied source are not authorized inputs.
        if root not in path.resolve().parents:
            raise ValueError('image symlink escapes supplied source')
        digest = sha256_file(path)
        width, height = image_dimensions(path)
        row = dict(cut_id='rough-' + digest, relative_path=path.relative_to(root).as_posix(),
                   image_sha256=digest, bytes=path.stat().st_size, width=width, height=height,
                   artist_id=None, project_id=None, scene_group_id=None, split=None,
                   person_count=None, evaluation_permission='unknown', sharing_permission='unknown',
                   external_vlm_permission='unknown')
        rows.append(row)
        by_hash[digest].append(row)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / 'inventory.jsonl', rows)
    write_jsonl(destination / 'unresolved_sources.jsonl', rows)
    write_jsonl(destination / 'duplicate_groups.jsonl', [
        {'image_sha256': digest, 'cut_id': group[0]['cut_id'],
         'relative_paths': [r['relative_path'] for r in group]}
        for digest, group in sorted(by_hash.items()) if len(group) > 1])
    intake = [{'컷ID': group[0]['cut_id'], '파일명': group[0]['relative_path'],
               '평가사용허용': '확인 필요', '팀내공유허용': '확인 필요', '외부VLM전송허용': '확인 필요'}
              for group in by_hash.values()]
    write_csv(destination / 'rough-intake.csv', INTAKE_FIELDS, intake)
    write_csv(destination / 'person-intent.csv', PERSON_FIELDS, [])
    write_csv(destination / 'profile-matrix.csv', ['character_id', 'asset_sha256', 'rig_sha256',
              'controller_revision', 'shape_revision', 'shape_parameters', 'left_hand_id',
              'right_hand_id', 'hand_library_sha256', 'output_scope', 'review_status'], [])
    write_csv(destination / 'review-sessions.csv', ['session_id', 'labeler_id', 'tool', 'tool_version',
              'OS', 'assignment_id', 'task_id', 'started_at', 'finished_at', 'completed',
              'timeout', 'usability', 'defect_id', 'notes'], [])
    sizes = Counter(f"{r['width']}x{r['height']}" for r in rows)
    summary = dict(schema_version='rough-inventory-v1', files=len(rows),
                   unique_image_contents=len(by_hash), duplicate_files=len(rows)-len(by_hash),
                   duplicate_groups=sum(len(g)>1 for g in by_hash.values()),
                   total_bytes=sum(r['bytes'] for r in rows), dimensions=dict(sorted(sizes.items())),
                   unknown_dimensions=sum(r['width'] is None or r['height'] is None for r in rows),
                   unresolved_source_files=len(rows), human_person_rows=0,
                   independent_groups=None, external_transmission_authorized_files=0,
                   inventory_sha256=hash_jsonl(rows), status='INCONCLUSIVE')
    write_json(destination / 'summary.json', summary)
    # Local provenance only; never include it in public reports.
    write_json(destination / 'local-source.json', {'source_root': str(root), 'read_only': True})
    return summary
