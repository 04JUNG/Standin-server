"""Validate completed human worksheets and export the existing dataset schema.

Blank/unknown human decisions block export, never become inferred zero people,
consent, labels, scene groups or routing. The original inventory remains sealed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .dataset import EvalDataset, seal_dataset, validate_dataset
from .schemas import SCHEMA_VERSION
from .util import hash_jsonl, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def _csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('missing or duplicate CSV header')
        rows = list(reader)
        if any(None in r or None in r.values() for r in rows):
            raise ValueError('malformed CSV row')
        return rows


def import_intake(inventory: str | Path, intake: str | Path, persons: str | Path,
                  output: str | Path, *, dataset_id: str) -> dict:
    root, destination = Path(inventory).resolve(), Path(output).resolve()
    if destination == root or root in destination.parents:
        raise ValueError('dataset output must preserve the original inventory')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('dataset output must be new')
    original = read_jsonl(root / 'inventory.jsonl')
    if hash_jsonl(original) != read_json(root / 'summary.json')['inventory_sha256']:
        raise ValueError('inventory changed after preparation')
    source = Path(read_json(root / 'local-source.json')['source_root'])
    by_cut = {r['cut_id']: r for r in original}
    cuts, people, errors, seen = [], [], [], set()
    intake_rows, person_rows = _csv(intake), _csv(persons)
    split_map = {'D0': 'engineering', 'D1': 'calibration', 'D2': 'holdout', 'D3': 'pilot'}
    for row in intake_rows:
        identity = row.get('컷ID', '')
        if identity not in by_cut or identity in seen:
            errors.append(f'{identity}: unknown or duplicate cut ID')
            continue
        seen.add(identity)
        unresolved = [f for f in ('작가그룹ID', '작품ID', '원본페이지장면그룹ID')
                      if not row.get(f, '').strip() or row[f].lower().startswith(('unknown', 'unresolved'))]
        if unresolved:
            errors.append(f'{identity}: unresolved source groups: {unresolved}')
        if row.get('평가사용허용') != '허용':
            errors.append(f'{identity}: explicit evaluation permission required')
        for field in ('팀내공유허용', '외부VLM전송허용'):
            if row.get(field) not in {'허용', '불허', '확인 필요'}:
                errors.append(f'{identity}: invalid permission {field}')
        split = split_map.get(row.get('평가세트'))
        if split is None or row.get('expected_route') not in {'core', 'bust', 'skip'}:
            errors.append(f'{identity}: explicit split and expected_route required')
        try:
            count = int(row['인물수'])
            if count < 0 or str(count) != row['인물수'].strip():
                raise ValueError()
        except (ValueError, KeyError):
            errors.append(f'{identity}: person count requires human confirmation')
            count = None
        originals = [r for r in original if r['cut_id'] == identity]
        canonical = originals[0]
        image_path = source / canonical['relative_path']
        if source not in image_path.resolve().parents or sha256_file(image_path) != canonical['image_sha256']:
            errors.append(f'{identity}: source image changed or escaped source root')
        cuts.append(dict(schema_version=SCHEMA_VERSION, cut_id=identity, image_path=str(image_path),
                         image_sha256=canonical['image_sha256'], image_width=canonical['width'],
                         image_height=canonical['height'], aliases=[str(source / r['relative_path']) for r in originals],
                         artist_id=row.get('작가그룹ID'), project_id=row.get('작품ID'),
                         scene_group_id=row.get('원본페이지장면그룹ID'), split=split,
                         expected_route=row.get('expected_route'), num_people_gt=count,
                         license_scope='human-confirmed-evaluation-only',
                         external_vlm_permission=row.get('외부VLM전송허용'), sharing_permission=row.get('팀내공유허용'),
                         sampling_cohort=row.get('대표셋_어려운셋'), tags=row.get('태그')))
    if set(by_cut) != seen:
        errors.append('all inventory cuts must have intake rows; silent exclusions forbidden')
    person_ids = set()
    for row in person_rows:
        identity, cut_id = row.get('person_id', ''), row.get('컷ID', '')
        if not identity or identity in person_ids or cut_id not in seen:
            errors.append(f'{identity}: explicit permanent unique person ID and known cut required')
        person_ids.add(identity)
        try:
            box = json.loads(row.get('bbox_xyxy', ''))
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in box):
                raise ValueError('nonfinite bbox')
        except (ValueError, TypeError):
            box = None
        if row.get('bbox_confirmed') != '예' or not row.get('bbox_source', '').strip():
            errors.append(f'{identity}: human-confirmed bbox required')
        if row.get('3D보조대상') not in {'예', '아니오'} or row.get('out_of_scope') not in {'예', '아니오'}:
            errors.append(f'{identity}: explicit target and out_of_scope decisions required')
        people.append(dict(schema_version=SCHEMA_VERSION, person_id=identity, cut_id=cut_id,
                           bbox_xyxy=box, bbox_source=row.get('bbox_source'),
                           eligible={'예': True, '아니오': False}.get(row.get('3D보조대상')),
                           out_of_scope={'예': True, '아니오': False}.get(row.get('out_of_scope')),
                           scale_class=row.get('scale_class'), human_intent=dict(row)))
    # No writer/project/scene may straddle calibration and the sealed holdout.
    for field in ('artist_id', 'project_id', 'scene_group_id'):
        tune = {r[field] for r in cuts if r['split'] == 'calibration'}
        holdout = {r[field] for r in cuts if r['split'] == 'holdout'}
        if tune & holdout:
            errors.append(f'D1/D2 leakage in {field}')
    candidate = EvalDataset(destination, dict(schema_version=SCHEMA_VERSION, dataset_id=dataset_id), cuts, people)
    errors.extend(i.message for i in validate_dataset(candidate, check_seal=False) if i.level == 'error')
    if errors:
        return {'status': 'INCONCLUSIVE', 'exported': False, 'errors': errors,
                'expected_cuts': len(by_cut), 'person_rows': len(people)}
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / 'cuts.jsonl', cuts)
    write_jsonl(destination / 'persons.jsonl', people)
    candidate.manifest['purpose'] = 'engineering'
    seal_dataset(candidate)
    write_json(destination / 'intake-provenance.json', dict(
        inventory_sha256=hash_jsonl(original), intake_sha256=sha256_file(intake), persons_sha256=sha256_file(persons)))
    return {'status': 'ready', 'exported': True, 'dataset_id': dataset_id,
            'cuts': len(cuts), 'persons': len(people), 'target_persons': len(candidate.target_persons)}
