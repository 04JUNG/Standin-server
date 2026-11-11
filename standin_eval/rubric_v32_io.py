"""Version 3.2 native package adapter and portable scope-aware reports."""
from pathlib import Path
import html
import shutil
import tempfile

from .rubric_v32 import RUBRIC, REVIEW_SCHEMA, CONTRACT, summarize
from .rubric_score import _schema, _verified_file
from .util import read_json, read_jsonl, write_json, write_jsonl, sha256_file, hash_json, atomic_write_text


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asset(root, value):
    _verified_file(root, value['path'], value['sha256'])
    require((root / value['path']).stat().st_size == value['bytes'], 'asset byte size mismatch')


def normalize_manifest(manifest, root):
    """Retain native metadata while adding the existing harness's local binding."""
    if not manifest['cases'] or 'input' not in manifest['cases'][0]:
        return manifest
    cases = []
    for c in manifest['cases']:
        refs = c['expected_person_ids']
        require(len(set(refs)) == len(refs) == c['expected_people'], 'native expected people mismatch')
        selections = c['final_selections']
        require(len(selections) == c['returned_people'], 'native returned people mismatch')
        ids = [s['reference_person_id'] for s in selections]
        require(len(ids) == len(set(ids)) and set(ids) <= set(refs), 'invalid native person association')
        require(set(c['missing_expected_person_ids']) == set(refs) - set(ids), 'missing people mismatch')
        asset(root, c['input'])
        ev_path = root / c['source_execution_evidence']
        require(not Path(c['source_execution_evidence']).is_absolute() and root.resolve() in ev_path.resolve().parents,
                'evidence path escapes root')
        ev = read_json(ev_path)
        require(ev['case_id'] == c['case_id'] and ev['job_id'] == c['job_id'], 'execution identity mismatch')
        completed = ev['job_status']['status'] == 'completed' and ev['job_status']['error'] is None
        if c['result_status'] in ('returned_all', 'returned_partial', 'returned_none'):
            require(completed, 'successful result lacks completed execution evidence')
        require(ev['candidates'] == c['returned_candidates'], 'candidate execution evidence mismatch')
        pool = c['returned_candidates']
        keys = [(p['reference_person_id'], p['original_rank']) for p in pool]
        identity = [(p['reference_person_id'], p['candidate_id']) for p in pool]
        require(len(keys) == len(set(keys)) and len(identity) == len(set(identity)), 'duplicate native candidate')
        for p in pool:
            require(type(p['original_rank']) is int and 1 <= p['original_rank'] <= 5, 'invalid native candidate rank')
            require(p['reference_person_id'] in refs, 'unknown candidate person')
        candidates, chosen = [], []
        boxes = {b['reference_person_id']: b['box'] for b in c.get('source_person_boxes', [])}
        for s in selections:
            pid = s['reference_person_id']
            require(type(s['service_person_index']) is int and s['service_person_index'] >= 0,
                    'invalid service person index')
            require(any(p['reference_person_id'] == pid and p['original_rank'] == s['original_rank']
                        and p['candidate_id'] == s['candidate_id'] for p in pool), 'selected candidate not returned')
            for key in ('final_image', 'fbx', 'final_bvh'):
                if s.get(key) is not None:
                    asset(root, s[key])
            prefix = root / 'evidence' / c['case_id'] / f"P{s['service_person_index'] + 1}"
            require(root.resolve() in prefix.resolve().parents, 'receipt path escapes root')
            choice = read_json(str(prefix) + '-choice.json')
            require((choice['candidate_id'], choice['rank']) == (s['candidate_id'], s['original_rank']), 'choice receipt mismatch')
            refine = read_json(str(prefix) + '-refine-receipt.json')
            require(refine.get('person_index') == s['service_person_index'], 'refine person mismatch')
            require((refine['candidate_id'], refine['rank'], refine['job_id']) ==
                    (s['candidate_id'], s['original_rank'], c['job_id']), 'refine receipt mismatch')
            if s.get('fbx'):
                fbx = read_json(str(prefix) + '-fbx-receipt.json')
                require((fbx['candidate_id'], fbx['chosen_rank'], fbx['job_id'], fbx['fbx_sha256']) ==
                        (s['candidate_id'], s['original_rank'], c['job_id'], s['fbx']['sha256']), 'FBX receipt mismatch')
            candidates.append({'reference_person_id': pid, 'candidate_id': s['candidate_id'], 'rank': s['original_rank'],
                               'final_image': s['final_image']['path'], 'final_image_sha256': s['final_image']['sha256'],
                               'refine_status': s['refine_status']})
            chosen.append({'reference_person_id': pid, 'candidate_id': s['candidate_id'], 'rank': s['original_rank'],
                           **({'reference_bbox_xyxy': boxes[pid]} if pid in boxes else {})})
        cases.append({'case_id': c['case_id'], 'input_image': c['input']['path'], 'input_sha256': c['input']['sha256'],
                      'reference_person_ids': refs, 'framing': c['framing'], 'evaluation_scope': c['evaluation_scope'],
                      'result_status': c['result_status'], 'response_completed': completed, 'candidates': candidates,
                      'selections': chosen, 'execution_error': c.get('execution_error') or ev['job_status']['error'],
                      'native_evidence': c})
    return {**manifest, 'cases': cases, 'native_manifest_sha256': hash_json(manifest)}


def score_bundle_v32(manifest_path, reviews_path, config_path, output, artifacts_root, *, validate_only=False):
    root, output = Path(artifacts_root).resolve(), Path(output)
    require(not output.exists(), 'use a new output directory')
    original_manifest, config = read_json(manifest_path), read_json(config_path)
    manifest = normalize_manifest(original_manifest, root)
    rows = read_jsonl(reviews_path)
    cases = manifest['cases']; ids = [c['case_id'] for c in cases]
    require(len(ids) == len(set(ids)) and set(ids) == {r['case_id'] for r in rows}, 'fixed case set mismatch')
    require(config.get('mode') in ('replay_review', 'live_e2e'), 'source run mode required')
    require(config.get('rubric_id', RUBRIC['rubric_id']) == RUBRIC['rubric_id'], 'rubric id mismatch')
    by_id = {c['case_id']: c for c in cases}
    for r in rows:
        c = by_id[r['case_id']]
        require(r['run_id'] == config['run_id'], 'run_id mismatch')
        scope = 'upper_body' if c['framing'] == '01_상반신' else 'full_body'
        require(scope == c['evaluation_scope'] == r['evaluation_scope'], 'scope/framing mismatch')
        require(c['result_status'] == r['result_status'], 'product status changed')
        if 'selections' in c:
            selected_keys = [(p['reference_person_id'], p['candidate_id'], p['rank']) for p in c['selections']]
            reviewed_keys = [(p['reference_person_id'], p['candidate_id'], p['chosen_rank']) for p in r['people']]
            require(sorted(selected_keys) == sorted(reviewed_keys), 'review differs from frozen selections')
        refs = c['reference_person_ids']
        require(len(refs) == len(set(refs)) == r['expected_people'], 'expected identity count mismatch')
        require({p['reference_person_id'] for p in r['people']} <= set(refs), 'unknown reference person')
        _verified_file(root, c['input_image'], c['input_sha256'])
        pool = c['candidates']
        keys = [(p['reference_person_id'], p['rank']) for p in pool]
        require(len(keys) == len(set(keys)), 'duplicate candidate rank')
        identities = [(p['reference_person_id'], p['candidate_id']) for p in pool]
        require(len(identities) == len(set(identities)), 'duplicate candidate identity')
        require(all(type(rank) is int and 1 <= rank <= 5 and pid in refs for pid, rank in keys), 'bad rank/person')
        if r['result_status'] == 'returned_none':
            require(c.get('response_completed') is True and not pool, 'normal no-result lacks evidence')
        for p in r['people']:
            matches = [x for x in pool if (x['reference_person_id'], x['candidate_id'], x['rank']) ==
                       (p['reference_person_id'], p['candidate_id'], p['chosen_rank'])]
            require(len(matches) == 1, 'selected candidate mismatch')
            selected = matches[0]
            require(all(selected[k] == p[k] for k in ('final_image', 'final_image_sha256', 'refine_status')), 'final output changed')
            _verified_file(root, p['final_image'], p['final_image_sha256'])
    blocked = manifest.get('blocked_review_case_ids', [])
    require(set(blocked) <= set(ids), 'unknown blocked case ID')
    computed, summary = summarize(rows, blocked_case_ids=blocked)
    origin = config.get('review_origin', config.get('review_status'))
    require(origin == summary['review_status'], 'review provenance mismatch')
    summary.update(mode=config['mode'], review_origin=origin, total_cases=len(rows),
                   rubric_sha256=sha256_file(CONTRACT / 'rubric.json'), manifest_sha256=hash_json(original_manifest),
                   run_config_sha256=hash_json(config), reviews_sha256=hash_json(rows),
                   selected_final_images=sum(len(r['people']) for r in rows),
                   missing_core_artifact_cases=blocked,
                   verified_fbx_count=sum(bool(s.get('fbx')) for c in cases
                       for s in c.get('native_evidence', {}).get('final_selections', [])),
                   refine_timeout_with_valid_fallback_cases=[r['case_id'] for r in rows
                       if any(p['refine_status'] == 'timeout' for p in r['people'])])
    if validate_only:
        return summary
    write_json(output / 'summary.json', summary)
    write_json(output / 'computed-scores.json', computed)
    write_json(output / 'dataset-manifest.json', manifest)
    write_json(output / 'run-config.json', config)
    write_jsonl(output / 'reviews.jsonl', rows)
    write_jsonl(output / 'computed.jsonl', computed)
    pictures = {}
    for r in rows:
        c = by_id[r['case_id']]
        pictures[r['case_id']] = []
        for path in [c['input_image']] + [p['final_image'] for p in r['people']]:
            src = root / path
            suffix = src.suffix.lower()
            require(suffix in ('.png', '.jpg', '.jpeg', '.webp'), 'unsupported review image type')
            dest = Path('images') / (sha256_file(src) + suffix)
            (output / 'images').mkdir(exist_ok=True)
            shutil.copyfile(src, output / dest)
            pictures[r['case_id']].append(dest.as_posix())
    report(output, rows, computed, summary, pictures)
    return summary


def report(output, rows, computed, summary, pictures):
    e = html.escape
    sections = []
    for r, score in zip(rows, computed):
        low, high, maximum = score['score_low'], score['score_high'], score['score_max']
        label = '집계 제외' if low is None else f'{low}–{high} / {maximum}' if low != high else f'{low} / {maximum}'
        images = ''.join(f'<figure><figcaption>{"원본" if i == 0 else e(r["people"][i-1]["reference_person_id"])}</figcaption><img loading="lazy" src="{e(p)}"></figure>' for i,p in enumerate(pictures[r['case_id']]))
        reasons = ''.join(f'<tr><td>{e(p["reference_person_id"])}</td><td>{e(k)}</td><td>{e(v["status"])}</td><td>{e(v["reason"])}</td></tr>' for p in r['people'] for k,v in p['criteria'].items())
        sections.append(f'<details id="{e(r["case_id"], quote=True)}"><summary>{e(r["case_id"])} · {e(r["evaluation_scope"])} · {label} · {e(score["exclusion_reason"] or score["score_kind"])}</summary><div class="images">{images}</div><table>{reasons}</table></details>')
    scope_cards = ''.join(f'<p>{e(k)}: {s["mean_low"]}–{s["mean_high"]} / {s["score_max"]} · 평가 가능 {s["E"]} · 전부 확정 {s["strict_definitive_E"]}</p>' for k,s in summary['scopes'].items())
    atomic_write_text(Path(output) / 'report.html', f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Standin v3.2 검증·재집계</title><style>body{{font:16px system-ui;margin:30px auto;padding:16px;max-width:1200px;color:#203040}}summary{{padding:16px;cursor:pointer;background:#edf2f6}}details{{margin:14px 0}}.images{{display:flex;flex-wrap:wrap}}figure{{margin:10px;max-width:100%}}img{{max-width:100%;width:320px;max-height:500px;object-fit:contain}}td{{border-bottom:1px solid #ddd;padding:8px;overflow-wrap:anywhere}}table{{width:100%;table-layout:fixed}}</style><h1>Standin v3.2 · {summary['N']}컷</h1><p>{e(summary['review_status'])} · 전신100 / 상반신45 별도 집계</p><p>범위는 보류 판정에 따른 경계이며 확률·신뢰구간이 아닙니다. 상한 만점은 확정 만점이 아닙니다.</p>{scope_cards}<p><a href="summary.json">전체 지표</a> · <a href="computed-scores.json">컷별 계산</a></p>{''.join(sections)}</html>''')


def import_v32(root, output, expected_count):
    """Verify all packaged bytes and independently recalculate without executing ZIP code."""
    root, output = Path(root), Path(output)
    require(read_json(root / 'rubric.json') == RUBRIC, 'v3.2 rubric differs from pinned contract')
    require(read_json(root / 'review-record.schema.json') == REVIEW_SCHEMA, 'v3.2 schema differs from pinned contract')
    files = read_json(root / 'artifact-hashes.json')['files']
    paths = [f['path'] for f in files]
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'artifact-hashes.json'}
    require(len(paths) == len(set(paths)) and set(paths) == actual, 'artifact manifest coverage mismatch')
    for f in files:
        asset(root, f)
    manifest = read_json(root / 'dataset-manifest.json')
    rows = read_jsonl(root / 'reviews.jsonl')
    require(len(rows) == len(manifest['cases']) == expected_count, 'fixed case count mismatch')
    for r in rows:
        _schema(r, REVIEW_SCHEMA)  # Incoming native package remains the exact AI-only source contract.
    gaps = read_json(root / 'missing-artifacts.json')
    blocked = gaps['blocked_review_case_ids']
    require(not gaps.get('core_artifacts_complete') or not blocked, 'core missing status contradicts blocked list')
    computed, summary = summarize(rows, blocked_case_ids=blocked)
    source_scores = read_json(root / 'computed-scores.json')
    source_summary = read_json(root / 'summary.json')
    differences = {k: {'submitted': source_summary.get(k), 'computed': v}
                   for k,v in summary.items() if source_summary.get(k) != v}
    config = read_json(root / 'run-config.json')
    require(sha256_file(root / 'rubric.json') == config['active_contract']['rubric_sha256'], 'config rubric hash mismatch')
    require(sha256_file(root / 'review-record.schema.json') == config['active_contract']['schema_sha256'], 'config schema hash mismatch')
    # The native adapter verifies every selection against execution, choice, refine and FBX receipts.
    with tempfile.TemporaryDirectory(prefix='standin-v32-scoring-') as temporary:
        stage = Path(temporary)
        normalized = normalize_manifest(manifest, root)
        normalized['blocked_review_case_ids'] = blocked
        write_json(stage / 'manifest.json', normalized)
        recomputed = score_bundle_v32(stage / 'manifest.json', root / 'reviews.jsonl', root / 'run-config.json',
                                     stage / 'scored', root)
        differences.update({k: {'submitted': source_summary.get(k), 'computed': recomputed[k]}
                           for k in ('selected_final_images', 'verified_fbx_count', 'missing_core_artifact_cases',
                                     'refine_timeout_with_valid_fallback_cases') if source_summary.get(k) != recomputed[k]})
        # Preserve byte-for-byte source and its provenance; publication is after successful validation.
        output.mkdir(parents=True)
        shutil.copytree(root, output / 'package')
        shutil.copytree(stage / 'scored', output / 'scored')
        from .rubric_package import review_queue
        review_queue(output / 'scored', output / 'final-review.jsonl')
        write_json(output / 'submitted-summary.json', source_summary)
        return {'status': 'valid', 'rubric_version': '3.2', 'total_cases': len(rows),
                'review_origin': 'ai_assisted_unreviewed', 'verified_files': len(files),
                'summary_differences': differences, 'computed_scores_equal': source_scores == computed,
                'optional_missing_artifacts': sum(x.get('severity') == 'optional' for x in gaps['entries']),
                'source_programs_executed': False}
