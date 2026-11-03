"""Bounded ZIP intake and final human sign-off for rubric 3.1.

Received summary is evidence only; metrics are always recomputed locally.
"""
from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

from .rubric_score import RUBRIC, score_bundle
from .util import hash_json, read_json, read_jsonl, sha256_file, write_json, write_jsonl

REQUIRED = {'rubric.json', 'dataset-manifest.json', 'run-config.json', 'reviews.jsonl',
            'README.md', 'missing-artifacts.json'}


def _extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > 5000 or sum(i.file_size for i in infos) > 2_000_000_000:
            raise ValueError('ZIP exceeds 5000 entries or 2 GB expanded size')
        names = set()
        for i in infos:
            p = PurePosixPath(i.filename)
            if (not i.filename or '\\' in i.filename or p.is_absolute() or '..' in p.parts
                    or ':' in i.filename or i.filename.rstrip('/') != str(p)):
                raise ValueError('unsafe ZIP path')
            mode = i.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise ValueError('ZIP links or special files are forbidden')
            normalized = str(p).casefold()
            if normalized in names:
                raise ValueError('duplicate ZIP entry')
            names.add(normalized)
            if i.flag_bits & 1 or i.file_size > 200_000_000:
                raise ValueError('encrypted or oversized ZIP member')
        for i in infos:
            path = destination / i.filename
            if i.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as src, path.open('xb') as dst:
                    remaining = i.file_size
                    while chunk := src.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError('ZIP member size mismatch')
                        dst.write(chunk)


def import_package(archive, output, *, expected_count=50):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new output directory')
    receipt = {'archive_sha256': sha256_file(archive), 'expected_count': expected_count,
               'status': 'invalid', 'errors': [], 'summary_trusted': False}
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('positive expected_count required')
    with tempfile.TemporaryDirectory(prefix='standin-intake-') as temp:
        temp = Path(temp)
        try:
            _extract(archive, temp)
            manifests = list(temp.rglob('dataset-manifest.json'))
            if len(manifests) != 1:
                raise ValueError('exactly one dataset-manifest.json required')
            root = manifests[0].parent
            missing = REQUIRED - {p.name for p in root.iterdir()}
            if missing:
                raise ValueError(f'missing files: {sorted(missing)}')
            if read_json(root / 'rubric.json').get('rubric_version') == '3.2':
                from .rubric_v32_io import import_v32
                receipt.update(import_v32(root, output, expected_count))
                write_json(output / 'validation.json', receipt)
                return receipt
            if read_json(root / 'rubric.json') != RUBRIC:
                raise ValueError('rubric differs from pinned v3.1 contract')
            manifest = read_json(root / 'dataset-manifest.json')
            if len(manifest['cases']) != expected_count:
                raise ValueError(f'fixed dataset requires {expected_count} cases')
            gaps = read_json(root / 'missing-artifacts.json')
            if gaps not in ([], {'items': []}):
                raise ValueError('missing artifacts declared; resolve them before full-set scoring')
            submitted = read_json(root / 'summary.json') if (root / 'summary.json').exists() else None
            if submitted is not None and not isinstance(submitted, dict):
                raise ValueError('summary.json must be an object')
            result = score_bundle(root / 'dataset-manifest.json', root / 'reviews.jsonl',
                                  root / 'run-config.json', temp / 'computed', root)
            # Keep selected assets portable for later final review, without extracting arbitrary ZIP contents there.
            import shutil
            saved = temp / 'accepted-package'
            saved.mkdir()
            for name in REQUIRED:
                shutil.copyfile(root / name, saved / name)
            paths = {c['input_image'] for c in manifest['cases']}
            paths.update(p['final_image'] for r in read_jsonl(root / 'reviews.jsonl') for p in r['people'])
            for rel in paths:
                dest = saved / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise ValueError('artifact path collides with contract file')
                shutil.copyfile(root / rel, dest)
            output.mkdir(parents=True)
            shutil.copytree(temp / 'computed', output / 'scored')
            shutil.copytree(saved, output / 'package')
            receipt.update(status='valid', total_cases=result['total_cases'], review_origin=result['review_origin'])
            if submitted is not None:
                write_json(output / 'submitted-summary.json', submitted)
                receipt['summary_differences'] = {k: {'submitted': submitted[k], 'computed': result[k]}
                    for k in submitted.keys() & result.keys() if submitted[k] != result[k]}
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            receipt['errors'].append(f'{type(error).__name__}: {error}')
        write_json(output / 'validation.json', receipt)
    return receipt


def review_queue(scored, output):
    """Every case gets a final-review row; no auto-approval of confident guesses."""
    scored, output = Path(scored), Path(output)
    if output.exists():
        raise FileExistsError('use a new queue file')
    rows = read_jsonl(scored / 'reviews.jsonl')
    queue = [{'case_id': r['case_id'], 'review_sha256': hash_json(r), 'decision': 'pending',
              'replacement': None, 'reviewer': None, 'reviewed_at': None} for r in rows]
    write_jsonl(output, queue)
    return {'cases': len(queue), 'auto_approved': 0}


def finalize_review(scored, decisions, artifacts_root, output):
    scored, output = Path(scored), Path(output)
    source = read_jsonl(scored / 'reviews.jsonl')
    decisions = read_jsonl(decisions)
    index = {d['case_id']: d for d in decisions}
    if len(index) != len(decisions) or set(index) != {r['case_id'] for r in source}:
        raise ValueError('one final decision per case required')
    corrected, history = [], []
    for original in source:
        d = index[original['case_id']]
        if d['review_sha256'] != hash_json(original):
            raise ValueError('stale final-review decision')
        if d['decision'] not in ('approve', 'replace'):
            raise ValueError('all cases need explicit final approval or correction')
        if any(not isinstance(d.get(k), str) or not d[k].strip() for k in ('reviewer', 'reviewed_at')):
            raise ValueError('reviewer and reviewed_at required')
        row = copy.deepcopy(original if d['decision'] == 'approve' else d['replacement'])
        if row['case_id'] != original['case_id'] or row['run_id'] != original['run_id']:
            raise ValueError('cannot change case/run identity')
        # Review may correct judgements only, not select a more favourable output.
        def identity(r):
            return {**r, 'people': [{k: v for k, v in p.items() if k not in ('criteria', 'reviewer')}
                                    for p in r['people']]}
        if identity(row) != identity(original):
            raise ValueError('final review cannot change frozen outputs or product status')
        for p in row['people']:
            p['reviewer'] = d['reviewer']
        if row.get('rubric_version') == '3.2':
            row['review_status'] = 'human_reviewed'
        corrected.append(row)
        history.append({**d, 'replacement': None, 'final_review_sha256': hash_json(row)})
    config = read_json(scored / 'run-config.json')
    before_origin = config.get('review_origin', config.get('review_status'))
    config.update(review_origin='human_reviewed', official_comparison=False,
                  parent_reviews_sha256=hash_json(source))
    if config.get('rubric_version') == '3.2':
        config.update(review_status='human_reviewed', human_reviewed=True)
    with tempfile.TemporaryDirectory(prefix='standin-final-') as temp:
        temp = Path(temp)
        write_json(temp / 'config.json', config)
        write_jsonl(temp / 'reviews.jsonl', corrected)
        result = score_bundle(scored / 'dataset-manifest.json', temp / 'reviews.jsonl', temp / 'config.json', output, artifacts_root)
    write_jsonl(output / 'review-history.jsonl', history)
    from .rubric_score import WEIGHTS, ASSESSED
    comparisons = []
    for before, after in zip(source, corrected):
        for a, b in zip(before['people'], after['people']):
            for k in WEIGHTS:
                automatic, human = a['criteria'][k]['status'], b['criteria'][k]['status']
                if automatic == human == 'not_applicable':
                    continue
                comparisons.append({'case_id': before['case_id'], 'person_id': a['reference_person_id'],
                                    'part': k, 'before': automatic, 'human': human})
    scored_pairs = [r for r in comparisons if r['before'] in ASSESSED and r['human'] in ASSESSED]
    positives = [r for r in scored_pairs if r['human'] == 'mismatch']
    negatives = [r for r in scored_pairs if r['human'] == 'match']
    def rate(n, d):
        return {'numerator': n, 'denominator': d, 'rate': n / d if d else None}
    write_json(output / 'review-feedback.json', {
        'origin': 'final_human_review_not_independent_holdout', 'automatic_retraining': False,
        'before_origin': before_origin,
        'target_estimates_promoted_to_gold': False,
        'status_agreement': rate(sum(r['before'] == r['human'] for r in comparisons), len(comparisons)),
        'assessed_pair_coverage': rate(len(scored_pairs), len(comparisons)),
        'false_deduction_rate': rate(sum(r['before'] == 'mismatch' for r in negatives), len(negatives)),
        'missed_error_rate': rate(sum(r['before'] == 'match' for r in positives), len(positives)),
        'changes': [r for r in comparisons if r['before'] != r['human']]})
    return result


def audit_judge(reference, automatic, output):
    """Compare AI against human reviews of identical input/output pairs, not new cuts."""
    reference, automatic, output = Path(reference), Path(automatic), Path(output)
    if output.exists():
        raise FileExistsError('use a new audit directory')
    gold_config = read_json(reference / 'run-config.json')
    ai_config = read_json(automatic / 'run-config.json')
    if gold_config.get('review_origin', gold_config.get('review_status')) != 'human_reviewed' or ai_config.get('review_origin', ai_config.get('review_status')) != 'ai_assisted_unreviewed':
        raise ValueError('requires human-reviewed reference and unreviewed AI results')
    gold_rows = read_jsonl(reference / 'reviews.jsonl')
    gold = {r['case_id']: r for r in gold_rows}
    ai_rows = read_jsonl(automatic / 'reviews.jsonl')
    ai = {r['case_id']: r for r in ai_rows}
    if set(gold) != set(ai) or len(ai) != len(ai_rows) or len(gold) != len(gold_rows):
        raise ValueError('audit requires identical fixed cases')
    def input_hashes(root):
        return {r['case_id']: r['input_sha256'] for r in read_json(root / 'dataset-manifest.json')['cases']}
    if input_hashes(reference) != input_hashes(automatic):
        raise ValueError('audit input images differ')
    from .rubric_score import WEIGHTS, score_case
    if gold_config.get('rubric_version') != ai_config.get('rubric_version'):
        raise ValueError('audit rubric versions differ')
    if gold_config.get('rubric_version') == '3.2':
        from .rubric_v32 import score_case
    comparisons = []
    for cid, a in ai.items():
        g = gold[cid]
        if a.get('evaluation_scope') != g.get('evaluation_scope'):
            raise ValueError('audit evaluation scope differs')
        score_case(a); score_case(g)
        def identities(r):
            return sorted((p['reference_person_id'], p['candidate_id'], p['chosen_rank'],
                           p['final_image_sha256'], p['refine_status']) for p in r['people'])
        if identities(a) != identities(g) or (a['result_status'], a['expected_people']) != (g['result_status'], g['expected_people']):
            raise ValueError('audit requires identical frozen outputs and product status')
        gp = {p['reference_person_id']: p for p in g['people']}
        for p in a['people']:
            for k in WEIGHTS:
                if p['criteria'][k]['status'] == gp[p['reference_person_id']]['criteria'][k]['status'] == 'not_applicable':
                    continue
                comparisons.append({'case_id': cid, 'person_id': p['reference_person_id'], 'part': k,
                                    'automatic': p['criteria'][k]['status'],
                                    'human': gp[p['reference_person_id']]['criteria'][k]['status']})
    counts = {}
    for row in comparisons:
        key = row['human'] + ' -> ' + row['automatic']
        counts[key] = counts.get(key, 0) + 1
    result = {'cases': len(ai), 'judgements': len(comparisons), 'confusion_human_to_ai': counts,
              'agreement': sum(r['human'] == r['automatic'] for r in comparisons) / len(comparisons) if comparisons else None,
              'reference_sha256': hash_json(gold), 'automatic_sha256': hash_json(ai),
              'evaluation_scope': 'known_reference_set_not_generalization',
              'automatic_promotion_or_training': False,
              'disagreements': [r for r in comparisons if r['human'] != r['automatic']]}
    write_json(output / 'judge-audit.json', result)
    return result
