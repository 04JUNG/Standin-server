"""Bind the handoff's rough, person slot, selected BVH, and final FBX by hashes."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
from zipfile import ZipFile

from experiments.refine_v5.ab_common import load_json, save_json


def unique(rows, key):
    result = {}
    for row in rows:
        name = key(row)
        if name in result:
            raise ValueError(f'duplicate record: {name}')
        result[name] = row
    return result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def store(root, kind, name, data):
    destination = root / kind / name
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f'artifact name collides: {destination}')
    destination.write_bytes(data)
    return str(destination.relative_to(root))


def intake(archive, old_data, out):
    import json

    old = unique(load_json(old_data / 'curated_inventory_v2.json'), lambda r: r['case_id'])
    out.mkdir(parents=True, exist_ok=False)
    for kind in ('rough', 'search_input', 'original_bvh', 'ground_truth'):
        (out / kind).mkdir()
    with ZipFile(archive) as bundle:
        entries = bundle.namelist()
        if len(set(entries)) != len(entries):
            raise ValueError('archive has duplicate paths')
        cuts = [json.loads(row) for row in bundle.read('cuts.jsonl').splitlines() if row.strip()]
        assets = json.loads(bundle.read('asset_manifest.json'))
        inventory = unique(assets['ground_truth_fbx'], lambda r: r['zip_path'])
        slots = unique(assets['selections'], lambda r: (r['rough_id'], r['person_index']))
        cut_index = unique(cuts, lambda r: r['rough_id'])
        selected = {}
        for key, item in slots.items():
            cut = cut_index[item['rough_id']]
            person = next(p for p in cut['people'] if p['person_index'] == item['person_index'])
            candidate = next(c for c in person['candidates'] if c['candidate_id'] == person['selected_candidate_id'])
            if (cut['analysis_job']['job_id'] != item['job_id'] or candidate['pose_id'] != item['selected_pose_id']
                    or candidate['rank'] != item['selected_rank'] or not candidate['is_library_candidate']
                    or candidate['library_bvh_sha256'] != item['selected_bvh_sha256']):
                raise ValueError(f'selection evidence differs: {key}')
            path = item['selected_bvh_zip_path']
            bvh = bundle.read(path)
            if sha(bvh) != item['selected_bvh_sha256']:
                raise ValueError(f'selected BVH hash differs: {key}')
            chosen_path = store(out, 'original_bvh', Path(path).name, bvh)
            evidence = dict(item, verified_selected_bvh=chosen_path)
            if item['handoff_starter_match'] != 'same_as_selected':
                historical = bundle.read(item['handoff_starter_bvh_zip_path'])
                if sha(historical) != item['handoff_starter_bvh_sha256']:
                    raise ValueError(f'handoff starter hash differs: {key}')
                evidence['verified_handoff_bvh'] = store(out, 'original_bvh', 'handoff-' +
                                                          Path(item['handoff_starter_bvh_zip_path']).name, historical)
            selected[key] = evidence
        rows = []
        for cut in cuts:
            search = bundle.read(cut['input_rough_path'])
            search_hash = sha(search)
            if search_hash != cut['image_sha256']:
                raise ValueError(f'search image hash differs: {cut["rough_id"]}')
            search_path = store(out, 'search_input', Path(cut['input_rough_path']).name, search)
            for final in cut['final_files']:
                cid = final['sample_id']
                prior = old[cid]
                if prior['original_cut_id'] != f"{cut['rough_number']:03d}":
                    raise ValueError(f'cut differs from previous inventory: {cid}')
                path = final['fbx_path']
                gt = bundle.read(path)
                blob = hashlib.sha1(f'blob {len(gt)}\0'.encode() + gt).hexdigest()
                entry = inventory[path]
                if (blob != entry['github_blob_sha'] or len(gt) != entry['size_bytes']
                        or sha(gt) != prior['files']['gt']['sha256']):
                    raise ValueError(f'GT differs from pinned inventory: {cid}')
                rough = (old_data / prior['image']).read_bytes()
                if sha(rough) != prior['files']['image']['sha256'] or Path(prior['image']).name != final['rough_filename']:
                    raise ValueError(f'final rough differs from pinned inventory: {cid}')
                person = final['person_index']
                item = selected.get((cut['rough_id'], person))
                if item and (item['final_sample_id'] != cid or item['final_fbx_zip_path'] != path):
                    raise ValueError(f'person-to-final mapping differs: {cid}')
                if person is None:
                    status = 'ambiguous_person_slot'
                elif item is None:
                    status = 'no_selected_seed'
                elif item['selection_source'] == 'later_local_save':
                    status = 'later_selection_not_confirmed'
                elif item['handoff_starter_match'] != 'same_as_selected':
                    status = 'handoff_source_conflict'
                else:
                    status = 'original_library_bvh_confirmed'
                rows.append(dict(
                    case_id=cid, rough_id=cut['rough_id'], person_index=person, status=status,
                    original_cut_id=prior['original_cut_id'], source_risk_group=prior['source_risk_group'],
                    split=prior['split'], split_certification=prior['split_certification'],
                    representation=prior['representation'], scope=prior['scope'],
                    target_identity=prior['target_identity'], part_visibility=prior['part_visibility'],
                    rough=store(out, 'rough', final['rough_filename'], rough), rough_sha256=sha(rough),
                    search_input=search_path, search_input_sha256=search_hash,
                    rough_matches_search=sha(rough) == search_hash,
                    ground_truth=store(out, 'ground_truth', final['fbx_filename'], gt), ground_truth_sha256=sha(gt),
                    selection=item, mapping_note=final['mapping_note'],
                ))
    if len(unique(rows, lambda row: row['case_id'])) != len(old) or len(rows) != len(inventory):
        raise ValueError('missing final samples')
    groups = {}
    for row in rows:
        groups.setdefault(row['source_risk_group'], set()).add(row['split'])
    if any(len(split) > 1 for split in groups.values()):
        raise ValueError('same rough group crosses splits')
    approved = [r for r in rows if r['status'] == 'original_library_bvh_confirmed']
    by_split = Counter(r['split'] for r in approved)
    p1 = [r for r in approved if r['split'] != 'test' and r['target_identity'] != 'ambiguous']
    result = dict(schema='refine-v5-original-seed-v1', archive_sha256=archive_sha(archive),
                  gt_commit=assets['github_commit'], search_inputs=len(cuts), selections=len(slots),
                  final_samples=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
                  confirmed_by_split=dict(by_split), p1_eligible=len(p1),
                  rough_bytes_match_search=sum(r['rough_matches_search'] for r in rows), rows=rows)
    save_json(out / 'manifest.json', result)
    save_json(out / 'p1-cohort.json', dict(
        manifest_sha256=archive_sha(out / 'manifest.json'), test_split_used=False,
        case_ids=[r['case_id'] for r in p1],
    ))
    print({k: v for k, v in result.items() if k != 'rows'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'old-data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    intake(args.archive, args.old_data, args.out)
