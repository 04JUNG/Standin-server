"""Import the authored VLM pose suite without inventing assist-target labels.

This recognition dataset deliberately does not set legacy near/eligible/route
flags: COCO17 annotation and product usefulness are different populations.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from src.schema import COCO17
from .util import hash_json, image_dimensions, read_json, read_jsonl, sha256_file, write_json, write_jsonl

SCHEMA = 'pose-recognition-dataset-v1'


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('source paths must be relative')
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('source path escapes the selected suite')
    return path


def validate_points(person):
    points, visibility = person.get('keypoints'), person.get('visibility')
    if not isinstance(points, list) or len(points) != 17 or not isinstance(visibility, list) or len(visibility) != 17:
        raise ValueError('GT requires 17 keypoints and visibility values')
    if any(type(v) is not int or v not in (0, 1, 2) for v in visibility):
        raise ValueError('GT visibility must be 0/1/2')
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or any(
                x is not None and (type(x) not in (int, float) or not math.isfinite(x)) for x in point):
            raise ValueError('GT coordinates must be finite pixels or null')


def import_pose_suite(source, output, *, source_commit, usage_authorized=False, set_number=1):
    """Only dataset.cuts defines inclusion. Historical metadata paths are data."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not usage_authorized:
        raise ValueError('explicit user authorization for evaluation use required')
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('pin the 40-character source Git commit')
    if output == source or source in output.parents:
        raise ValueError('preserve the original source tree')
    if output.exists():
        raise FileExistsError('dataset output must be a new version')
    index = read_json(source / 'index.json')
    if index.get('schema_version') != 'standin-vlm-pose-suite-v1':
        raise ValueError('unsupported suite schema')
    entries = [s for s in index['sets'] if s.get('set') == set_number]
    if len(entries) != 1:
        raise ValueError('select exactly one suite set')
    manifest_path = contained(source, entries[0]['manifest'])
    manifest = read_json(manifest_path)
    if manifest.get('schema_version') != 'standin-vlm-pose-dataset-v1':
        raise ValueError('unsupported source dataset schema')
    selected = manifest['cuts']
    ids = [c['id'] for c in selected]
    if any(not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', cid) or cid in {'.','..'} for cid in ids):
        raise ValueError('cut IDs must be safe stable identifiers, not paths')
    excluded = set(index.get('excluded_cut_ids', []))
    for revision in manifest.get('revision_history', []):
        excluded.update(revision.get('excluded_cut_ids', []))
    if len(ids) != len(set(ids)) or not ids or set(ids) & excluded:
        raise ValueError('duplicate or explicitly excluded active cut')
    if manifest['selection_policy']['cuts'] != len(selected):
        raise ValueError('active cut denominator differs from selection policy')
    if entries[0].get('active_cut_count', len(selected)) != len(selected):
        raise ValueError('suite and selected manifest disagree')
    cuts, persons, source_files = [], [], {}
    source_files['index.json'] = sha256_file(source / 'index.json')
    source_files[manifest_path.relative_to(source).as_posix()] = sha256_file(manifest_path)
    seen_images = set()
    for cut in selected:
        if type(cut.get('person_count')) is not int or cut['person_count'] < 0:
            raise ValueError('explicit nonnegative person count required')
        image = contained(manifest_path.parent, cut['image_path'])
        digest = sha256_file(image)
        if digest != cut['image_sha256'] or digest in seen_images:
            raise ValueError('source image SHA mismatch or duplicate content')
        seen_images.add(digest)
        if image_dimensions(image) != (cut['width'], cut['height']):
            raise ValueError('source image dimensions differ')
        source_files[image.relative_to(source).as_posix()] = digest
        gt_path = contained(manifest_path.parent, cut['gt_path']) if cut.get('gt_path') else None
        gt = read_json(gt_path) if gt_path else None
        if gt:
            if gt.get('schema_version') != 'standin-coco17-gt-draft-v1' or gt.get('keypoint_names') != COCO17:
                raise ValueError('GT must explicitly use the COCO17 order')
            if gt.get('coordinate_system', {}) != dict(unit='pixel', origin='top_left', x_direction='right', y_direction='down', left_right='subject_anatomical'):
                raise ValueError('unsupported GT coordinate system')
            if gt['image'].get('sha256') != digest or (gt['image'].get('width'), gt['image'].get('height')) != (cut['width'], cut['height']):
                raise ValueError('GT bound to a different image')
            if contained(manifest_path.parent, str(gt_path.parent.relative_to(manifest_path.parent) / gt['image']['relative_path'])) != image:
                raise ValueError('GT relative image mapping differs')
            if len(gt['people']) != cut['person_count']:
                raise ValueError('GT and manifest person denominator disagree')
            source_files[gt_path.relative_to(source).as_posix()] = sha256_file(gt_path)
        cuts.append(dict(cut_id=cut['id'], image_path=str(image), image_sha256=digest,
            width=cut['width'], height=cut['height'], person_count=cut['person_count'],
            visible_ranges=cut.get('visible_ranges', []), expression_styles=cut.get('expression_styles', []),
            gt_status=cut.get('gt_status'), human_verified=gt.get('human_verified') is True if gt else False,
            evaluation_ready=gt.get('evaluation_ready') is True if gt else False))
        local_ids = set()
        if gt is None:
            # Unknown identities remain explicit pending slots, not annotated people.
            for i in range(cut['person_count']):
                persons.append(dict(cut_id=cut['id'], person_id=f"{cut['id']}:pending:{i}", annotation_id=None,
                    keypoints=[[None, None] for _ in range(17)], visibility=[0]*17, gt_missing=True))
            continue
        for person in gt['people']:
            validate_points(person)
            pid = person.get('person_id')
            if type(pid) not in (str, int) or pid in local_ids:
                raise ValueError('explicit unique annotation person_id required')
            local_ids.add(pid)
            persons.append(dict(cut_id=cut['id'], person_id=f"{cut['id']}:gt:{pid}", annotation_id=pid,
                keypoints=person['keypoints'], visibility=person['visibility'], gt_missing=False,
                review_status=person.get('review_status'), human_verified=gt.get('human_verified') is True,
                evaluation_ready=gt.get('evaluation_ready') is True,
                human_confirmed_keypoint_indices=person.get('human_confirmed_keypoint_indices', [])))
    if len(persons) != manifest['selection_policy']['people']:
        raise ValueError('fixed person denominator differs from source policy')
    frozen = dict(schema_version=SCHEMA, dataset_id=manifest['dataset_id'], set_number=set_number,
        source_repository='04JUNG/Standin-master-docs', source_commit=source_commit,
        source_manifest=manifest_path.relative_to(source).as_posix(), source_files=source_files,
        source_root=str(source), excluded_cut_ids=sorted(excluded), keypoint_names=COCO17,
        counts=dict(cuts=len(cuts), persons=len(persons), visible_joints=sum(p['visibility'].count(2) for p in persons),
                    occluded_estimated_joints=sum(p['visibility'].count(1) for p in persons)),
        usage_authorization=dict(evaluation='allow', internal_sharing='allow', external_vlm_input='allow',
            issuer='user', statement='다 써도되는 자료야', real_provider_calls_authorized_by_this_import=False),
        ground_truth_status=manifest.get('ground_truth'), cuts_sha256=hash_json(cuts), persons_sha256=hash_json(persons))
    output.mkdir(parents=True)
    write_jsonl(output/'cuts.jsonl', cuts)
    write_jsonl(output/'persons.jsonl', persons)
    write_json(output/'dataset.json', {**frozen, 'seal_sha256':hash_json(frozen)})
    return load_pose_dataset(output)


def load_pose_dataset(root):
    root = Path(root).resolve()
    frozen = read_json(root/'dataset.json')
    payload = {k:v for k,v in frozen.items() if k != 'seal_sha256'}
    if frozen.get('schema_version') != SCHEMA or frozen.get('seal_sha256') != hash_json(payload):
        raise ValueError('recognition dataset seal mismatch')
    cuts, persons = read_jsonl(root/'cuts.jsonl'), read_jsonl(root/'persons.jsonl')
    if hash_json(cuts) != frozen['cuts_sha256'] or hash_json(persons) != frozen['persons_sha256']:
        raise ValueError('recognition annotation seal mismatch')
    for rel, expected in frozen['source_files'].items():
        if sha256_file(contained(Path(frozen['source_root']), rel)) != expected:
            raise ValueError('source changed after recognition dataset import')
    return frozen, cuts, persons
