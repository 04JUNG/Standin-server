"""Assemble complete offline checkpoints inside AWS, never copy source assets."""
from collections import Counter
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converter.hand_screen_catalog import binding, digest, make_catalog, validate_entry
from converter.hand_library import normalize_request
from scripts.prevalidate_hand_screen_batch import encoded, put_checkpoint


def assemble(summaries, read, *, expected_pose_count, expected_characters, expected_hands,
             retry_summaries=()):
    if not summaries or any(s.get('pilot') is not False for s in summaries):
        raise ValueError('pilot data cannot become a runtime release')
    shards = summaries[0]['shards']
    if (len(summaries) != shards or {s['shard'] for s in summaries} != set(range(shards))
            or any(s.get('schema') != 'hand-screen-preparation-job.v1' or s['shards'] != shards for s in summaries)):
        raise ValueError('incomplete or duplicate shard set')
    profiles = {s['profile_sha256'] for s in summaries}
    libraries = {s['library_sha256'] for s in summaries}
    if len(profiles) != 1 or len(libraries) != 1:
        raise ValueError('incompatible preparation distributions')
    profile = next(iter(profiles))
    pairs, seen, counts = [], set(), Counter()
    failures = {}
    bodies = {(c['id'], c['sha256']) for c in expected_characters}
    hands = [normalize_request(h['hands']) for h in expected_hands]
    def rows(summary):
        if summary['hand_settings'] != expected_hands:
            raise ValueError('unexpected prepared hand settings')
        actual = {k: v['character_sha256'] for k, v in summary['target_profiles'].items()}
        if actual != {c['id']: c['sha256'] for c in expected_characters}:
            raise ValueError('unexpected prepared base bodies')
        local = Counter()
        for index, part in enumerate(summary['parts']):
            raw = read(part['key'], part['size'])
            if len(raw) != part['size'] or hashlib.sha256(raw).hexdigest() != part['sha256']:
                raise ValueError('preparation checkpoint checksum mismatch')
            value = json.loads(raw)
            if (value.get('schema') != 'hand-screen-preparation-part.v1'
                    or value['shard'] != summary.get('shard', 0) or value['part'] != index
                    or value['profile_sha256'] != profile
                    or value['library_sha256'] != summary['library_sha256']):
                raise ValueError('checkpoint source mismatch')
            for key, row in (value.get('catalog') or {}).get('entries', {}).items():
                validate_entry(row['binding'], row['receipt'])
                identity = row['binding']
                if (key != digest(row['binding'])
                        or identity['profile_sha256'] != profile
                        or (identity['character_id'], identity['character_sha256']) not in bodies
                        or identity['hands'] not in hands):
                    raise ValueError('duplicate or incompatible certificate')
                local['attempted'] += 1
                local[row['receipt']['status']] += 1
                yield key, identity, row['receipt']
            for failure in value['unavailable']:
                identity = failure['binding']
                expected_identity = binding(identity['pose_id'], identity['source_bvh_sha256'],
                    identity['character_id'], identity['character_sha256'], identity['hands'], profile)
                if (identity != expected_identity or identity['hands'] not in hands
                        or (identity['character_id'], identity['character_sha256']) not in bodies
                        or any(not re.fullmatch('[a-f0-9]{64}', identity[name]) for name in (
                            'source_bvh_sha256','character_sha256','profile_sha256','hand_library_sha256'))):
                    raise ValueError('invalid unavailable source binding')
                local['attempted'] += 1
                local['unavailable'] += 1
                yield digest(identity), identity, None
        if dict(local) != {k: v for k, v in summary['counts'].items() if v}:
            raise ValueError('preparation checkpoint counts mismatch')
    for summary in summaries:
        for key, identity, receipt in rows(summary):
            if key in seen:raise ValueError('duplicate preparation identity')
            seen.add(key)
            counts['attempted'] += 1
            if receipt is None:failures[key]=identity
            else:pairs.append((identity,receipt))
    expected = expected_pose_count * len(expected_characters) * len(expected_hands)
    if counts['attempted'] != expected:
        raise ValueError('incomplete preparation coverage')
    repaired = set()
    for summary in retry_summaries:
        if (summary.get('schema') != 'hand-screen-preparation-retry.v1'
                or summary['profile_sha256'] != profile
                or summary['library_sha256'] != next(iter(libraries))):
            raise ValueError('retry distribution mismatch')
        for key, identity, receipt in rows(summary):
            if receipt is None:
                raise ValueError('unavailable retry cannot become a certificate')
            if key in repaired or failures.get(key) != identity:
                raise ValueError('retry does not match an original unavailable inspection')
            repaired.add(key)
            pairs.append((identity,receipt))
    if set(failures) != repaired:
        raise ValueError('unavailable inspections remain; no complete release published')
    coverage, source_hashes = {}, {}
    counts = Counter(attempted=len(pairs))
    for identity, receipt in pairs:
        group = (identity['character_id'],digest(identity['hands']))
        coverage.setdefault(group,set()).add(identity['pose_id'])
        pid = identity['pose_id']
        if pid in source_hashes and source_hashes[pid] != identity['source_bvh_sha256']:
            raise ValueError('body certificates disagree on source pose bytes')
        source_hashes[pid] = identity['source_bvh_sha256']
        counts[receipt['status']] += 1
    if (len(pairs) != expected or len(coverage) != len(expected_characters)*len(expected_hands)
            or any(len(ids) != expected_pose_count for ids in coverage.values())
            or len({frozenset(ids) for ids in coverage.values()}) != 1):
        raise ValueError('inconsistent body/hand pose coverage')
    catalog = encoded(make_catalog(pairs))
    if len(catalog) > 64 * 1024 * 1024:
        raise ValueError('runtime catalog exceeds size limit')
    manifest = dict(schema='hand-screen-release.v1', preparation_complete=True,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        library_sha256=next(iter(libraries)), profile_sha256=profile,
        catalog=dict(path='catalog.json', size=len(catalog),
                     sha256=hashlib.sha256(catalog).hexdigest(), entries=expected),
        scope=dict(pose_count=expected_pose_count, characters=expected_characters,
                   hand_settings=expected_hands, base_shapes_only=True),
        counts=dict(counts), reinspected_entries=len(repaired))
    return catalog, manifest


def existing_manifest(fresh, previous):
    """A lost acknowledgement may be retried without replacing a release."""
    new = dict(fresh);old = dict(previous)
    new.pop('created_at',None);old.pop('created_at',None)
    if new != old:
        raise ValueError('immutable release manifest collision')
    return previous


def main():
    import argparse
    import boto3
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    s3 = boto3.client('s3')
    def read(key, size):
        response = s3.get_object(Bucket=plan['bucket'], Key=key)
        with response['Body'] as body:
            return body.read(size+1)
    summaries = [json.loads(read(key, 1024*1024)) for key in plan['summaries']]
    retries = [json.loads(read(key, 1024*1024)) for key in plan.get('retry_summaries', [])]
    raw, manifest = assemble(summaries, read, expected_pose_count=plan['pose_count'],
        expected_characters=plan['characters'], expected_hands=plan['hand_settings'],
        retry_summaries=retries)
    # put_checkpoint uses canonical JSON; this roundtrip retains exactly those bytes.
    cat = put_checkpoint(s3, plan['bucket'], plan['catalog'], json.loads(raw))
    assert cat['sha256'] == manifest['catalog']['sha256']
    try:
        prior=json.loads(read(plan['manifest']['key'],65536))
    except s3.exceptions.NoSuchKey:
        prior=None
    if prior is not None:
        manifest=existing_manifest(manifest,prior)
    result = put_checkpoint(s3, plan['bucket'], plan['manifest'], manifest)
    print(json.dumps(dict(event='hand_screen_release_published', manifest=result,
                         catalog=cat, entries=manifest['catalog']['entries'],
                         profile_sha256=manifest['profile_sha256'], counts=manifest['counts'])), flush=True)


if __name__ == '__main__':
    main()
