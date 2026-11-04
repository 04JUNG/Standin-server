"""Offline AWS preparation against an isolated, pinned Converter distribution.

Private source assets stay in AWS. A plan names immutable inputs and narrow,
temporary PUT URLs for checkpoint objects; URLs are never logged. This module
does not update a service, grant export permissions, or enable runtime lookup.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixtures(archive, expected_sha256, expected_count, shard, shards, root):
    """Verify the entire archive, then extract only safe source BVHs."""
    digest = hashlib.sha256()
    with Path(archive).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('library archive hash mismatch')
    if not 0 <= shard < shards <= 64:
        raise ValueError('invalid shard')
    result, seen = [], set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not member.name.endswith('.bvh'):
                continue
            if (len(parts) != 2 or parts[0] != 'bvh' or not member.isfile()
                    or member.issym() or member.islnk() or not 0 < member.size <= 2 * 1024 * 1024):
                raise ValueError('unsafe library BVH')
            pid = Path(parts[1]).stem
            if (not pid or Path(pid).name != pid or pid in ('.', '..') or pid in seen
                    or any(c in pid for c in '/\\\0') or any(ord(c) < 32 for c in pid)):
                raise ValueError('invalid or duplicate library pose')
            seen.add(pid)
            if int(sha(pid.encode()), 16) % shards == shard:
                raw = tar.extractfile(member).read(member.size + 1)
                if len(raw) != member.size:
                    raise ValueError('truncated library BVH')
                raw.decode('utf-8')
                path = root / (pid + '.bvh')
                path.write_bytes(raw)
                result.append(dict(id=pid, path=str(path), sha256=sha(raw)))
            # The full archive hash and count are operator-pinned. Avoid
            # decompressing its unrelated 187k thumbnail files a second time.
            if len(seen) == expected_count:
                break
    if len(seen) != expected_count:
        raise ValueError('library pose count mismatch')
    return sorted(result, key=lambda row: row['id'])


def inspect_one(endpoint, identity, path, attempts=2, budget_seconds=14.85):
    from starlette.datastructures import UploadFile
    from converter.hand_screen_catalog import validate_entry
    from converter_api.work_queue import WorkControl, current_work
    raw = Path(path).read_bytes()
    if sha(raw) != identity['source_bvh_sha256']:
        raise ValueError('source changed during preparation')
    hands = identity['hands'] or {}
    for attempt in range(attempts):
        control = WorkControl(kind='screen', budget_seconds=budget_seconds)
        token = current_work.set(control)
        upload = UploadFile(file=io.BytesIO(raw), filename='screen.bvh', size=len(raw))
        try:
            receipt = endpoint(bvh=upload, character_id=identity['character_id'],
                expected_bvh_sha256=identity['source_bvh_sha256'],
                expected_character_sha256=identity['character_sha256'],
                hand_left=hands.get('left'), hand_right=hands.get('right'))
            validate_entry(identity, receipt)
            return dict(binding=identity, receipt=receipt)
        except Exception as error:
            code = getattr(error, 'code', type(error).__name__)
            if not isinstance(code, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,80}', code):
                code = 'INSPECTION_FAILED'
            if attempt + 1 == attempts:
                return dict(unavailable=dict(binding=identity, reason=code))
        finally:
            upload.file.close()
            current_work.reset(token)
        time.sleep(.25)


def put_checkpoint(s3, bucket, output, value):
    """The delegated URL can write only its named immutable staging object."""
    url = output['url']
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or not parsed.hostname.endswith('.amazonaws.com'):
        raise ValueError('invalid output destination')
    raw = encoded(value)
    request = Request(url, data=raw, method='PUT', headers={
        'Content-Type': 'application/json', 'If-None-Match': '*'})
    try:
        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError('checkpoint write failed')
    except Exception as error:
        # A retried write may find the immutable object already committed.
        try:
            old = s3.get_object(Bucket=bucket, Key=output['key'])['Body'].read(len(raw) + 1)
        except Exception:
            raise ValueError('checkpoint write failed') from None
        if old != raw:
            raise ValueError('checkpoint collision') from None
    return dict(key=output['key'], sha256=sha(raw), size=len(raw))


def prepare(plan, s3):
    from converter.hand_screen_catalog import binding, make_catalog
    from converter_api.app import app
    profile = next(route.endpoint for route in app.routes if route.path == '/hand-screen-profile')
    endpoint = next(route.endpoint for route in app.routes if route.path == '/screen-hand-contact')
    before = {c['id']: profile(character_id=c['id']) for c in plan['characters']}
    for c in plan['characters']:
        p = before[c['id']]
        if p['character_sha256'] != c['sha256'] or p.get('shape_sha256', 'missing') is not None:
            raise ValueError('unexpected base character')
    fingerprints = {p['profile_sha256'] for p in before.values()}
    if len(fingerprints) != 1:
        raise ValueError('inconsistent target runtime')
    fingerprint = next(iter(fingerprints))
    print(json.dumps(dict(event='prevalidation_started', shard=plan['shard'],
        shards=plan['shards'], profile_sha256=fingerprint, characters=before,
        workers=plan['workers'])), flush=True)
    started = time.monotonic()
    parts, pending, counts = [], [], Counter()
    with tempfile.TemporaryDirectory(prefix='hand-release-') as temp:
        root = Path(temp)
        archive = root / 'library.tar.gz'
        source = plan['library']
        s3.download_file(source['bucket'], source['key'], str(archive))
        rows = fixtures(archive, source['sha256'], source['pose_count'],
                        plan['shard'], plan['shards'], root)
        if plan.get('limit'):
            rows = rows[:plan['limit']]
        work = [(binding(row['id'], row['sha256'], c['id'], c['sha256'],
                         hands['hands'], fingerprint), row['path'])
                for c in plan['characters'] for hands in plan['hand_settings'] for row in rows]
        if not 1 <= plan['workers'] <= 5:
            raise ValueError('invalid worker capacity')
        print(json.dumps(dict(event='prevalidation_inputs_ready', shard=plan['shard'],
                              poses=len(rows), inspections=len(work))), flush=True)
        def inspect(item):
            if time.monotonic() - started > plan.get('max_seconds', 7200):
                return dict(unavailable=dict(binding=item[0], reason='PREPARATION_DEADLINE'))
            return inspect_one(endpoint, *item, attempts=plan.get('inspection_attempts',2),
                               budget_seconds=plan.get('inspection_budget_seconds',14.85))
        with ThreadPoolExecutor(plan['workers']) as executor:
            for row in executor.map(inspect, work):
                pending.append(row)
                counts['attempted'] += 1
                counts[row.get('receipt', {}).get('status', 'unavailable')] += 1
                if len(pending) == plan['checkpoint_every'] or counts['attempted'] == len(work):
                    # Compare the actual distribution again before publishing.
                    if {c['id']: profile(character_id=c['id']) for c in plan['characters']} != before:
                        raise ValueError('target profile changed during preparation')
                    entries = [(r['binding'], r['receipt']) for r in pending if 'receipt' in r]
                    catalog = make_catalog(entries) if entries else None
                    value = dict(schema='hand-screen-preparation-part.v1', shard=plan['shard'],
                        part=len(parts), profile_sha256=fingerprint,
                        library_sha256=source['sha256'], catalog=catalog,
                        unavailable=[r['unavailable'] for r in pending if 'unavailable' in r])
                    parts.append(put_checkpoint(s3, plan['output_bucket'],
                        plan['parts'][len(parts)], value))
                    pending = []
                    print(json.dumps(dict(event='prevalidation_progress', shard=plan['shard'],
                        total=len(work), counts=dict(counts),
                        elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
    summary = dict(schema='hand-screen-preparation-job.v1', shard=plan['shard'],
        shards=plan['shards'], profile_sha256=fingerprint, library_sha256=source['sha256'],
        counts=dict(counts), parts=parts, target_profiles=before,
        hand_settings=plan['hand_settings'], elapsed_seconds=round(time.monotonic()-started, 2),
        pilot=bool(plan.get('limit')))
    put_checkpoint(s3, plan['output_bucket'], plan['summary'], summary)
    print(json.dumps(dict(event='prevalidation_finished', shard=plan['shard'],
        counts=dict(counts), elapsed_seconds=summary['elapsed_seconds'], pilot=summary['pilot'])), flush=True)
    return summary


def main():
    import argparse
    import boto3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.plan.read_text()), boto3.client('s3'))


if __name__ == '__main__':
    main()
