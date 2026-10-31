"""Install one operator-pinned screening catalog before inference startup."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_CATALOG = 64 * 1024 * 1024
HEX = re.compile(r'[a-f0-9]{64}')
KEY = re.compile(r'hand-screen/releases/[a-z0-9-]+/manifest\.json')


def unique_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate release key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def manifest(raw, expected_sha256):
    if (len(raw) > 65536 or not isinstance(expected_sha256, str)
            or not HEX.fullmatch(expected_sha256)
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError('hand screen release manifest checksum mismatch')
    data = unique_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get('catalog'), dict):
        raise ValueError('invalid hand screen release manifest')
    item = data.get('catalog', {})
    if (data.get('schema') != 'hand-screen-release.v1'
            or data.get('preparation_complete') is not True
            or item.get('path') != 'catalog.json'
            or type(item.get('size')) is not int or not 0 < item['size'] <= MAX_CATALOG
            or not isinstance(item.get('sha256'), str) or not HEX.fullmatch(item['sha256'])
            or type(item.get('entries')) is not int or not 0 < item['entries'] <= 100_000
            or not isinstance(data.get('profile_sha256'), str) or not HEX.fullmatch(data['profile_sha256'])
            or not isinstance(data.get('library_sha256'), str) or not HEX.fullmatch(data['library_sha256'])):
        raise ValueError('invalid hand screen release manifest')
    return data


def install(fetch, env=os.environ):
    from converter.hand_screen_catalog import LibraryScreenCatalog
    key = env['HAND_SCREEN_RELEASE_MANIFEST_KEY']
    if not KEY.fullmatch(key):
        raise ValueError('invalid hand screen release key')
    expected = env['HAND_SCREEN_RELEASE_MANIFEST_SHA256']
    release = manifest(fetch(key, 65536), expected)
    item = release['catalog']
    raw = fetch(key.rsplit('/', 1)[0] + '/' + item['path'], item['size'])
    if len(raw) != item['size'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('hand screen catalog checksum mismatch')
    root = Path(env.get('HAND_SCREEN_RELEASE_ROOT', '/app/hand-screen-release'))
    if root.is_symlink():
        raise ValueError('unsafe hand screen release root')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, prefix='.catalog-', delete=False) as stream:
        path = Path(stream.name)
        stream.write(raw)
    try:
        # Full receipt/lineage validation, beyond the download's file checksum.
        catalog = LibraryScreenCatalog(path, item['sha256'])
        data = unique_json(raw)
        if (len(data['entries']) != item['entries'] or any(
                row['binding']['profile_sha256'] != release['profile_sha256']
                for row in data['entries'].values())):
            raise ValueError('hand screen release profile/count mismatch')
        del catalog
        target = root / 'catalog.json'
        if target.is_symlink():
            raise ValueError('unsafe hand screen catalog target')
        path.chmod(0o444)
        os.replace(path, target)
    finally:
        path.unlink(missing_ok=True)
    receipt = dict(event='hand_screen_release_ready', entries=item['entries'],
        catalog_sha256=item['sha256'], manifest_sha256=expected,
        profile_sha256=release['profile_sha256'])
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    import boto3
    from botocore.config import Config
    client = boto3.client('s3', config=Config(connect_timeout=5, read_timeout=30,
                                            retries={'max_attempts': 2}))
    def fetch(key, limit):
        response = client.get_object(Bucket=os.environ['HAND_SCREEN_RELEASE_BUCKET'], Key=key)
        with response['Body'] as body:
            return body.read(limit + 1)
    install(fetch)


if __name__ == '__main__':
    main()
