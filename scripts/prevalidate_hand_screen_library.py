"""Explicit offline library release preparation; no runtime traffic is learned.

The manifest contains fixtures [{id,bvh,sha256}] and optional hand_settings
[{id,hands}]. Paths are relative to that manifest. Default hands are unchanged.
Use a Converter matching the intended deployment runtime; Mac certificates are
intentionally incompatible with Linux. This command never deploys its output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from converter.hand_screen_catalog import binding, make_catalog
from converter.hand_library import load_library
from src.hand_skin_screen import HandSkinClient


def prepare(client, manifest, base):
    before=client.screening_profile(timeout=5.)
    if not isinstance(before,dict) or not before.get('profile_sha256'):
        raise ValueError('Converter profile unavailable')
    if (before.get('character_id')!=client.character_id
            or before.get('character_sha256')!=client.character_sha256
            or before.get('hand_library_sha256')!=load_library().sha256
            or 'shape_sha256' not in before or before['shape_sha256'] is not None):
        raise ValueError('Converter body/profile binding mismatch')
    entries=[]
    for preset in manifest.get('hand_settings',[{'id':'original','hands':None}]):
        for fixture in manifest['fixtures']:
            pid=fixture['id']
            if not pid or Path(pid).name!=pid or '/' in pid or '\\' in pid or pid in ('.','..'):
                raise ValueError('unsafe library pose id')
            raw=(base/fixture['bvh']).read_bytes()
            sha=hashlib.sha256(raw).hexdigest()
            if sha!=fixture['sha256']:
                raise ValueError('library BVH hash mismatch')
            receipt=client.check(raw.decode('utf-8'),preset['hands'],timeout=15.)
            entries.append((binding(pid,sha,client.character_id,client.character_sha256,
                preset['hands'],before['profile_sha256']),receipt))
    if client.screening_profile(timeout=5.)!=before:
        raise ValueError('Converter profile changed during offline validation')
    return make_catalog(entries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--converter-url',required=True)
    p.add_argument('--character-id',required=True)
    p.add_argument('--character-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    client=HandSkinClient(args.converter_url,args.character_id,args.character_sha256)
    catalog=prepare(client,json.loads(args.manifest.read_text()),args.manifest.resolve().parent)
    raw=(json.dumps(catalog,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    with args.output.open('xb') as stream:stream.write(raw)
    print(json.dumps({'entries':len(catalog['entries']),
        'sha256':hashlib.sha256(raw).hexdigest(),'deployed':False}))


if __name__=='__main__':main()
