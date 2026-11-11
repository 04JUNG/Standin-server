"""Exercise a real local converter and Blender using a private JSON fixture.

python scripts/verify_outline_http.py --fixture /private/path/fixture.json
The fixture contains source paths/fields and a token_file path, never a token.
This is a local acceptance check; it does not enable or deploy any service.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

import httpx


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    args=parser.parse_args()
    fixture=json.loads(args.fixture.read_text())
    destination=Path(fixture['output']);destination.mkdir(parents=True,exist_ok=True)
    token=Path(fixture['token_file']).read_text().strip()
    results=[]
    with httpx.Client(base_url=fixture['url'],timeout=360,trust_env=False) as client:
        capability=client.get('/outline-contract').raise_for_status().json()
        assert capability['enabled'] and capability['csp_certified']
        assert capability['acceptance_scope']=='posed-fbx-import'
        def request(case,*,changes=None,authenticated=True):
            source=Path(case['source']).read_bytes()
            assert digest(source)==case['fields']['source_artifact_sha256']
            fields={**case['fields'],**(changes or {})}
            fields['appearance']=json.dumps(fields['appearance'],sort_keys=True)
            return client.post('/outline-fbx',data=fields,
                files={'fbx':('source.fbx',source,'application/octet-stream')},
                headers={'Authorization':'Bearer '+token} if authenticated else {})
        for case in fixture['cases']:
            start=time.monotonic();response=request(case);response.raise_for_status()
            body=response.json();receipt=body['receipt']
            data=base64.b64decode(body['fbx_base64'],validate=True)
            assert body['artifact_contract']=='outlined-fbx-v1'
            assert digest(data)==body['fbx_sha256']==receipt['fbx_sha256']
            if case.get('expected_sha256'):assert digest(data)==case['expected_sha256']
            for name,value in case['fields'].items():assert receipt[name]==value,(name,case['name'])
            for name in ('expression','mapping','runtime_revision'):
                assert receipt[name]==capability[name]
            assert receipt['file_verified'] and receipt['csp_certified']
            assert receipt['outline']['manifest']['surface']=='outward-transparent'
            assert receipt['outline']['skin_verified'] and receipt['outline']['displacement_verified']
            assert receipt['base']['typed_fbx_values_preserved']
            assert all(value==0 for key,value in receipt['base'].items() if key.startswith('max'))
            path=destination/(case['name']+'.fbx');path.write_bytes(data)
            (destination/(case['name']+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            results.append(dict(name=case['name'],sha256=digest(data),bytes=len(data),
                seconds=time.monotonic()-start,path=str(path.resolve()),receipt=receipt))
        case=fixture['cases'][0]
        negative=[('zero',dict(appearance=dict(version='fbx-outline-v1',width=0)),400,'OUTLINE_ZERO_USE_LEGACY'),
            ('untested-crop',dict(output_scope='head',appearance=dict(version='fbx-outline-v1',width=.65)),503,'OUTLINE_NOT_CSP_CERTIFIED'),
            ('source-hash',dict(source_artifact_sha256='f'*64),409,'SOURCE_FBX_SHA256_MISMATCH'),
            ('personal',dict(character_id='personal'),422,'OUTLINE_UNSUPPORTED')]
        failures=[]
        for name,changes,status,code in negative:
            response=request(case,changes=changes)
            assert response.status_code==status and response.json()['error']['code']==code
            failures.append(dict(name=name,status=status,code=code))
        assert request(case,authenticated=False).status_code==401
        report=dict(capability=capability,cases=results,negative=failures,
            authentication_rejected=True,scope='real local HTTP + fresh Blender export/verify; no BFF, native save or CSP UI claim')
        (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=len(results),negative=len(failures)+1,
            report=str((destination/'report.json').resolve()))))


if __name__=='__main__':main()
