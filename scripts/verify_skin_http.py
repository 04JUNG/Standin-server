"""Real local skin -> optional outline HTTP check; private fixture, no paid calls."""
import argparse,base64,hashlib,json,time
from pathlib import Path
import httpx


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args();f=json.loads(a.fixture.read_text());out=Path(f['output']);out.mkdir(parents=True,exist_ok=True);headers={'Authorization':'Bearer '+Path(f['token_file']).read_text().strip()};rows=[];negative=[]
    with httpx.Client(base_url=f['url'],timeout=360,trust_env=False) as client:
        capability=client.get('/skin-contract').raise_for_status().json();outline=client.get('/outline-contract').raise_for_status().json()
        assert capability['enabled'] and capability['assets_available'] and capability['csp_certified']
        def post(case,*,changes=None,data=None,auth=True):
            raw=Path(case['source']).read_bytes() if data is None else data;fields={**case['fields'],**(changes or {})};fields['skin']=json.dumps(fields['skin'],sort_keys=True)
            return client.post('/skin-fbx',headers=headers if auth else {},data=fields,files={'fbx':('source.fbx',raw,'application/octet-stream')})
        for case in f['cases']:
            source=Path(case['source']).read_bytes();assert sha(source)==case['fields']['source_artifact_sha256'];start=time.monotonic();response=post(case);response.raise_for_status();body=response.json();r=body['receipt'];data=base64.b64decode(body['fbx_base64'],validate=True)
            assert body['artifact_contract']=='textured-fbx-v1' and sha(data)==body['fbx_sha256']==r['fbx_sha256']
            if case.get('expected_sha256'):assert sha(data)==case['expected_sha256']
            for key,value in case['fields'].items():assert r[key]==value,(case['name'],key)
            for key in ('expression','runtime_revision','asset_manifest_sha256','asset_version'):assert r[key]==capability[key]
            assert r['skin_sha256']==sha(json.dumps(r['skin'],sort_keys=True,separators=(',',':')).encode())
            uv=next(x for x in capability['assets'] if x['kind']=='uv' and x['character_id']==r['character_id'])
            texture=next(x for x in capability['assets'] if x['kind']=='texture' and x['character_id']==r['character_id'] and x['skin_id']==r['skin']['id'])
            assert r['uv_sha256']==uv['sha256'] and r['texture_sha256']==texture['sha256']
            assert r['file_verified'] and r['csp_certified'] and r['base']['typed_fbx_values_preserved'] and r['base']['approved_surface_changes_only']
            assert all(value==0 for key,value in r['base'].items() if key.startswith('max'))
            assert all(r['material'][key] for key in ('embedded','uv_verified','image_verified'))
            path=out/(case['name']+'.fbx');path.write_bytes(data);(out/(case['name']+'.receipt.json')).write_text(json.dumps(r,indent=2)+'\n');skin_seconds=time.monotonic()-start
            row=dict(name=case['name'],skin_sha256=sha(data),skin_bytes=len(data),skin_seconds=skin_seconds,skin_path=str(path),receipt=r)
            if case.get('outline_width'):
                fields=dict(source_artifact_sha256=sha(data),appearance=json.dumps(dict(version='fbx-outline-v1',width=case['outline_width'])),binding_sha256=case['fields']['binding_sha256'],character_id=case['fields']['character_id'],character_sha256=case['fields']['character_sha256'],output_scope='full',stature_m=case.get('stature_m',1.75));start=time.monotonic()
                result=client.post('/outline-fbx',headers=headers,data=fields,files={'fbx':('skin.fbx',data,'application/octet-stream')}).raise_for_status().json();final=base64.b64decode(result['fbx_base64'],validate=True);proof=result['receipt']
                assert sha(final)==result['fbx_sha256']==proof['fbx_sha256'] and proof['source_artifact_sha256']==r['fbx_sha256']
                assert proof['file_verified'] and proof['csp_certified'] and proof['runtime_revision']==outline['runtime_revision']
                assert all(value==0 for key,value in proof['base'].items() if key.startswith('max'))
                assert proof['outline']['manifest']['surface']=='outward-transparent'
                (out/(case['name']+'-outline.fbx')).write_bytes(final);(out/(case['name']+'-outline.receipt.json')).write_text(json.dumps(proof,indent=2)+'\n')
                row.update(combined_sha256=sha(final),combined_bytes=len(final),outline_seconds=time.monotonic()-start,outline_receipt=proof)
            rows.append(row);(out/'report.json').write_text(json.dumps(dict(capability=capability,cases=rows),indent=2)+'\n');print(case['name'],'PASS',flush=True)
        case=f['cases'][0]
        for name,changes,status,code in [('default',dict(skin=dict(version='fbx-skin-v1',id='default')),400,'SKIN_DEFAULT_USE_LEGACY'),('crop',dict(output_scope='head'),422,'SKIN_UNSUPPORTED'),('source-sha',dict(source_artifact_sha256='f'*64),409,'SOURCE_FBX_SHA256_MISMATCH'),('personal',dict(character_id='personal'),422,'SKIN_UNSUPPORTED')]:
            response=post(case,changes=changes);assert response.status_code==status and response.json()['error']['code']==code;negative.append(dict(name=name,status=status,code=code))
        assert post(case,auth=False).status_code==401;negative.append(dict(name='auth',status=401))
        for asset in capability['assets']:
            response=client.get(asset['url'],headers=headers).raise_for_status();assert sha(response.content)==asset['sha256'] and len(response.content)==asset['bytes']
        (out/'report.json').write_text(json.dumps(dict(capability=capability,cases=rows,negative=negative,assets_verified=len(capability['assets']),scope='Real local TCP skin/outline + fresh Blender; no BFF/native/CSP UI claim'),indent=2)+'\n')
        print(json.dumps(dict(positive=len(rows),combined=sum('combined_sha256' in r for r in rows),negative=len(negative),assets=len(capability['assets']),report=str(out/'report.json'))))


if __name__=='__main__':main()
