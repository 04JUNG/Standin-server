"""Offline private asset publication from an independently verified state/FBX.

No card request calls this. Assets remain outside git; manifest publishes last.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from converter.solved_state import SolvedPoseState,canonical,digest
from converter.preview_model import encode_surface,MODEL_VERSION
from converter.interactive_adapter import atomic_write
from converter_api.interactive_static import key


def publish(state,receipt,destination,*,registered_character_sha256):
    import numpy as np
    payload=state.payload;i=payload['identity'];model=payload['model']
    if (i['body']['characterSha256'] not in registered_character_sha256 or i['source']['refinedBvhSha256'] is not None
        or receipt.get('verified') is not True or receipt.get('stateHash')!=state.state_hash or receipt.get('identity')!=i
        or receipt.get('packetModelSha256')!=digest(canonical(model))):raise ValueError('asset lacks approved exact file proof')
    world=[];transforms=[]
    for b in model['bones']:
        m=np.array(b['poseLocal']).reshape(4,4)
        world.append(m if b['parent']<0 else world[b['parent']]@m)
        transforms.append(world[-1]@np.array(b['inverseBind']).reshape(4,4))
    normal_transforms=[np.linalg.inv(t[:3,:3]).T for t in transforms]
    positions=[];normals=[];indices=[]
    for mesh in model['meshes']:
        skinned=[]
        for p,weights in zip(mesh['positions'],mesh['weights']):
            skinned.append(sum((transforms[b]@[*p,1])[:3]*w for b,w in weights))
        for tri,nt in zip(mesh['triangles'],mesh['normalTriangles']):
            for vertex,n in zip(tri,nt):
                positions.extend(skinned[vertex]);v=np.array(mesh['normals'][n])
                transformed=sum(normal_transforms[b]@v*w for b,w in mesh['weights'][vertex]);transformed/=np.linalg.norm(transformed)
                normals.extend(transformed);indices.append(len(indices))
    identity=dict(sourceSha256=i['source']['originalBvhSha256'],characterId=i['body']['characterId'],characterSha256=i['body']['characterSha256'],scope=i['scope'])
    revision='file-canonical-static-v1'
    metadata=dict(version=MODEL_VERSION,source_bvh_sha256=identity['sourceSha256'],character_id=identity['characterId'],
        character_sha256=identity['characterSha256'],coordinates='Y-up-hips-origin',scope=identity['scope'],revision=revision)
    raw=encode_surface(positions,normals,indices,metadata);sha=digest(raw)
    out=Path(destination)/'interactive-static-v1';out.mkdir(parents=True,exist_ok=True)
    asset=out/(sha+'.glb')
    if not asset.exists():atomic_write(asset,raw)
    elif asset.read_bytes()!=raw:raise ValueError('asset path changed')
    manifest=dict(**identity,version=MODEL_VERSION,coordinates='Y-up-hips-origin',revision=revision,sha256=sha,bytes=len(raw),
        bones=model['bones'],stateHash=state.state_hash,artifactSha256=receipt['artifactSha256'])
    atomic_write(out/(key(identity['sourceSha256'],identity['characterSha256'],identity['scope'])+'.json'),json.dumps(manifest,separators=(',',':')).encode())
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--canonical-state',required=True);p.add_argument('--receipt',required=True);p.add_argument('--destination',required=True);p.add_argument('--approved-character-sha256',required=True);a=p.parse_args()
    result=publish(SolvedPoseState(Path(a.canonical_state).read_bytes()),json.loads(Path(a.receipt).read_bytes()),a.destination,registered_character_sha256={a.approved_character_sha256})
    print(json.dumps({k:result[k] for k in ('sourceSha256','sha256','bytes','stateHash')}))
