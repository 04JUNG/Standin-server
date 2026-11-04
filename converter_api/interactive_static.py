"""Read-only prebuilt, FBX-verified static assets. Never invokes Blender."""
import json
from converter.preview_model import MODEL_VERSION, MAX_MODEL_BYTES
from converter.solved_state import canonical,digest
from converter.interactive_camera import camera_for_model
from converter_api.preview_models import PreviewModelStore


def key(source,character,scope):
    if scope not in ('full','half','bust','head','shoulder_head'):raise ValueError('invalid scope')
    from converter.solved_state import sha
    sha(source,'source');sha(character,'character')
    return digest(canonical([source,character,scope]))


class StaticStore(PreviewModelStore):
    def resolve(self,requests):
        if not isinstance(requests,list) or len(requests)>50:raise ValueError('static batch exceeds bound')
        result=[]
        for request in requests:
            identity=request['identity'];k=key(identity['sourceSha256'],identity['characterSha256'],identity['scope'])
            try:
                manifest=json.loads(self._read('interactive-static-v1/'+k+'.json',256*1024))
                expected={**identity,'version':MODEL_VERSION,'coordinates':'Y-up-hips-origin'}
                if any(manifest.get(f)!=v for f,v in expected.items()):raise ValueError('static binding changed')
                camera=camera_for_model(request['camera'],dict(bones=manifest['bones']),identity['characterSha256'],identity['scope'])
                data=self.file(manifest['sha256'])
                if len(data)!=manifest['bytes']:raise ValueError('static size changed')
                row=dict(schemaVersion='interactive-static-asset-v1',sha256=manifest['sha256'],bytes=len(data),
                    sourceSha256=identity['sourceSha256'],characterId=identity['characterId'],characterSha256=identity['characterSha256'],
                    scope=identity['scope'],revision=manifest['revision'],modelVersion=MODEL_VERSION,
                    camera=dict(schemaVersion='interactive-camera-v1',projection=camera['projection'],
                        viewMatrix=[camera['viewMatrix'][r*4+c] for c in range(4) for r in range(4)],
                        projectionMatrix=[camera['projectionMatrix'][r*4+c] for c in range(4) for r in range(4)],
                        matrixLayout='column-major',basis='right-handed-Y-up',clipSpace='webgl-minus-one-to-one',origin='hips',units='meters',
                        distanceTorso=camera['distanceTorso'],torsoReferenceSha256=camera['torsoReferenceSha256'],target=[0.,0.,0.],referenceAspect=1.,
                        framing='projection-includes-center-scale-crop',baked='none'))
                result.append(row)
            except FileNotFoundError:result.append(None)
        return result

    def file(self,sha):
        from converter.solved_state import sha as check_sha
        check_sha(sha,'asset')
        with self._lock:
            if sha in self._cache:
                self._cache.move_to_end(sha);return self._cache[sha]
            data=self._read('interactive-static-v1/'+sha+'.glb',MAX_MODEL_BYTES)
            if digest(data)!=sha or data[:4]!=b'glTF':raise ValueError('static asset integrity changed')
            while self._cache and self._size+len(data)>self.max_bytes:
                _,old=self._cache.popitem(last=False);self._size-=len(old)
            if len(data)<=self.max_bytes:self._cache[sha]=data;self._size+=len(data)
            return data
