"""Exact source-graph body-control evaluation before reduced-mesh deformation.

The retained original numeric graph is NOT a second Blender mesh. The actual
neutral, pose, PSD evaluation, skin, solver FBX and final validation all use the
reduced mesh. No post-export simplification is involved. Pure body-control math
still costs the original graph; benchmark this cost rather than hiding it.
"""
import base64
import io
import json
import types
import numpy as np


def payload(source, properties):
    arrays = {k: source[k] for k in ('co', 'faces', 'W', 'edges_raw')}
    if 'mass_W' in source:
        arrays['mass_W'] = source['mass_W']
    for category in ('deltas', 'fat_smooth'):
        for name, values in source.get(category, {}).items():
            arrays[category + '::' + name] = values
    arrays['metadata'] = np.array(json.dumps({
        'names': source['names'], 'rest': source['rest'],
        'properties': properties,
    }, sort_keys=True))
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def controller_tail(original_code):
    # Embed only the pinned original evaluator code, never load a user path.
    return '''
# Exact approved control graph, then immutable source-vertex correspondence.
_LOD_REFERENCE_CODE = %r
_LOD_REFERENCE_CACHE = {}
def _lod_reference(scene=None):
    import io, types
    m, a = objects(scene)
    key = m.as_pointer()
    if key not in _LOD_REFERENCE_CACHE:
        raw = base64.b64decode(m['lod_reference_graph'])
        if hashlib.sha256(raw).hexdigest() != m['lod_reference_graph_sha256']:
            raise ValueError('reference graph checksum changed')
        archive = np.load(io.BytesIO(raw), allow_pickle=False)
        metadata = json.loads(str(archive['metadata']))
        source_data = {k: archive[k].copy() for k in ('co','faces','W','edges_raw')}
        if 'mass_W' in archive: source_data['mass_W'] = archive['mass_W'].copy()
        source_data.update(names=metadata['names'],rest=metadata['rest'],deltas={})
        for name in archive.files:
            if name.startswith('deltas::'):
                source_data['deltas'][name.split('::',1)[1]] = archive[name].copy()
            elif name.startswith('fat_smooth::'):
                source_data.setdefault('fat_smooth',{})[name.split('::',1)[1]] = archive[name].copy()
        reference = types.ModuleType('_standin_exact_reference')
        exec(compile(_LOD_REFERENCE_CODE,'<pinned-reference-controller>','exec'),reference.__dict__)
        reference.source = lambda scene=None: source_data
        reference.objects = lambda scene=None: (metadata['properties'], a)
        if 'lod_source_projection' in m:
            raw_projection = base64.b64decode(m['lod_source_projection'])
            if hashlib.sha256(raw_projection).hexdigest() != m['lod_source_projection_sha256']:
                raise ValueError('source projection checksum changed')
            projection = np.load(io.BytesIO(raw_projection),allow_pickle=False)
            ids, bary = projection['indices'].copy(), projection['weights'].copy()
            offsets = projection['offsets'].copy() if 'offsets' in projection else np.zeros((len(ids),3))
        else:
            ids = np.frombuffer(base64.b64decode(m['lod_source_indices']),dtype='<i4').copy()[:,None]
            bary = np.ones(ids.shape)
            offsets = np.zeros((len(ids),3))
        _LOD_REFERENCE_CACHE[key] = (reference, ids, bary, offsets)
    return _LOD_REFERENCE_CACHE[key]

def compute(c, scene=None):
    reference, ids, bary, offsets = _lod_reference(scene)
    result, weights, neutral = reference.compute(c, scene)
    def embed(values):
        tri=values[ids];out=np.einsum('ni,nij->nj',bary,tri)
        if offsets.any():
            u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];cross=np.cross(u,v)
            normal=cross/np.sqrt(np.maximum(np.linalg.norm(cross,axis=1),1e-30))[:,None]
            out+=u*offsets[:,0,None]+v*offsets[:,1,None]+normal*offsets[:,2,None]
        return out
    return embed(result), weights, embed(neutral)
''' % original_code
