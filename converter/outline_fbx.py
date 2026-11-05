"""Add explicit FBX objects without re-exporting any original body objects.

Blender's bundled parser/encoder is used only inside a bounded child process.
Original geometry, normals, materials, morphs, bind matrices and clusters are
preserved as exact typed FBX values. New skin clusters share original bones.
"""
import array
import copy
import hashlib
import struct

PREFIX = b'Standin_Outline_'
CULLING = (b'CullingOff', b'CullingOnCCW', b'CullingOnCW')


def one(node, name):
    found = [e for e in node.elems if e.id == name]
    if len(found) != 1:
        raise ValueError('ambiguous or absent FBX field: ' + repr(name))
    return found[0]


def fingerprint(node):
    h = hashlib.sha256()
    def visit(n):
        h.update(struct.pack('<I', len(n.id))); h.update(n.id)
        h.update(bytes(n.props_type))
        for p in n.props:
            if isinstance(p, array.array):
                data = p.typecode.encode() + p.tobytes()
            elif isinstance(p, bytes):
                data = p
            elif isinstance(p, float):
                data = struct.pack('>d', p)
            else:
                data = str(p).encode('ascii')
            h.update(struct.pack('<Q', len(data))); h.update(data)
        h.update(struct.pack('<I', len(n.elems)))
        for c in n.elems:
            visit(c)
    visit(node)
    return h.hexdigest()


def write(path, root, version):
    from io_scene_fbx import encode_bin
    methods = dict(B='add_bool', C='add_char', Z='add_int8', Y='add_int16',
        I='add_int32', L='add_int64', F='add_float32', D='add_float64',
        R='add_bytes', S='add_string', i='add_int32_array', l='add_int64_array',
        f='add_float32_array', d='add_float64_array', b='add_bool_array', c='add_byte_array')
    def convert(n):
        e = encode_bin.FBXElem(n.id)
        for t, p in zip(n.props_type, n.props):
            getattr(e, methods[chr(t)])(p)
        e.elems = [convert(c) for c in n.elems]
        return e
    with encode_bin.FBXElem.enable_multithreading_cm():
        encoded = convert(root)
    encode_bin.write(str(path), encoded, version)


def element(name, props=(), types=b'', children=()):
    from io_scene_fbx.parse_fbx import FBXElem
    return FBXElem(name, list(props), bytearray(types), list(children))


def black_material(uid):
    def prop(name, kind, values):
        return element(b'P', [name, kind, b'', b'A', *values],
            b'SSSS' + (b'D' * len(values)))
    return element(b'Material', [uid, PREFIX+b'Black\x00\x01Material', b''], b'LSS', [
        element(b'Version', [102], b'I'), element(b'ShadingModel', [b'lambert'], b'S'),
        element(b'MultiLayer', [0], b'I'), element(b'Properties70', children=[
            prop(b'DiffuseColor', b'Color', [0., 0., 0.]),
            prop(b'DiffuseFactor', b'Number', [1.]),
            prop(b'EmissiveColor', b'Color', [0., 0., 0.]),
            prop(b'TransparencyFactor', b'Number', [0.])])])


def augment(root, displacement, *, culling=b'CullingOnCCW'):
    """displacement maps source Geometry ID -> one local-space delta/vertex.

    Returns an exact allowlist of new object/connection fingerprints, which the
    fresh verifier independently reconstructs from the source and settings.
    """
    if culling not in CULLING:
        raise ValueError('unsupported culling expression')
    objects = one(root, b'Objects'); connections = one(root, b'Connections')
    original = list(objects.elems); links = list(connections.elems)
    by_id = {e.props[0]: e for e in original}
    if len(by_id) != len(original) or any(PREFIX in e.props[1] for e in original if len(e.props)>1 and isinstance(e.props[1], bytes)):
        raise ValueError('duplicate IDs or preexisting outline objects')
    next_id = max(by_id) + 1
    added = []; added_links = []
    def uid():
        nonlocal next_id
        result=next_id; next_id+=1
        if result >= 2**63: raise ValueError('FBX ID overflow')
        return result
    material = black_material(uid()); added.append(material)
    def link(child, parent):
        added_links.append(element(b'C', [b'OO', child, parent], b'SLL'))
    geometries = [e for e in original if e.id==b'Geometry' and e.props[2]==b'Mesh']
    if set(displacement) != {g.props[0] for g in geometries}:
        raise ValueError('outline geometry allowlist mismatch')
    for index, geom in enumerate(geometries):
        gid = geom.props[0]
        mids = [c.props[2] for c in links if c.props[:2]==[b'OO', gid]]
        if len(mids)!=1 or by_id[mids[0]].id!=b'Model':
            raise ValueError('expected one geometry instance')
        model = by_id[mids[0]]
        skin_ids = [c.props[1] for c in links if c.props[0]==b'OO' and c.props[2]==gid and by_id.get(c.props[1]) and by_id[c.props[1]].id==b'Deformer']
        if len(skin_ids)!=1 or by_id[skin_ids[0]].props[2]!=b'Skin':
            raise ValueError('expected one source skin')
        skin = by_id[skin_ids[0]]
        clusters = [by_id[c.props[1]] for c in links if c.props[0]==b'OO' and c.props[2]==skin.props[0]]
        if not clusters or any(c.id!=b'Deformer' or c.props[2]!=b'Cluster' for c in clusters):
            raise ValueError('unsupported source deformers')
        cloned = [copy.deepcopy(e) for e in (geom, model, skin, *clusters)]
        remap = {a.props[0]: uid() for a in cloned}
        for e in cloned:
            e.props[0] = remap[e.props[0]]
            e.props[1] = PREFIX+str(index).encode()+b'_'+e.props[1]
        hull, hm = cloned[:2]
        one(hm, b'Culling').props[0] = culling
        points = one(hull, b'Vertices').props[0]
        delta = displacement[gid]
        if len(points)!=len(delta)*3:
            raise ValueError('source vertex correspondence mismatch')
        for i, d in enumerate(delta):
            for axis in range(3): points[3*i+axis] += d[axis]
        # Reverse winding, retaining the first vertex so normal indices can be
        # reversed in exactly the same order. Drop unused UV/edge layers on hull.
        indices = one(hull, b'PolygonVertexIndex').props[0]
        faces=[]; face=[]; permutation=[]; start=0
        for i, v in enumerate(indices):
            face.append(-v-1 if v<0 else v)
            if v<0:
                order=[0, *range(len(face)-1,0,-1)]
                faces.extend(face[j] for j in order)
                faces[-1] = -faces[-1]-1
                permutation.extend(start+j for j in order)
                face=[]; start=i+1
        if face: raise ValueError('unterminated FBX polygon')
        indices[:] = array.array(indices.typecode, faces)
        normals = one(hull, b'LayerElementNormal')
        reference=one(normals,b'ReferenceInformationType').props[0]
        mapping=one(normals,b'MappingInformationType').props[0]
        if (mapping not in (b'ByPolygonVertex',b'ByVertice')
                or reference not in (b'IndexToDirect',b'Direct')):
            raise ValueError('unsupported source normal mapping')
        values=one(normals,b'Normals').props[0]
        if mapping==b'ByVertice':
            values[:] = array.array(values.typecode,(-v for v in values))
        elif reference==b'Direct':
            values[:] = array.array(values.typecode, (-values[3*i+a] for i in permutation for a in range(3)))
        else:
            values[:] = array.array(values.typecode, (-v for v in values))
            ni=one(normals,b'NormalsIndex').props[0]
            ni[:] = array.array(ni.typecode, (ni[i] for i in permutation))
        material_layer=one(hull,b'LayerElementMaterial')
        one(material_layer,b'MappingInformationType').props[0]=b'AllSame'
        one(material_layer,b'Materials').props[0]=array.array('i',[0])
        hull.elems[:] = [e for e in hull.elems if e.id in {
            b'Properties70',b'GeometryVersion',b'Vertices',b'PolygonVertexIndex',
            b'LayerElementNormal',b'LayerElementMaterial',b'Layer'}]
        for layer in [e for e in hull.elems if e.id==b'Layer']:
            layer.elems[:] = [e for e in layer.elems if e.id!=b'LayerElement' or
                one(e,b'Type').props[0] in (b'LayerElementNormal',b'LayerElementMaterial')]
        for c in links:
            if c.props[0]!=b'OO': continue
            child,parent=c.props[1:3]
            if child in remap and (parent in remap or child==model.props[0]):
                link(remap[child],remap.get(parent,parent))
            elif parent in remap and child in by_id and by_id[child].id==b'Model' and by_id[child].props[2]==b'LimbNode':
                link(child,remap[parent])
        link(material.props[0], hm.props[0])
        # Separate bind pose; never append/change a node inside the original.
        for pose in [e for e in original if e.id==b'Pose']:
            nodes=[copy.deepcopy(n) for n in pose.elems if n.id==b'PoseNode']
            if not any(one(n,b'Node').props[0]==model.props[0] for n in nodes): continue
            for n in nodes:
                n_id=one(n,b'Node').props[0]
                if n_id in remap: one(n,b'Node').props[0]=remap[n_id]
            added.append(element(b'Pose',[uid(),PREFIX+str(index).encode()+b'\x00\x01Pose',b'BindPose'],b'LSS',[
                element(b'Type',[b'BindPose'],b'S'),element(b'Version',[100],b'I'),
                element(b'NbPoseNodes',[len(nodes)],b'I'),*nodes]))
        added.extend(cloned)
    objects.elems.extend(added); connections.elems.extend(added_links)
    # Definitions must count all new objects; preserve templates exactly.
    definitions=one(root,b'Definitions')
    one(definitions,b'Count').props[0]+=len(added)
    for kind in {e.id for e in added}:
        d=[e for e in definitions.elems if e.id==b'ObjectType' and e.props==[kind]]
        if len(d)!=1: raise ValueError('unsupported FBX definition')
        one(d[0],b'Count').props[0]+=sum(e.id==kind for e in added)
    return {'objects': {str(e.props[0]):fingerprint(e) for e in added},
        'connections':[fingerprint(c) for c in added_links], 'culling':culling.decode(),
        'meshes':len(geometries)}
