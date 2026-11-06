"""Typed FBX material/UV/embedded-Video additions, no body Blender re-export."""
import array
import hashlib
from converter.outline_fbx import one,element,fingerprint,PREFIX as OUTLINE_PREFIX

PREFIX=b'Standin_SkinV1_'
UV_NAME=b'StandinSkinUV_v1'


def _property(name,kind,values,types=None):
    return element(b'P',[name,kind,b'',b'A',*values],b'SSSS'+(types or b'D'*len(values)))


def augment(root,asset,uv,png,skin_id):
    objects=one(root,b'Objects');connections=one(root,b'Connections');original=list(objects.elems);by_id={e.props[0]:e for e in original}
    if len(by_id)!=len(original):raise ValueError('duplicate FBX object IDs')
    if any(len(e.props)>1 and isinstance(e.props[1],bytes) and (e.props[1].startswith(OUTLINE_PREFIX) or e.props[1].startswith(PREFIX)) for e in original):
        raise ValueError('SKIN_OUTLINE_ORDER')
    geometries=[e for e in original if e.id==b'Geometry' and e.props[2]==b'Mesh']
    if len(geometries)!=1:raise ValueError('SKIN_UV_INCOMPATIBLE')
    geom=geometries[0];indices=one(geom,b'PolygonVertexIndex').props[0]
    if (len(one(geom,b'Vertices').props[0])//3!=asset['vertices']
            or hashlib.sha256(indices.tobytes()).hexdigest()!=asset['polygon_indices_sha256']
            or len(uv['uv'])!=len(indices) or any(e.id==b'LayerElementUV' for e in geom.elems)):
        raise ValueError('SKIN_UV_INCOMPATIBLE')
    mids=[c.props[2] for c in connections.elems if c.props[:2]==[b'OO',geom.props[0]]]
    if len(mids)!=1:raise ValueError('instanced geometry is not supported')
    links=[c for c in connections.elems if c.props[0]==b'OO' and c.props[2]==mids[0] and c.props[1] in by_id and by_id[c.props[1]].id==b'Material']
    if len(links)!=1:raise ValueError('unsupported source material slots')
    # Existing LayerElementMaterial remains untouched: it must address slot 0.
    material_layer=one(geom,b'LayerElementMaterial')
    if any(value!=0 for value in one(material_layer,b'Materials').props[0]):raise ValueError('unsupported material indices')
    first=max(by_id)+1;matid,texid,vidid=first,first+1,first+2
    if vidid>=2**63:raise ValueError('FBX ID overflow')
    filename=PREFIX+skin_id.encode()+b'_'+hashlib.sha256(png).hexdigest()[:16].encode()+b'.png';name=PREFIX+skin_id.encode()
    material=element(b'Material',[matid,name+b'\x00\x01Material',b''],b'LSS',[
        element(b'Version',[102],b'I'),element(b'ShadingModel',[b'Phong'],b'S'),element(b'MultiLayer',[0],b'I'),
        element(b'Properties70',children=[_property(b'DiffuseColor',b'Color',[1.,1.,1.]),_property(b'DiffuseFactor',b'Number',[1.]),
            _property(b'EmissiveColor',b'Color',[0.,0.,0.]),_property(b'EmissiveFactor',b'Number',[0.]),
            _property(b'SpecularFactor',b'Number',[0.]),_property(b'TransparencyFactor',b'Number',[0.]),_property(b'Opacity',b'Number',[1.])])])
    texture=element(b'Texture',[texid,name+b'\x00\x01Texture',b''],b'LSS',[
        element(b'Type',[b'TextureVideoClip'],b'S'),element(b'Version',[202],b'I'),
        element(b'TextureName',[name+b'\x00\x01Texture'],b'S'),element(b'Media',[name+b'\x00\x01Video'],b'S'),
        element(b'FileName',[filename],b'S'),element(b'RelativeFilename',[filename],b'S'),
        element(b'Properties70',children=[_property(b'UVSet',b'KString',[UV_NAME],b'S'),_property(b'UseMaterial',b'bool',[1],b'I'),
            _property(b'AlphaSource',b'enum',[0],b'I'),_property(b'CurrentMappingType',b'enum',[0],b'I')])])
    video=element(b'Video',[vidid,name+b'\x00\x01Video',b'Clip'],b'LSS',[
        element(b'Type',[b'Clip'],b'S'),element(b'UseMipMap',[0],b'I'),element(b'Filename',[filename],b'S'),
        element(b'RelativeFilename',[filename],b'S'),element(b'Content',[png],b'R')])
    uvnode=element(b'LayerElementUV',[0],b'I',[
        element(b'Version',[101],b'I'),element(b'Name',[UV_NAME],b'S'),
        element(b'MappingInformationType',[b'ByPolygonVertex'],b'S'),element(b'ReferenceInformationType',[b'IndexToDirect'],b'S'),
        element(b'UV',[array.array('d',(v for pair in uv['uv'] for v in pair))],b'd'),
        element(b'UVIndex',[array.array('i',range(len(uv['uv'])))],b'i')])
    geom.elems.append(uvnode)
    layer=next((e for e in geom.elems if e.id==b'Layer' and e.props==[0]),None)
    if layer is None:raise ValueError('missing source Layer0')
    uvlink=element(b'LayerElement',children=[element(b'Type',[b'LayerElementUV'],b'S'),element(b'TypedIndex',[0],b'I')]);layer.elems.append(uvlink)
    connections.elems.remove(links[0]);added_links=[element(b'C',[b'OO',matid,mids[0]],b'SLL'),
        element(b'C',[b'OP',texid,matid,b'DiffuseColor'],b'SLLS'),element(b'C',[b'OO',vidid,texid],b'SLL')]
    added=[material,texture,video];objects.elems.extend(added);connections.elems.extend(added_links)
    definitions=one(root,b'Definitions');one(definitions,b'Count').props[0]+=len(added)
    for kind in (b'Material',b'Texture',b'Video'):
        matches=[e for e in definitions.elems if e.id==b'ObjectType' and e.props==[kind]]
        if matches:one(matches[0],b'Count').props[0]+=1
        else:definitions.elems.append(element(b'ObjectType',[kind],b'S',[element(b'Count',[1],b'I')]))
    return dict(objects={str(e.props[0]):fingerprint(e) for e in added},connections=[fingerprint(c) for c in added_links],
        removed_connections=[fingerprint(links[0])],geometry=str(geom.props[0]),uv=fingerprint(uvnode),uv_layer_link=fingerprint(uvlink),
        texture_sha256=hashlib.sha256(png).hexdigest(),embedded=True,meshes=1)
