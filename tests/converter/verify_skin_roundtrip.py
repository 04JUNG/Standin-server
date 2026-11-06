"""Synthetic binary skin integrity gate; no private models or CSP claims."""
from pathlib import Path
import array,base64,copy,hashlib,json,struct,sys,zlib
import bpy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from make_fixtures import build_character_fbx
from converter.bone_map import MIXAMO
from converter.outline_fbx import one,write,PREFIX as OUTLINE_PREFIX
from converter.outline_worker import tree,import_source
from converter.skin_fbx import PREFIX
from converter.skin_spec import weights_hash
from converter import skin_worker as worker


def main():
    out=Path(sys.argv[sys.argv.index('--')+1]);out.mkdir(parents=True,exist_ok=False)
    source=out/'source.fbx';build_character_fbx(str(out/'initial.fbx'),MIXAMO)
    body=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    while body.data.uv_layers:body.data.uv_layers.remove(body.data.uv_layers[0])
    mat=bpy.data.materials.new('UnchangedSourceMaterial');mat.use_nodes=True
    body.data.materials.clear();body.data.materials.append(mat)
    for poly in body.data.polygons:poly.material_index=0
    bpy.ops.export_scene.fbx(filepath=str(source),use_selection=False,object_types={'ARMATURE','MESH'},add_leaf_bones=True,bake_anim=False)
    arm,meshes=import_source(source);root,version=tree(source);g=next(e for e in one(root,b'Objects').elems if e.id==b'Geometry' and e.props[2]==b'Mesh')
    indices=one(g,b'PolygonVertexIndex').props[0]
    asset=dict(vertices=len(meshes[0].data.vertices),bones=len(arm.data.bones),weights_sha256=weights_hash(meshes[0]),polygon_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),uv={'sha256':'a'*64})
    uv=dict(uv=[[float(i%2),float((i//2)%2)] for i in range(len(indices))])
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',2,2,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xfa\xdf\xca'*2)*2))+chunk(b'IEND',b'')
    # Asset stub only, never a certification override or public/API acceptance.
    worker.checked_asset=lambda *args:(asset,uv,png)
    job=dict(character_id='synthetic-not-supported',character_sha256='c'*64,skin=dict(version='fbx-skin-v1',id='flesh'))
    expected,_,manifest,*_=worker.expected_tree(source,job);valid=out/'valid.fbx';write(valid,expected,version)
    worker.verify_tree(source,valid,job);errors,final=worker.verify_import(source,valid,asset,uv,png)
    actual,_=tree(valid);objects=one(actual,b'Objects').elems
    geom=next(e for e in objects if e.id==b'Geometry' and e.props[2]==b'Mesh')
    video=next(e for e in objects if e.id==b'Video');newmat=next(e for e in objects if e.id==b'Material' and e.props[1].startswith(PREFIX))
    cluster=next(e for e in objects if e.id==b'Deformer' and e.props[2]==b'Cluster')
    def obj(r,uid):return next(e for e in one(r,b'Objects').elems if e.props[0]==uid)
    cases={
        'base-position':lambda r:one(obj(r,geom.props[0]),b'Vertices').props[0].__setitem__(0,999.),
        'base-weight':lambda r:one(obj(r,cluster.props[0]),b'Weights').props[0].__setitem__(0,.1),
        'uv':lambda r:one(one(obj(r,geom.props[0]),b'LayerElementUV'),b'UV').props[0].__setitem__(0,.123),
        'embedded-bytes':lambda r:one(obj(r,video.props[0]),b'Content').props.__setitem__(0,png+b'corruption'),
        'external-image':lambda r:one(obj(r,video.props[0]),b'Filename').props.__setitem__(0,b'../../external.png'),
        'missing-image':lambda r:one(r,b'Objects').elems.remove(obj(r,video.props[0])),
        'material':lambda r:one(obj(r,newmat.props[0]),b'Properties70').elems.pop(),
        'connection':lambda r:one(r,b'Connections').elems.pop(),
        'extra-mesh':lambda r:one(r,b'Objects').elems.append(copy.deepcopy(geom)),
    }
    rejected=[]
    for name,mutate in cases.items():
        bad=copy.deepcopy(actual);mutate(bad);path=out/(name+'.fbx');write(path,bad,version)
        try:worker.verify_tree(source,path,job)
        except ValueError:rejected.append(name)
        else:raise AssertionError('accepted corrupted skin FBX: '+name)
    bad=copy.deepcopy(root);sg=next(e for e in one(bad,b'Objects').elems if e.id==b'Geometry' and e.props[2]==b'Mesh');one(sg,b'PolygonVertexIndex').props[0][0]+=1
    path=out/'incompatible-source.fbx';write(path,bad,version)
    try:worker.expected_tree(path,job)
    except ValueError:rejected.append('source-topology')
    else:raise AssertionError('guessed UV correspondence')
    bad=copy.deepcopy(root);one(bad,b'Objects').elems[0].props[1]=OUTLINE_PREFIX+b'preexisting'
    path=out/'wrong-order.fbx';write(path,bad,version)
    try:worker.expected_tree(path,job)
    except ValueError as exc:assert str(exc)=='SKIN_OUTLINE_ORDER';rejected.append('skin-after-outline')
    else:raise AssertionError('allowed wrong composition order')
    (out/'report.json').write_text(json.dumps(dict(valid=True,rejected=rejected,base=errors,embedded_sha256=hashlib.sha256(png).hexdigest(),synthetic_only=True,csp_certified=False),indent=2)+'\n')


if __name__=='__main__':main()
