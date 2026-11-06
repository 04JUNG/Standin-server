"""Blender real-binary adversarial check, no private model assets required.

blender --background --factory-startup --python-exit-code 1 \
  --python tests/converter/verify_outline_roundtrip.py -- OUT
"""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.outline_probe import small_source
from converter.outline_worker import expected_tree,verify_tree,tree,import_source
from converter.outline_fbx import one,write,PREFIX


def main():
    destination=Path(sys.argv[sys.argv.index('--')+1]);destination.mkdir(parents=True,exist_ok=False)
    source=destination/'source.fbx';small_source(source)
    expected,version,manifest=expected_tree(source,.002625,'CullingOnCCW')
    valid=destination/'valid.fbx';write(valid,expected,version)
    verify_tree(source,valid,.002625,'CullingOnCCW')
    actual,_=tree(valid);objects=one(actual,b'Objects').elems
    base_geometry=next(e for e in objects if e.id==b'Geometry' and not e.props[1].startswith(PREFIX))
    hull_geometry=next(e for e in objects if e.id==b'Geometry' and e.props[1].startswith(PREFIX))
    base_material=next(e for e in objects if e.id==b'Material' and not e.props[1].startswith(PREFIX))
    base_cluster=next(e for e in objects if e.id==b'Deformer' and e.props[2]==b'Cluster' and not e.props[1].startswith(PREFIX))
    tests={
        'base-position':lambda r:one(next(e for e in one(r,b'Objects').elems if e.props[0]==base_geometry.props[0]),b'Vertices').props[0].__setitem__(0,999.),
        'base-material':lambda r:next(e for e in one(r,b'Objects').elems if e.props[0]==base_material.props[0]).props.__setitem__(1,b'changed'),
        'base-weight':lambda r:one(next(e for e in one(r,b'Objects').elems if e.props[0]==base_cluster.props[0]),b'Weights').props[0].__setitem__(0,.1),
        'outline-position':lambda r:one(next(e for e in one(r,b'Objects').elems if e.props[0]==hull_geometry.props[0]),b'Vertices').props[0].__setitem__(0,999.),
        'extra-mesh':lambda r:one(r,b'Objects').elems.append(copy.deepcopy(hull_geometry)),
        'missing-outline':lambda r:one(r,b'Objects').elems.__delitem__(next(i for i,e in enumerate(one(r,b'Objects').elems) if e.props[0]==hull_geometry.props[0])),
        'connection':lambda r:one(r,b'Connections').elems.pop(),
    }
    rejected=[]
    for name,mutate in tests.items():
        bad=copy.deepcopy(actual);mutate(bad);path=destination/(name+'.fbx');write(path,bad,version)
        try:verify_tree(source,path,.002625,'CullingOnCCW')
        except ValueError:rejected.append(name)
        else:raise AssertionError('accepted corrupt FBX: '+name)
    for distance,culling in ((.001,'CullingOnCCW'),(.002625,'CullingOff')):
        try:verify_tree(source,valid,distance,culling)
        except ValueError:rejected.append(str((distance,culling)))
        else:raise AssertionError('accepted stale appearance')
    native,_,native_manifest=expected_tree(source,.002625,'CullingOff',surface='outward-transparent')
    native_path=destination/'native-outward-transparent.fbx';write(native_path,native,version)
    verify_tree(source,native_path,.002625,'CullingOff',surface='outward-transparent')
    _,imported_meshes=import_source(native_path)
    for obj in imported_meshes:
        if not obj.name.encode().startswith(PREFIX):continue
        assert len(obj.material_slots)==1
        mat=obj.material_slots[0].material
        shaders=[n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
        assert len(shaders)==1 and shaders[0].inputs['Alpha'].default_value==0.
    # Every original object stays exactly the same; transparency and outward
    # normals apply exclusively to the explicitly added guide, never the body.
    source_root,_=tree(source);native_root,_=tree(native_path)
    from converter.outline_fbx import fingerprint
    before={e.props[0]:fingerprint(e) for e in one(source_root,b'Objects').elems}
    after={e.props[0]:fingerprint(e) for e in one(native_root,b'Objects').elems if e.props[0] in before}
    assert before==after
    material=next(e for e in one(native_root,b'Objects').elems if e.id==b'Material' and e.props[1].startswith(PREFIX))
    props={e.props[0]:e for e in one(material,b'Properties70').elems}
    assert props[b'Opacity'].props[-1]==0. and props[b'TransparencyFactor'].props[-1]==1.
    props[b'Opacity'].props[-1]=1.
    corrupted=destination/'native-opaque-corruption.fbx';write(corrupted,native_root,version)
    try:verify_tree(source,corrupted,.002625,'CullingOff',surface='outward-transparent')
    except ValueError:rejected.append('native-opacity')
    else:raise AssertionError('accepted native guide with opaque material')
    (destination/'report.json').write_text(json.dumps(dict(valid=True,rejected=rejected,
        explicit_outline_objects=len(manifest['objects']),native_explicit_objects=len(native_manifest['objects']),
        native_body_exact=True,csp_certified=False),indent=2))


if __name__=='__main__':main()
