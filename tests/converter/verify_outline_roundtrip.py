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
from converter.outline_worker import expected_tree,verify_tree,tree
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
    (destination/'report.json').write_text(json.dumps(dict(valid=True,rejected=rejected,
        explicit_outline_objects=len(manifest['objects']),csp_certified=False),indent=2))


if __name__=='__main__':main()
