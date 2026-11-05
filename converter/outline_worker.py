"""Isolated candidate writer / independent verifier. Never certifies CSP."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from converter.outline_spec import (normalize, appearance_hash, physical_thickness, runtime_revision,
    certified, expression_for, LEGACY_SURFACE, NATIVE_SURFACE)
from converter.outline_fbx import one, augment, fingerprint, write, PREFIX


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_source(path):
    from converter.retarget import reset_scene
    from converter.shape_bake import import_full_character
    reset_scene()
    return import_full_character(str(path))


def offsets(root, meshes, distance):
    """World-metre expansion transformed back into each FBX geometry space."""
    from mathutils import Vector
    import numpy as np
    result={}
    geometry=[g for g in one(root,b'Objects').elems if g.id==b'Geometry' and g.props[2]==b'Mesh']
    links=one(root,b'Connections').elems
    objects={g.props[0]:g for g in one(root,b'Objects').elems}
    for g in geometry:
        model_ids=[c.props[2] for c in links if c.props[:2]==[b'OO',g.props[0]]]
        if len(model_ids)!=1: raise ValueError('geometry is instanced')
        name=objects[model_ids[0]].props[1].split(b'\x00\x01')[0].decode()
        matches=[m for m in meshes if m.name==name]
        if len(matches)!=1: raise ValueError('source mesh identity mismatch')
        obj=matches[0]
        p=np.asarray(one(g,b'Vertices').props[0],dtype=np.float64).reshape(-1,3)
        if len(p)!=len(obj.data.vertices): raise ValueError('source vertex count mismatch')
        imported=np.array([v.co[:] for v in obj.data.vertices])
        if not np.allclose(p,imported,atol=1e-7,rtol=1e-7):
            raise ValueError('unsupported FBX geometric transform')
        # Use the source's smooth geometric vertex normal, not an importer
        # material flag or sharp split-normal which would tear adjacent faces.
        normals=np.array([v.normal[:] for v in obj.data.vertices],dtype=np.float64)
        matrix=obj.matrix_world.to_3x3()
        if abs(matrix.determinant())<1e-10: raise ValueError('singular source mesh')
        inv=matrix.inverted(); normal_matrix=inv.transposed()
        delta=[]
        for n in normals:
            world=(normal_matrix @ Vector(n)).normalized()
            if world.length<.99: raise ValueError('degenerate source normal')
            local=inv @ (world*distance)
            if not all(math.isfinite(v) for v in local): raise ValueError('nonfinite outline')
            delta.append(tuple(local))
        result[g.props[0]]=delta
    return result


def tree(path):
    from io_scene_fbx import parse_fbx
    return parse_fbx.parse(str(path))


def expected_tree(source, distance, culling, *, surface=LEGACY_SURFACE):
    root,version=tree(source)
    if any(e.id.startswith(b'Animation') for e in one(root,b'Objects').elems):
        raise ValueError('animated source FBX is outside the rigged-rest contract')
    arm,meshes=import_source(source)
    displacement=offsets(root,meshes,distance)
    manifest=augment(root,displacement,culling=culling.encode(),surface=surface)
    return root,version,manifest


def verify_tree(source, output, distance, culling, *, surface=LEGACY_SURFACE):
    expected,version,manifest=expected_tree(source,distance,culling,surface=surface)
    actual,actual_version=tree(output)
    if version!=actual_version: raise ValueError('FBX version changed')
    # The encoder normalizes header timestamps/FileId. Everything affecting the
    # scene, including unrelated objects/UV/morph/materials, must match exactly.
    ignored={b'FBXHeaderExtension',b'FileId',b'CreationTime',b'Creator'}
    exp=[fingerprint(e) for e in expected.elems if e.id not in ignored]
    got=[fingerprint(e) for e in actual.elems if e.id not in ignored]
    if exp!=got: raise ValueError('base or explicit outline FBX tree changed')
    return manifest


def run(stage, directory):
    import bpy
    from converter.interactive_scene import capture, compare
    directory=Path(directory); job=json.loads((directory/'job.json').read_bytes())
    source=directory/'source.fbx'; output=directory/'artifact.fbx'
    if sha(source)!=job['source_artifact_sha256']: raise ValueError('source checksum mismatch')
    appearance=normalize(job['appearance'])
    surface=job.get('surface',LEGACY_SURFACE)
    expression,mapping=expression_for(surface)
    if not appearance['width']: raise ValueError('zero must retain original output path')
    distance=physical_thickness(appearance,job['stature_m'],job['output_scope'],surface=surface)
    start=time.monotonic()
    if stage=='export':
        if output.exists(): raise ValueError('output already exists')
        root,version,manifest=expected_tree(source,distance,job['culling'],surface=surface)
        write(output,root,version)
        (directory/'candidate.json').write_text(json.dumps(manifest,sort_keys=True))
    else:
        output_sha=sha(output)
        manifest=verify_tree(source,output,distance,job['culling'],surface=surface)
        # Reimport source and output separately. Delete only manifest-authored
        # hulls AFTER exact tree validation, then reuse the strict existing body
        # comparator unchanged. Arbitrary extra meshes never reach this step.
        arm,meshes=import_source(source)
        base=capture(); origin=base['coordinates']['originBlender']
        source_names={m.name for m in meshes}
        source_points={m.name:[m.matrix_world @ v.co for v in m.data.vertices] for m in meshes}
        from converter.final_artifact import validate
        from converter import retarget as rt
        from converter.shape_bake import import_full_character
        rt.import_character=import_full_character
        skin=validate(output,job['output_scope'],len(base['bones']))
        hulls=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.encode().startswith(PREFIX)]
        if len(hulls)!=manifest['meshes']: raise ValueError('outline mesh count changed on import')
        imported_culling=[all(s.material and s.material.use_backface_culling for s in h.material_slots) for h in hulls]
        displacement_error=0.
        for h in hulls:
            matches=[name for name in source_names if h.name.endswith('_'+name)]
            if len(matches)!=1: raise ValueError('outline source mesh correspondence changed')
            points=source_points[matches[0]]
            if len(points)!=len(h.data.vertices): raise ValueError('outline vertex correspondence changed')
            for point,vertex in zip(points,h.data.vertices):
                measured=(h.matrix_world @ vertex.co-point).length
                error=abs(measured-distance)
                if not math.isfinite(error) or error>1e-6:
                    raise ValueError('outline physical displacement changed on import')
                displacement_error=max(displacement_error,error)
            if len(h.material_slots)!=1: raise ValueError('outline material count changed')
            mat=h.material_slots[0].material
            alpha=0. if surface==NATIVE_SURFACE else 1.
            # Blender's FBX importer writes transparency to the Principled
            # socket, while the separate viewport diffuse alpha can stay 1.
            # Verify the actual imported shader, not that viewport cache.
            nodes=[] if mat is None or mat.node_tree is None else [
                n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
            if (len(nodes)!=1 or nodes[0].inputs['Alpha'].is_linked
                    or nodes[0].inputs['Base Color'].is_linked
                    or any(abs(v)>1e-6 for v in nodes[0].inputs['Base Color'].default_value[:3])
                    or abs(nodes[0].inputs['Alpha'].default_value-alpha)>1e-6):
                raise ValueError('outline material differs from its expression')
            bpy.data.objects.remove(h,do_unlink=True)
        if {o.name for o in bpy.context.scene.objects if o.type=='MESH'}!=source_names:
            raise ValueError('base mesh names changed')
        errors=compare(base,capture(origin))
        if sha(output)!=output_sha or sha(source)!=job['source_artifact_sha256']:
            raise ValueError('verification changed artifact')
        receipt=dict(version='outlined-fbx-receipt-v1',expression=expression,mapping=mapping,
            runtime_revision=runtime_revision(),blender_version=bpy.app.version_string,
            source_artifact_sha256=job['source_artifact_sha256'],fbx_sha256=output_sha,
            appearance=appearance,appearance_sha256=appearance_hash(appearance),
            binding_sha256=job['binding_sha256'],character_id=job['character_id'],
            character_sha256=job['character_sha256'],output_scope=job['output_scope'],
            stature_m=job['stature_m'],thickness_m=distance,culling=job['culling'],
            base=dict(typed_fbx_values_preserved=True,**errors),
            outline=dict(manifest=manifest,skin_verified=skin['skin_verified'],
                displacement_verified=True,max_displacement_error_m=displacement_error,
                blender_import_backface_culling=imported_culling),
            file_verified=True,csp_certified=certified(job['character_sha256'],job['output_scope'],surface=surface),
            verification_seconds=time.monotonic()-start)
        (directory/'verified.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    run(args[0],args[1])
