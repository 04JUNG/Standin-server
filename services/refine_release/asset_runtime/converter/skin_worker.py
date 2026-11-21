"""Fresh typed writer and independent file/UV/material/body verifier, Blender-only."""
from pathlib import Path
import hashlib,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from converter.outline_worker import tree,import_source,sha
from converter.outline_fbx import one,write,fingerprint
from converter.skin_fbx import augment,UV_NAME,PREFIX
from converter.skin_spec import normalize,skin_hash,checked_asset,manifest_hash,runtime_revision,weights_hash,certified,EXPRESSION,manifest


def expected_tree(source,job):
    asset,uv,png=checked_asset(job['character_id'],job['character_sha256'],job['skin'])
    root,version=tree(source)
    if any(len(e.props)>1 and isinstance(e.props[1],bytes) and (e.props[1].startswith(PREFIX) or e.props[1].startswith(b'Standin_Outline_')) for e in one(root,b'Objects').elems):raise ValueError('SKIN_OUTLINE_ORDER')
    arm,meshes=import_source(source)
    if len(meshes)!=1 or len(arm.data.bones)!=asset['bones'] or weights_hash(meshes[0])!=asset['weights_sha256']:
        raise ValueError('SKIN_UV_INCOMPATIBLE')
    root,version=tree(source)
    if any(e.id.startswith(b'Animation') for e in one(root,b'Objects').elems):raise ValueError('animated source unsupported')
    approved=augment(root,asset,uv,png,normalize(job['skin'])['id'])
    return root,version,approved,asset,uv,png


def verify_tree(source,output,job):
    expected,version,approved,asset,uv,png=expected_tree(source,job)
    actual,got_version=tree(output);ignored={b'FBXHeaderExtension',b'FileId',b'CreationTime',b'Creator'}
    if version!=got_version or [fingerprint(e) for e in expected.elems if e.id not in ignored]!=[fingerprint(e) for e in actual.elems if e.id not in ignored]:
        raise ValueError('SKIN_INTEGRITY_ERROR')
    return approved,asset,uv,png


def verify_import(source,output,asset,uv,png):
    import bpy
    from converter.interactive_scene import capture,compare
    from converter.final_artifact import validate
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    arm,meshes=import_source(source);base=capture();origin=base['coordinates']['originBlender']
    source_names={o.name for o in meshes}
    rt.import_character=import_full_character
    final=validate(output,'full',asset['bones'])
    actual=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if len(actual)!=1 or {o.name for o in actual}!=source_names:raise ValueError('skin changed mesh identity')
    obj=actual[0]
    if weights_hash(obj)!=asset['weights_sha256']:raise ValueError('skin changed neutral index/weight correspondence')
    if len(obj.data.uv_layers)!=1 or obj.data.uv_layers[0].name!=UV_NAME.decode():raise ValueError('UV layer changed')
    got=[tuple(item.uv) for item in obj.data.uv_layers[0].data]
    if len(got)!=len(uv['uv']) or any(abs(a-b)>1e-7 for v,w in zip(got,uv['uv']) for a,b in zip(v,w)):
        raise ValueError('UV corner correspondence changed')
    if len(obj.material_slots)!=1:raise ValueError('skin material slot changed')
    mat=obj.material_slots[0].material
    shaders=[n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
    textures=[n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE']
    if len(shaders)!=1 or len(textures)!=1 or abs(shaders[0].inputs['Alpha'].default_value-1)>1e-7:
        raise ValueError('skin material must be opaque and image-bound')
    links=list(shaders[0].inputs['Base Color'].links)
    if len(links)!=1 or links[0].from_node!=textures[0]:raise ValueError('image not linked to body base color')
    image=textures[0].image
    data=bytes(image.packed_file.data) if image.packed_file else Path(bpy.path.abspath(image.filepath)).read_bytes()
    if hashlib.sha256(data).digest()!=hashlib.sha256(png).digest():raise ValueError('embedded image bytes changed')
    errors=compare(base,capture(origin))
    return errors,final


def run(stage,directory):
    import bpy
    directory=Path(directory);job=json.loads((directory/'job.json').read_bytes());source=directory/'source.fbx';output=directory/'artifact.fbx'
    if sha(source)!=job['source_artifact_sha256']:raise ValueError('source checksum changed')
    settings=normalize(job['skin'])
    if settings['id']=='default':raise ValueError('default must retain legacy bytes/path')
    if job['output_scope']!='full':raise ValueError('SKIN_UV_INCOMPATIBLE')
    start=time.monotonic()
    if stage=='export':
        if output.exists():raise ValueError('immutable artifact already exists')
        root,version,approved,asset,uv,png=expected_tree(source,job);write(output,root,version)
        (directory/'candidate.json').write_text(json.dumps(approved,sort_keys=True))
    elif stage=='verify':
        output_sha=sha(output);approved,asset,uv,png=verify_tree(source,output,job)
        errors,final=verify_import(source,output,asset,uv,png)
        if output_sha!=sha(output) or sha(source)!=job['source_artifact_sha256']:raise ValueError('verification modified source/output')
        receipt=dict(version='skin-fbx-receipt-v1',expression=EXPRESSION,runtime_revision=runtime_revision(),
            source_artifact_sha256=job['source_artifact_sha256'],fbx_sha256=output_sha,skin=settings,skin_sha256=skin_hash(settings),
            binding_sha256=job['binding_sha256'],character_id=job['character_id'],character_sha256=job['character_sha256'],output_scope=job['output_scope'],
            asset_manifest_sha256=manifest_hash(),asset_version=manifest()['revision'],uv_sha256=asset['uv']['sha256'],texture_sha256=hashlib.sha256(png).hexdigest(),
            base=dict(typed_fbx_values_preserved=True,approved_surface_changes_only=True,**errors),
            material=dict(manifest=approved,embedded=True,uv_verified=True,image_verified=True),
            file_verified=True,csp_certified=certified(job['character_sha256'],job['output_scope'],settings),
            blender_version=bpy.app.version_string,verification_seconds=time.monotonic()-start)
        (directory/'verified.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False))
    else:raise ValueError('invalid skin worker stage')


if __name__=='__main__':run(*sys.argv[sys.argv.index('--')+1:])
