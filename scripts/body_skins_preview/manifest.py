"""Create a relocatable skin manifest with asset hashes and explicit verification scope."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;assets=out/'assets';here=Path(__file__).resolve().parent
sources=json.loads((here/'source_assets.json').read_text());style=json.loads((here/'skin_style.json').read_text())
manifest={'schema':'standin-body-skins-preview.v1','version':'0.1.2','status':'review_draft','base_commit':sources['base_commit'],'source_evidence':sources['evidence'],'production_registry_queried':False,'default_material':{'source':'converter/preview_model.py existing PBR','base_color_linear':[.72,.74,.76,1],'roughness':.8,'metallic':0,'source_authoring_materials_preserved_in_blend':True},'uv':{'layer':'StandinSkinUV_v1','source_uv_layers':[],'distance_range_m':.006,'channels':{'R':'sagittal/limb axis','G':'construction sections','B':'major muscle boundaries'},'generation':'rest-space authored curves baked to new UV; orange axes and muscles use ray-projected 3D surface distance'},'skins':[
 {'skinId':'default','version':'0.1.2','label':'기본','base':'existing preview PBR','mask':None},
 {'skinId':'gray-guide','version':'0.1.2','label':'회색 가이드선','base_color_linear':[.72,.74,.76],'channels':['R','G']},
 {'skinId':'flesh','version':'0.1.2','label':'살색','base_color_srgb':'#fadfca','channels':[]},
 {'skinId':'flesh-guide','version':'0.1.2','label':'살색 가이드선','base_color_srgb':'#fadfca','channels':['R','G']},
 {'skinId':'muscle','version':'0.1.2','label':'근육선 강조','base_color_srgb':'#d3c3b1','channels':['B'],'base_rationale':'Low-saturation clay separates anatomy lines from shading; base color is independently editable.'}],
 'line_parameters':{'center':{'color_srgb':'#ff8a35','half_width_m':.0022,'opacity':1.0},'construction':{'color_srgb':'#5b646e','half_width_m':.00135,'opacity':.62},'muscle':{'color_srgb':'#42332e','half_width_m':.00165,'opacity':.9},'preview_width_multiplier':[.5,2],'preview_opacity':[.2,1]},
 'rendering':{'preview':'Three.js MeshStandardMaterial + UV distance-mask shader; line color/width/opacity and base color independent','portable':'ordinary sRGB base-color PNGs and embedded-texture neutral FBXs','preview_default_source':'converter/preview_model.py: PBR [.72,.74,.76,1], roughness .8','lighting':'neutral fixed studio; same lights/camera for each comparison','poses':'Blender full-original-weight evaluated snapshots; no browser four-weight approximation','rigged_glb':'all influences exported; not used for pose QA because browser skinning support can truncate weights'},
 'verified_scope':{'models':['male-base-v0-7-2','female-base-v0-3-4'],'poses':['standing','bent','seated'],'deformation':'Spine1 local scale [1.12,1,1.08], diagnostic copy only; not an approved body template','all_body_templates':False,'fbx_blender_roundtrip':True,'csp_tested':False,'production_integration':False},'models':[],'assets':[],'style_parameters':style}
for row in sources['models']:
 name=row['id'];source=json.loads((assets/(name+'-source.json')).read_text());uv=json.loads((assets/(name+'-uv-report.json')).read_text());fbx=json.loads((assets/(name+'-fbx-roundtrip.json')).read_text())
 manifest['models'].append({**row,'protected_before':source['protected_before'],'protected_after':source['protected_after'],'blender':source['blender'],'blender_build':source['blender_build'],'uv_report':uv,'fbx_roundtrip':fbx,'path_templates':{'preview_glb':f'assets/{name}-{{pose}}.glb','authoring':f'assets/{name}.blend','rigged_glb':f'assets/{name}.glb','masks':f'assets/{name}-masks.png','baked_texture':f'assets/{name}-{{skinId}}.png','fbx':f'assets/fbx/{name}-{{skinId}}.fbx','authored_paths':f'assets/{name}-paths.json'}})
for path in sorted(assets.rglob('*')):
 if path.is_file() and path.suffix not in ('.blend1','.npz'):
  manifest['assets'].append({'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('MANIFEST',len(manifest['assets']),'assets')
