"""Run with Blender: create immutable staging inputs from confirmed source copies."""
import sys,json,shutil,hashlib,importlib.util,math
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from converter.shape_spec import PINS
OUT=ROOT/'artifacts/shape-staging/release';OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'male':(Path('/Users/dowon/Documents/Codex/2026-10-12/gusw/outputs/male-body-0.6.7-upperbody'),'standin-body-0.6.7.blend','standin-body-0.6.7'),
 'female':(Path('/Users/dowon/.codex/worktrees/female-body-shapekeys/Standin-server/artifacts/body-shapekey-female/2026-10-15-v0.2.2/model'),'standin-female-body-0.2.2.blend','standin-female-body-0.2.2'),
}
NAMES={'BASE':'기본형','SLIM':'슬림형','IDEAL83':'8.3등신','IDEAL70':'7등신','TOON64':'6.4등신','MUSCULAR':'근육형','MUSCLE_MAX':'고근육형','SOFT':'부드러운형','FAT_MAX':'통통형','NEW_FAT':'풍만형','NEW_MUSCULAR':'고근육형'}
recipes={};bodies=[];characters={};hashes={}
for sex,(directory,blendname,rules) in SOURCES.items():
    ver=rules.rsplit('-',1)[1]; stem=sex+'-'+ver.replace('.','-')
    paths={}
    for key,name in [('blend',blendname),('controller','standin_body.py')]:
        p=directory/name;sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==PINS[rules][key=='controller']
        dest=OUT/'sources'/(stem+('.blend' if key=='blend' else '.py'));dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        paths[key]=str(dest);paths[key+'_sha256']=sha;hashes[str(p)]=sha
    bpy.ops.wm.open_mainfile(filepath=paths['blend'])
    spec=importlib.util.spec_from_file_location('release_controller',paths['controller']);sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
    mesh,arm=sb.objects();asset=mesh['asset_identity'];rig=sb.RIG_VERSION
    for preset,(_,overrides) in sb.PRESETS.items():
        values={**sb.defaults(mesh),**overrides};sb.set_controls(values,correct=False);values=sb.controls()
        exported=sb.export_fixed(str(OUT.parent/'neutral-exports'))
        meta=json.loads((exported/'manifest.json').read_text());cid=sex+'-'+preset.lower().replace('_','-')+'-v'+ver.replace('.','-')
        fbx=OUT/'fbx'/(cid+'.fbx');fbx.parent.mkdir(exist_ok=True);shutil.copy2(exported/'model.fbx',fbx)
        fields=[]
        keys=['height_cm','heads','shoulders','muscle','physique']+(['fat_gain','breast_size'] if sex=='female' else [])
        keys += [k for k in sb.FIELDS if k not in keys]
        for k in keys:
            lo,hi,d,label=sb.FIELDS[k];v=values[k];step=1 if k=='height_cm' else .000001 if k=='heads' else .01
            # Default values such as original measured heads must lie on the same grid.
            # The base (min) is aligned to the preset's exact default; endpoint is bounded.
            origin=lo
            fields.append(dict(key='height' if k=='height_cm' else k,min=origin,max=hi,step=step,default=v))
        recipe={**paths,'rules':rules,'asset':asset,'preset':preset,'asset_sha256':meta['fbx_sha256'],'defaults':values,'controls':fields}
        recipes[cid]=recipe
        name=NAMES[preset]
        if sex=='female' and preset=='MUSCULAR':name='탄탄형'
        if sex=='female' and preset=='MUSCLE_MAX':name='근육형'
        bodies.append(dict(bodyId=cid,bodyVersion=ver,characterId=cid,assetSha256=meta['fbx_sha256'],rigVersion=rig,measurementVersion='shape-authoring-v1',displayName=('남성' if sex=='male' else '여성')+' · '+name,supportedPoseIds=[],shape={'schemaVersion':'body-shape.v1','controls':fields}))
        env='BODY_'+cid.upper().replace('-','_')+'_URI'
        characters[cid]=dict(display_name=name,artifact_uri_env=env,sha256=meta['fbx_sha256'],rig_profile='mixamo',revision=ver)
        print('SHAPE_ASSET_READY',cid,flush=True)
(OUT/'shape-bakes.local.json').write_text(json.dumps({'schema':'shape-bake.v1','characters':recipes},ensure_ascii=False,indent=2))
(OUT/'body-models.local.json').write_text(json.dumps({'version':'shape-staging-20261015','defaultCharacterId':bodies[0]['characterId'],'assets':bodies},ensure_ascii=False,indent=2))
(OUT/'characters.local.json').write_text(json.dumps({'schema_version':1,'characters':characters},ensure_ascii=False,indent=2))
(OUT.parent/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
