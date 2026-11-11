"""Create an immutable preset release, preserving every published historic identity."""
import argparse,copy,hashlib,json,re,shutil,struct
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();d=Path(args.approved);r=Path(args.out);old=d/'staging-package';target=r/'package'
old_plan=read(d/'staging-upload-plan.json');catalog=read(old/'bff/body-models.json');shapes=read(old/'converter/shape-bakes.json');characters=read(old/'converter/characters.json');old_shapes=copy.deepcopy(shapes);old_chars=copy.deepcopy(characters)
plan={'scope':'staging-only','bucket':old_plan['bucket'],'roles':{},'sources':{}}
assets=[]
for g in ('male','female'):
    q=read(r/(g+'-runtime-qa.json'))
    if not q['passed'] or not q.get('independent'):raise ValueError(g+' independent runtime validation failed')
    local=r/g/'release';verification=read(local/'verification.json');new=read(local/'body-models.local.json')['assets'];recipes=read(local/'recipes.json');chars=read(local/'characters.json')
    raw=(r/(g+'.ssr')).read_bytes();h=json.loads(raw[8:8+struct.unpack('<I',raw[4:8])[0]])
    if len(raw)>32*1024*1024 or len(h['presets'])!=len(new):raise ValueError('runtime preset count/size')
    rt='runtime/'+h['rules'].replace('.','-')+'-presets.ssr';write_target=target/'bff'/rt;write_target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/(g+'.ssr'),write_target)
    for a in new:
        cid=a['characterId'];recipe=recipes[cid];preset=recipe['preset'];v=verification[preset]
        if v['neutral']['roundtrip_max_error_mm']>.01 or v['neutral']['bones']!=54 or any(not x['finite'] for x in v['poses'].values()):raise ValueError('FBX/pose QA failed')
        if h['source']!={k:a['shapeSource'][k] for k in ('blendSha256','controllerSha256')}:raise ValueError('preview/source mismatch')
        a['shapeRuntime']={'path':'/app/body-staging/'+rt,'sha256':sha(write_target),'bytes':len(raw),'format':h['format'],'rules':h['rules'],'preset':preset}
        a['supportedPoseIds']=read(d/'pose-compatibility.json')['poseIds']
        for control in a['shape']['controls']:
            key=h.get('aliases',{}).get(control['key'],control['key']);bounds=q['publishedBounds'][preset][key]
            control['min']=max(control['min'],bounds['min']);control['max']=min(control['max'],bounds['max'])
            # Align to the existing preset grid when a checked bound cuts that grid.
            import math
            control['min']=round(control['default']-math.floor((control['default']-control['min'])/control['step']+1e-6)*control['step'],6)
            if not control['min']<=control['default']<=control['max']:raise ValueError('default outside checked slider bounds')
        recipe['controls']=[{k:v for k,v in c.items() if k!='label'} for c in a['shape']['controls']]
        if cid in old_chars['characters']:
            if old_chars['characters'][cid]['sha256']!=a['assetSha256']:raise ValueError('published base identity changed')
            original=next(x for x in catalog['assets'] if x['characterId']==cid)
            a['neutralPreview']=original['neutralPreview']
        else:
            shapes['characters'][cid]=recipe;characters['characters'][cid]=chars[cid]
            for role,folder,extension in [('bff','png','.png'),('converter','fbx','.fbx')]:
                destination=target/role/folder/(cid+extension);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(local/folder/(cid+extension),destination)
            a['neutralPreview']['path']='/app/body-staging/png/'+cid+'.png'
        assets.append(a)
assert all(shapes[section][k]==v for section in ('sources','characters') for k,v in old_shapes[section].items())
assert all(characters['characters'][k]==v for k,v in old_chars['characters'].items())
catalog['assets']=assets+[a for a in catalog['assets'] if a['characterId'] not in {x['characterId'] for x in assets}];catalog['version']='anatomy-presets-20261111-v1'
write(target/'bff/body-models.json',catalog);write(target/'converter/characters.json',characters);write(target/'converter/shape-bakes.json',shapes)
for role in ('converter','bff'):
    prior=old_plan['roles'][role];manifest=read(old/role/'manifest.json');entries={f['path']:f for f in manifest['files']};local={}
    for f in (target/role).rglob('*'):
        if f.is_file() and f.name!='manifest.json':
            rel=f.relative_to(target/role).as_posix();local[rel]=str(f);entries[rel]={'path':rel,'roles':[role],'size':f.stat().st_size,'sha256':sha(f)}
    total=sum(f['size'] for f in entries.values());limit=(768 if role=='converter' else 300)*1024*1024
    if total>limit or any(not 0<f['size']<=300*1024*1024 for f in entries.values()):raise ValueError('installer size budget')
    manifest={'schema':'body-assets.v1','version':catalog['version'],'files':list(entries.values())};mp=target/role/'manifest.json';write(mp,manifest);digest=sha(mp)
    if mp.stat().st_size>65536:raise ValueError('manifest size')
    prefix=(prior['prefix']+'/presets-'+digest[:12]) if role=='bff' else 'body-models/releases/anatomy-presets-20261111-'+digest[:12]+'-converter'
    plan['roles'][role]={'prefix':prefix,'manifestSha256':digest,'manifest':str(mp),'upload':[{'path':rel,'local':path} for rel,path in local.items()],'copy':[{'path':rel,'sourceKey':prior['prefix']+'/'+rel,'sha256':f['sha256'],'size':f['size']} for rel,f in entries.items() if rel not in local]}
    print('PACKAGE',role,len(entries),round(total/1024**2,2),'MiB',prefix,flush=True)
# The analysis worker only captures selection identities and control specs. It
# receives the identical catalog within its existing read-only S3 prefix; no
# model binaries, thumbnail copies or new IAM grant are needed there.
catalog_file=target/'bff/body-models.json'
wm={'schema':'body-assets.v1','version':catalog['version'],'files':[{'path':'body-models.json','roles':['bff'],'size':catalog_file.stat().st_size,'sha256':sha(catalog_file)}]}
wp=target/'worker/manifest.json';write(wp,wm);wh=sha(wp)
plan['roles']['worker']={'prefix':'body-models/releases/staging-9-20261012-9acb2ee7ec24/presets-'+wh[:12],'manifestSha256':wh,'manifest':str(wp),'upload':[{'path':'body-models.json','local':str(catalog_file)}],'copy':[]}
write(r/'staging-upload-plan.json',plan)
write(r/'release-summary.json',{'version':catalog['version'],'presets':[{'characterId':a['characterId'],'assetSha256':a['assetSha256'],'shapeSource':a['shapeSource'],'runtimeSha256':a['shapeRuntime']['sha256']} for a in assets],'roles':{k:{'prefix':v['prefix'],'manifestSha256':v['manifestSha256']} for k,v in plan['roles'].items()}})
