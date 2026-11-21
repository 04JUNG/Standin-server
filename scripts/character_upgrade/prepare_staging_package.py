"""Merge verified new anatomy into immutable staging catalogs, preserving old IDs.

Writes a reviewable local package and upload plan. Does not activate any service.
"""
import argparse,copy,hashlib,json,shutil,struct,re
from pathlib import Path
BUCKET='standinstagingapp-assetsbucket5cb76180-fiwlikrhl1sk'
OLD={'converter':'body-models/releases/csp-shape-20261030-170c4df72c41-converter',
     'bff':'body-models/releases/csp-shape-20261030-45ee1c64fa31-bff'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--converter-only',action='store_true')
args=ap.parse_args();root=Path(args.root);release=root/'release-verified';active=root/'active';package=root/('staging-package-converter' if args.converter_only else 'staging-package')
if package.exists():raise ValueError('use a fresh immutable package')
registry=read(release/'shape-bakes.local.json');chars=read(active/'characters.json');old_chars=copy.deepcopy(chars)
chars['characters'].update(read(release/'characters.local.json')['characters'])
shapes=read(active/'shape-bakes.json');old_shapes=copy.deepcopy(shapes)
catalog=read(active/'body-models.json');new=read(release/'body-models.local.json')
plan={'bucket':BUCKET,'roles':{},'scope':'staging-only','sources':{}}
for role in OLD:(package/role).mkdir(parents=True)
for gender in ('male','female'):
    qa=read(root/'authoring'/gender/'validation.json')
    if qa['default_error_mm']>.01:raise ValueError('approved default identity changed')
    conversion=release/'conversion-qa'/(gender+'-Arm_Stretching_00054')
    if not read(conversion/'result.json')['ok']:raise ValueError('actual pose conversion failed')
    cid=gender+'-base-v0-'+('8' if gender=='male' else '4')+'-0'
    body=next(x for x in new['assets'] if x['characterId']==cid);rule=body['shapeSource']['rules']
    source=registry['sources'][rule]
    for key,ext in [('blend','.blend'),('controller','.py')]:
        rel='sources/'+rule.replace('.','-')+ext;dst=package/'converter'/rel;dst.parent.mkdir(exist_ok=True);shutil.copyfile(source[key],dst)
        assert sha(dst)==source[key+'_sha256'];source[key]='/app/body-staging/'+rel
    for folder,role,ext in [('fbx','converter','.fbx'),('png','bff','.png')]:
        rel=folder+'/'+cid+ext;dst=package/role/rel;dst.parent.mkdir(exist_ok=True);shutil.copyfile(release/rel,dst)
    assert sha(package/'converter/fbx'/(cid+'.fbx'))==body['assetSha256']
    if args.converter_only:continue
    runtime=root/(gender+'.ssr');raw=runtime.read_bytes()
    if raw[:4]!=b'SSR2' or len(raw)>32*1024*1024:raise ValueError('unsupported runtime format/size')
    h=json.loads(raw[8:8+struct.unpack('<I',raw[4:8])[0]])
    expected={'blendSha256':source['blend_sha256'],'controllerSha256':source['controller_sha256']}
    if h['source']!=expected or h['rules']!=rule:raise ValueError('preview/export source mismatch')
    # Independent sample verification is required in addition to training/pruning QA.
    independent=read(root/(gender+'-runtime-independent.json'))
    if not independent.get('passed'):raise ValueError('independent runtime validation did not pass')
    if independent['quality']['BASE']['preset_mm']>.5:raise ValueError('default preview error exceeds 0.5 mm')
    if h['controls']!=independent['publishedControls']:raise ValueError('published range was not independently checked')
    for control in body['shape']['controls']:
        key=h.get('aliases',{}).get(control['key'],control['key']);bounds=h['controls'][key]
        control['min']=max(control['min'],bounds['min']);control['max']=min(control['max'],bounds['max'])
        if not control['min']<=control['default']<=control['max']:raise ValueError('default outside published bounds')
    body['supportedPoseIds']=read(root/'pose-compatibility.json')['poseIds']
    dst=package/'bff/runtime'/(rule.replace('.','-')+'.ssr');dst.parent.mkdir(exist_ok=True);shutil.copyfile(runtime,dst)
    body['shapeRuntime']={'path':'/app/body-staging/runtime/'+rule.replace('.','-')+'.ssr','sha256':sha(dst),'bytes':len(raw),
                          'format':h['format'],'rules':rule,'preset':'BASE'}
    body['neutralPreview']['path']='/app/body-staging/png/'+cid+'.png'
    assert body['neutralPreview']['sha256']==sha(package/'bff/png'/(cid+'.png'))
    plan['sources'][gender]={'approvedFbxSha256':read(root/'authoring'/gender/'build.json')['source_fbx_sha256'],
                            'characterId':cid,'assetSha256':body['assetSha256'],**expected}
shapes['sources'].update(registry['sources']);shapes['characters'].update(registry['characters'])
for a in catalog['assets']:
    if a.get('role')=='default':a['role']='previous_default'
catalog['assets']=new['assets']+catalog['assets'];catalog['version']=new['version'];catalog['defaultCharacterId']=new['defaultCharacterId']
assert all(chars['characters'][k]==v for k,v in old_chars['characters'].items())
assert all(shapes[section][k]==v for section in ('characters','sources') for k,v in old_shapes[section].items())
write(package/'converter/characters.json',chars);write(package/'converter/shape-bakes.json',shapes);write(package/'bff/body-models.json',catalog)
write(package/'converter/body-models.json',catalog)  # Future releases carry the exact BFF defaults for prewarming.
for role in (['converter'] if args.converter_only else OLD):
    manifest=read(active/(role+'-manifest.json'));entries={f['path']:f for f in manifest['files']}
    local={}
    for p in (package/role).rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(package/role).as_posix();local[rel]=str(p)
        entries[rel]={'path':rel,'roles':[role],'size':p.stat().st_size,'sha256':sha(p)}
    manifest={'schema':'body-assets.v1','version':catalog['version'],'files':list(entries.values())}
    allowed=re.compile(r'(?:shape-bakes\.json|sources/[a-z0-9-]+\.(?:blend|py)|runtime/[a-z0-9-]+\.ssr|body-catalog\.json|body-models\.json|characters\.json|fbx/[a-z0-9-]+\.fbx|png/[a-z0-9-]+\.png|qa/[a-z0-9-]+\.json|preview/posed-mesh-v1/[a-f0-9]{64}\.(?:json|glb))\Z')
    if any(not allowed.fullmatch(f['path']) or not 0<f['size']<=300*1024*1024 for f in entries.values()):raise ValueError('asset violates deployed installer path/size contract')
    if sum(f['size'] for f in entries.values())>(512 if role=='converter' else 300)*1024*1024:raise ValueError('release exceeds reviewed installer total limit')
    target=package/role/'manifest.json';write(target,manifest);digest=sha(target)
    if target.stat().st_size>65536:raise ValueError('manifest exceeds installer limit')
    prefix='body-models/releases/anatomy-20261111-'+digest[:12]+'-'+role
    plan['roles'][role]={'prefix':prefix,'manifestSha256':digest,'manifest':str(target),
        'upload':[{'path':rel,'local':path} for rel,path in local.items()],
        'copy':[{'path':rel,'sourceKey':OLD[role]+'/'+rel,'sha256':f['sha256'],'size':f['size']} for rel,f in entries.items() if rel not in local]}
write(root/'staging-upload-plan.json',plan)
print(json.dumps({'roles':{k:{'prefix':v['prefix'],'newFiles':len(v['upload']),'preservedFiles':len(v['copy'])} for k,v in plan['roles'].items()},'sources':plan['sources']},indent=2))
