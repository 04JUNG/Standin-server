"""Private approved final-FBX skin postprocess, independent of outline JSON."""
import base64,hashlib,hmac,json,os,re,subprocess,tempfile,time
from pathlib import Path
from fastapi import File,Form,Request,UploadFile
from fastapi.responses import JSONResponse,Response
from converter.skin_spec import (VERSION,EXPRESSION,SKINS,MAX_FBX_BYTES,normalize,skin_hash,
    manifest,manifest_hash,runtime_revision,certified,checked_asset,CSP_CERTIFIED)
from converter.scene_contract import FBX_MAGIC
from converter.framing import validate_scope
from converter_api.work_queue import current_work
SHA=re.compile(r'^[0-9a-f]{64}$')


def prepare(runner,data,job,*,checkpoint):
    runner.inspect_blender()
    with runner._process_slots,tempfile.TemporaryDirectory(prefix='standin-skin-',dir=runner.settings.temp_root) as tmp:
        root=Path(tmp);(root/'source.fbx').write_bytes(data);job_bytes=json.dumps(job,sort_keys=True,allow_nan=False).encode();(root/'job.json').write_bytes(job_bytes)
        for stage in ('export','verify'):
            checkpoint();started=time.monotonic();command=[runner.settings.blender_binary,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(Path(__file__).resolve().parents[1]/'converter/skin_worker.py'),'--',stage,str(root)]
            with tempfile.TemporaryFile() as log:
                process=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=(os.name=='posix'),env=runner._environment())
                try:
                    while process.poll() is None:
                        checkpoint()
                        if time.monotonic()-started>runner.settings.shape_timeout_seconds:raise TimeoutError('SKIN_TIMEOUT')
                        time.sleep(.05)
                    checkpoint()
                    if process.returncode:
                        log.seek(0);error=log.read().decode(errors='replace')
                        for code in ('SKIN_UV_INCOMPATIBLE','SKIN_OUTLINE_ORDER','SKIN_ASSETS_UNAVAILABLE'):
                            if code in error:raise ValueError(code)
                        raise ValueError('SKIN_INTEGRITY_ERROR')
                finally:
                    if process.poll() is None:runner._terminate_process_group(process,runner.settings.terminate_grace_seconds)
        output=(root/'artifact.fbx').read_bytes();receipt=json.loads((root/'verified.json').read_bytes());asset,_,png=checked_asset(job['character_id'],job['character_sha256'],job['skin'])
        if ((root/'job.json').read_bytes()!=job_bytes or (root/'source.fbx').read_bytes()!=data
                or len(output)>MAX_FBX_BYTES or not output.startswith(FBX_MAGIC)
                or receipt.get('fbx_sha256')!=hashlib.sha256(output).hexdigest()
                or any(receipt.get(k)!=job[k] for k in ('source_artifact_sha256','skin','binding_sha256','character_id','character_sha256','output_scope'))
                or receipt.get('runtime_revision')!=runtime_revision() or receipt.get('expression')!=EXPRESSION
                or receipt.get('asset_manifest_sha256')!=manifest_hash() or receipt.get('asset_version')!=manifest()['revision'] or receipt.get('skin_sha256')!=skin_hash(job['skin'])
                or receipt.get('uv_sha256')!=asset['uv']['sha256'] or receipt.get('texture_sha256')!=hashlib.sha256(png).hexdigest()
                or receipt.get('file_verified') is not True or receipt.get('base',{}).get('typed_fbx_values_preserved') is not True
                or any(receipt.get('material',{}).get(k) is not True for k in ('embedded','uv_verified','image_verified'))
                or receipt.get('csp_certified') is not certified(job['character_sha256'],job['output_scope'],job['skin'])):
            raise ValueError('SKIN_INTEGRITY_ERROR')
        checkpoint();return output,receipt


def register_skin(app,registry,runner):
    from converter_api.app import ApiProblem
    def auth(request):
        token=os.getenv('CONVERTER_INTERACTIVE_TOKEN','')
        if len(token)<32 or not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+token):raise ApiProblem(401,'UNAUTHORIZED','service authentication required')
    def enabled():return os.getenv('CONVERTER_SKIN_ENABLED','false').lower()=='true'
    def files():
        for cid,row in manifest()['models'].items():
            yield cid,'uv',None,row['uv']
            for sid,entry in row['textures'].items():yield cid,'texture',sid,entry
            for sid,entry in row['preview_textures'].items():yield cid,'preview-texture',sid,entry
    def asset_data(entry):
        root_value=os.getenv('CONVERTER_SKIN_ASSET_ROOT','')
        if not root_value:raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        root=Path(root_value).resolve(strict=True);p=root/entry['path']
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        data=p.read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('SKIN_ASSETS_UNAVAILABLE')
        return data
    @app.get('/skin-contract')
    def contract():
        available=True
        try:
            for _,_,_,entry in files():asset_data(entry)
        except (ValueError,OSError):available=False
        targets=[dict(character_id=cid,character_sha256=row['character_sha256'],output_scope='full',skin_ids=[sid for sid in SKINS[1:] if certified(row['character_sha256'],'full',dict(version=VERSION,id=sid))]) for cid,row in manifest()['models'].items()]
        targets=[row for row in targets if row['skin_ids']]
        assets=[dict(character_id=cid,kind=kind,skin_id=sid,sha256=entry['sha256'],bytes=entry['bytes'],url='/skin-assets/'+entry['sha256']) for cid,kind,sid,entry in files()]
        return JSONResponse(dict(version=VERSION,expression=EXPRESSION,enabled=enabled(),assets_available=available,
            csp_certified=bool(targets),acceptance_scope='posed-fbx-import',runtime_revision=runtime_revision(),
            asset_manifest_sha256=manifest_hash(),asset_version=manifest()['revision'],skins=list(SKINS),supported_targets=targets,
            candidate_models={cid:{k:v for k,v in row.items() if k not in ('uv','textures','preview_textures')} for cid,row in manifest()['models'].items()},assets=assets,
            zero='default-legacy-byte-preserving',order=['skin','outline']),headers={'Cache-Control':'no-store'})
    @app.get('/skin-assets/{digest}')
    def get_asset(digest:str,request:Request):
        auth(request)
        entry=next((e for _,_,_,e in files() if e['sha256']==digest),None)
        if not SHA.fullmatch(digest) or entry is None:raise ApiProblem(404,'SKIN_ASSET_NOT_FOUND','unknown approved asset')
        try:data=asset_data(entry)
        except (ValueError,OSError):raise ApiProblem(503,'SKIN_ASSETS_UNAVAILABLE','approved skin assets unavailable')
        return Response(data,media_type='application/octet-stream',headers={'X-Skin-Asset-SHA256':digest,'Cache-Control':'private, max-age=3600, immutable'})
    @app.post('/skin-fbx')
    def skin(request:Request,fbx:UploadFile=File(...),source_artifact_sha256:str=Form(...),skin:str=Form(...),binding_sha256:str=Form(...),character_id:str=Form(...),character_sha256:str=Form(...),output_scope:str=Form(...)):
        auth(request)
        try:
            settings=normalize(json.loads(skin));validate_scope(output_scope)
            if any(not SHA.fullmatch(v) for v in (source_artifact_sha256,binding_sha256,character_sha256)):raise ValueError('bad binding')
        except (ValueError,TypeError):raise ApiProblem(400,'INVALID_SKIN','invalid skin settings')
        if settings['id']=='default':raise ApiProblem(400,'SKIN_DEFAULT_USE_LEGACY','use the original FBX')
        if not enabled():raise ApiProblem(503,'SKIN_DISABLED','skin preparation unavailable')
        row=manifest()['models'].get(character_id)
        if not row or row['character_sha256']!=character_sha256 or output_scope!='full':raise ApiProblem(422,'SKIN_UNSUPPORTED','unsupported skin target')
        try:
            if registry.metadata(character_id).sha256!=character_sha256:raise ValueError('changed')
        except Exception:raise ApiProblem(409,'SKIN_BINDING_MISMATCH','skin character binding changed')
        if not certified(character_sha256,output_scope,settings):raise ApiProblem(503,'SKIN_NOT_CSP_CERTIFIED','skin import acceptance incomplete')
        try:checked_asset(character_id,character_sha256,settings)
        except (ValueError,OSError):raise ApiProblem(503,'SKIN_ASSETS_UNAVAILABLE','approved skin assets unavailable')
        data=fbx.file.read(MAX_FBX_BYTES+1)
        if len(data)>MAX_FBX_BYTES or not data.startswith(FBX_MAGIC):raise ApiProblem(400,'INVALID_SKIN','expected bounded binary FBX')
        if hashlib.sha256(data).hexdigest()!=source_artifact_sha256:raise ApiProblem(409,'SOURCE_FBX_SHA256_MISMATCH','source FBX changed')
        job=dict(source_artifact_sha256=source_artifact_sha256,skin=settings,binding_sha256=binding_sha256,character_id=character_id,character_sha256=character_sha256,output_scope=output_scope)
        control=current_work.get()
        def checkpoint():
            if control:control.check('execution')
        try:output,receipt=prepare(runner,data,job,checkpoint=checkpoint)
        except TimeoutError:raise ApiProblem(504,'SKIN_TIMEOUT','skin preparation timed out')
        except (ValueError,OSError) as exc:
            code=str(exc) if str(exc) in ('SKIN_UV_INCOMPATIBLE','SKIN_OUTLINE_ORDER','SKIN_ASSETS_UNAVAILABLE') else 'SKIN_INTEGRITY_ERROR'
            raise ApiProblem(422 if code in ('SKIN_UV_INCOMPATIBLE','SKIN_OUTLINE_ORDER') else 503 if code=='SKIN_ASSETS_UNAVAILABLE' else 500,code,'skin file verification failed')
        if receipt.get('csp_certified') is not True:raise ApiProblem(503,'SKIN_NOT_CSP_CERTIFIED','skin import acceptance incomplete')
        checkpoint();return JSONResponse(dict(artifact_contract='textured-fbx-v1',fbx_base64=base64.b64encode(output).decode('ascii'),fbx_sha256=receipt['fbx_sha256'],receipt=receipt),headers={'Cache-Control':'private, no-store'})
