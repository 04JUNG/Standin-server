"""Opt-in, fail-closed final appearance boundary, separate from pose solving."""
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from fastapi import File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from converter.outline_spec import (VERSION, EXPRESSION, MAPPING, CSP_CERTIFIED,
    CHARACTERS, MAX_FBX_BYTES, normalize, appearance_hash, thickness_m, physical_thickness,
    runtime_revision, certified, DEFAULT_SURFACE, ACCEPTANCE_SCOPE, supported_targets)
from converter.scene_contract import FBX_MAGIC
from converter.framing import validate_scope
from converter_api.work_queue import current_work, measured_slot

SHA = re.compile(r'^[0-9a-f]{64}$')


def prepare(runner, data, job, *, checkpoint):
    from converter_api.appearance_cache import verified_appearance
    return verified_appearance('outline', data, job, runtime_revision(),
        lambda: _prepare_uncached(runner, data, job, checkpoint=checkpoint), checkpoint)


def _prepare_uncached(runner, data, job, *, checkpoint):
    """Fresh exporter + fresh verifier; publish nothing after cancellation."""
    runner.inspect_blender()
    with measured_slot(runner._process_slots, worker=True), tempfile.TemporaryDirectory(prefix='standin-outline-',dir=runner.settings.temp_root) as tmp:
        root=Path(tmp); (root/'source.fbx').write_bytes(data)
        job_bytes=json.dumps(job,sort_keys=True,allow_nan=False).encode()
        (root/'job.json').write_bytes(job_bytes)
        for stage in ('export','verify'):
            checkpoint();start=time.monotonic()
            command=[runner.settings.blender_binary,'--background','--factory-startup',
                '--disable-autoexec','--python-exit-code','1','--python',
                str(Path(__file__).resolve().parents[1]/'converter/outline_worker.py'),'--',stage,str(root)]
            with tempfile.TemporaryFile() as log:
                p=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=(os.name=='posix'),env=runner._environment())
                try:
                    while p.poll() is None:
                        checkpoint()
                        if time.monotonic()-start>runner.settings.shape_timeout_seconds:
                            raise TimeoutError('OUTLINE_TIMEOUT')
                        time.sleep(.05)
                    checkpoint()
                    if p.returncode: raise ValueError('OUTLINE_INTEGRITY_ERROR')
                finally:
                    if p.poll() is None:
                        runner._terminate_process_group(p,runner.settings.terminate_grace_seconds)
        output=(root/'artifact.fbx').read_bytes();receipt=json.loads((root/'verified.json').read_bytes())
        if ((root/'job.json').read_bytes()!=job_bytes or (root/'source.fbx').read_bytes()!=data
                or len(output)>MAX_FBX_BYTES or not output.startswith(FBX_MAGIC)
                or receipt.get('fbx_sha256')!=hashlib.sha256(output).hexdigest()
                or receipt.get('runtime_revision')!=runtime_revision()
                or any(receipt.get(k)!=job[k] for k in ('source_artifact_sha256','appearance',
                    'binding_sha256','character_id','character_sha256','output_scope','stature_m','culling'))
                or receipt.get('appearance_sha256')!=appearance_hash(job['appearance'])
                or receipt.get('expression')!=EXPRESSION or receipt.get('mapping')!=MAPPING
                or receipt.get('thickness_m')!=physical_thickness(job['appearance'],job['stature_m'],
                    job['output_scope'],surface=job.get('surface',DEFAULT_SURFACE))
                or receipt.get('file_verified') is not True
                or receipt.get('base',{}).get('typed_fbx_values_preserved') is not True
                or receipt.get('outline',{}).get('skin_verified') is not True
                or receipt.get('outline',{}).get('displacement_verified') is not True
                or receipt.get('csp_certified') is not certified(job['character_sha256'],
                    job['output_scope'],job['appearance']['width'],
                    surface=job.get('surface',DEFAULT_SURFACE))):
            raise ValueError('OUTLINE_INTEGRITY_ERROR')
        checkpoint()
        return output,receipt


def register_outline(app,registry,runner):
    from converter_api.app import ApiProblem

    def enabled(): return os.getenv('CONVERTER_OUTLINE_ENABLED','false').lower()=='true'

    @app.get('/outline-contract')
    def contract():
        return JSONResponse(dict(version=VERSION,enabled=enabled(),expression=EXPRESSION,
            mapping=MAPPING,runtime_revision=runtime_revision(),csp_certified=bool(CSP_CERTIFIED),
            certified_expressions=sorted({e[0] for e in CSP_CERTIFIED}),
            supported_characters=sorted(k for k,v in CHARACTERS.items() if any(e[2]==v for e in CSP_CERTIFIED)),
            supported_scopes=sorted({e[3] for e in CSP_CERTIFIED}),
            supported_targets=supported_targets(),acceptance_scope=ACCEPTANCE_SCOPE,
            candidate_characters=CHARACTERS,width=dict(min=0,max=2),zero='legacy-byte-preserving'),
            headers={'Cache-Control':'no-store'})

    @app.post('/outline-fbx')
    def outline(request:Request,fbx:UploadFile=File(...),source_artifact_sha256:str=Form(...),
            appearance:str=Form(...),output_scope:str=Form(...),character_id:str=Form(...),
            character_sha256:str=Form(...),binding_sha256:str=Form(...),stature_m:float=Form(...)):
        # Service token matches the existing private interactive trust boundary.
        token=os.getenv('CONVERTER_INTERACTIVE_TOKEN','')
        if len(token)<32 or not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+token):
            raise ApiProblem(401,'UNAUTHORIZED','service authentication required')
        try:
            settings=normalize(json.loads(appearance));validate_scope(output_scope)
            thickness_m(settings,stature_m)
            if any(not SHA.fullmatch(v) for v in (source_artifact_sha256,character_sha256,binding_sha256)):
                raise ValueError('invalid binding')
        except (ValueError,TypeError):
            raise ApiProblem(400,'INVALID_OUTLINE','invalid outline settings')
        # Zero must stay on the existing save path. Do not manufacture a new
        # receipt/path/identity for it, even if this endpoint is called in error.
        if not settings['width']:
            raise ApiProblem(400,'OUTLINE_ZERO_USE_LEGACY','use the original FBX')
        if not enabled(): raise ApiProblem(503,'OUTLINE_DISABLED','outline preparation unavailable')
        if CHARACTERS.get(character_id)!=character_sha256:
            raise ApiProblem(422,'OUTLINE_UNSUPPORTED','unsupported outline character')
        try:
            if registry.metadata(character_id).sha256!=character_sha256: raise ValueError('character changed')
        except Exception:
            raise ApiProblem(409,'OUTLINE_BINDING_MISMATCH','outline character binding changed')
        if not certified(character_sha256,output_scope,settings['width']):
            raise ApiProblem(503,'OUTLINE_NOT_CSP_CERTIFIED','outline import acceptance incomplete')
        data=fbx.file.read(MAX_FBX_BYTES+1)
        if len(data)>MAX_FBX_BYTES or not data.startswith(FBX_MAGIC):
            raise ApiProblem(400,'INVALID_OUTLINE','expected bounded binary FBX')
        if hashlib.sha256(data).hexdigest()!=source_artifact_sha256:
            raise ApiProblem(409,'SOURCE_FBX_SHA256_MISMATCH','source FBX changed')
        job=dict(source_artifact_sha256=source_artifact_sha256,appearance=settings,
            output_scope=output_scope,character_id=character_id,character_sha256=character_sha256,
            binding_sha256=binding_sha256,stature_m=stature_m,culling='CullingOff',surface=DEFAULT_SURFACE)
        control=current_work.get()
        def check():
            if control: control.check('execution')
        try: output,receipt=prepare(runner,data,job,checkpoint=check)
        except TimeoutError: raise ApiProblem(504,'OUTLINE_TIMEOUT','outline preparation timed out')
        except (ValueError,OSError): raise ApiProblem(500,'OUTLINE_INTEGRITY_ERROR','outline verification failed')
        # A file proof alone must never become a CSP-ready download.
        if receipt.get('csp_certified') is not True:
            raise ApiProblem(503,'OUTLINE_NOT_CSP_CERTIFIED','outline import acceptance incomplete')
        check()
        return JSONResponse(dict(artifact_contract='outlined-fbx-v1',
            fbx_base64=base64.b64encode(output).decode('ascii'),
            fbx_sha256=receipt['fbx_sha256'],receipt=receipt),headers={'Cache-Control':'private, no-store'})
