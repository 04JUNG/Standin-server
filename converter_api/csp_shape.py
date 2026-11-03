"""Shape source validation + bounded fresh Blender execution, sharing work slots."""
import base64,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
from fastapi import Body
from fastapi.responses import JSONResponse
from converter.shape_spec import resolve_shape,ShapeError
from converter_api.work_queue import current_work,WorkStopped
from converter_api.registry import RegistryError


def validate_request(payload,registry):
    if not isinstance(payload,dict) or set(payload)!={'character_id','character_sha256','shape','expected_shape_source'}:
        raise ShapeError('invalid request fields')
    if len(json.dumps(payload))>20000:raise ShapeError('payload too large')
    if not isinstance(payload['shape'],dict) or not isinstance(payload['expected_shape_source'],dict):raise ShapeError('explicit shape and source required')
    meta=registry.metadata(payload['character_id'])
    if meta.sha256!=payload['character_sha256']:raise ShapeError('character checksum mismatch')
    recipe=resolve_shape(payload['character_id'],meta.sha256,payload['shape'],payload['expected_shape_source'])
    if recipe is None or recipe['shape_sha256'] is None:raise ShapeError('explicit shape required')
    return recipe


def run_shape(runner,payload,recipe):
    runner.inspect_blender();control=current_work.get()
    runner._process_slots.acquire()
    try:
        with tempfile.TemporaryDirectory(prefix='standin-csp-shape-',dir=runner.settings.temp_root) as temp:
            root=Path(temp);job={'character_id':payload['character_id'],'recipe':recipe,'output':str(root/'model.vrm'),'report':str(root/'report.json')};p=root/'job.json';p.write_text(json.dumps(job))
            worker=Path(__file__).resolve().parents[1]/'converter/csp_shape_worker.py'
            command=[runner.settings.blender_binary,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(worker),'--',str(p)]
            with (root/'worker.log').open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,**dict(runner.settings.process_env or {})})
                started=time.monotonic()
                try:
                    while proc.poll() is None:
                        if control:control.check('csp_shape_bake')
                        if time.monotonic()-started>runner.settings.shape_timeout_seconds:raise TimeoutError('shape timeout')
                        time.sleep(.1)
                    if proc.returncode!=0:raise RuntimeError('CSP shape worker failed')
                finally:
                    if proc.poll() is None:runner._terminate_process_group(proc,runner.settings.terminate_grace_seconds)
            raw=(root/'model.vrm').read_bytes();report=json.loads((root/'report.json').read_text())
            if len(raw)>25*1024*1024 or raw[:4]!=b'glTF' or hashlib.sha256(raw).hexdigest()!=report['artifact_sha256']:raise RuntimeError('CSP worker integrity')
            if report['shape_sha256']!=recipe['shape_sha256'] or report['controls']!=recipe['values']:raise RuntimeError('CSP shape lineage')
            if report['source']!={'rules':recipe['rules'],'blendSha256':recipe['blend_sha256'],'controllerSha256':recipe['controller_sha256']}:raise RuntimeError('CSP source lineage')
            return {'vrm_base64':base64.b64encode(raw).decode(),'report':report}
    finally:runner._process_slots.release()


def register_csp_shape(app,registry,runner):
    @app.post('/csp-shape')
    def csp_shape(payload:dict=Body(...)):
        if os.getenv('CSP_SHAPE_ENABLED')!='true':return JSONResponse({'error':{'code':'CSP_SHAPE_DISABLED'}},status_code=404)
        try:recipe=validate_request(payload,registry)
        except (ShapeError,RegistryError,KeyError,TypeError,ValueError):return JSONResponse({'error':{'code':'CSP_SHAPE_INVALID'}},status_code=409)
        try:return run_shape(runner,payload,recipe)
        except TimeoutError:return JSONResponse({'error':{'code':'CSP_SHAPE_TIMEOUT'}},status_code=504)
        except WorkStopped:raise
        except (RuntimeError,OSError,ValueError,KeyError):return JSONResponse({'error':{'code':'CSP_SHAPE_FAILED'}},status_code=502)
