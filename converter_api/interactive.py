"""Opt-in private snapshot transport. BFF retains release/ownership authority.

One service instance/private volume; bounded entries/TTL and existing Blender
scheduler. Opaque IDs and service authentication, no client-authored file paths.
Restart invalidates outstanding operations instead of reviving ready markers.
"""
from __future__ import annotations
import base64, hmac, json, logging, os, re, shutil, stat, tempfile, threading, time, uuid
from pathlib import Path
from fastapi import Request, HTTPException
from fastapi.responses import Response
from converter.interactive_adapter import VERSION, atomic_write, body_binding, runtime_revision, shape_binding, load_solved
from converter.interactive_camera import camera_for_model
from converter.interactive_export import export_bundle, Superseded
from converter.interactive_revision import LatestRevisionGate
from converter.solved_state import canonical, decode, digest
from converter.hand_library import load_library, normalize_request
from converter.protocol import SOLVER_VERSION, SOLVER_MANIFEST_SHA256, JOB_SCHEMA_VERSION
from converter.shape_spec import resolve_shape
from converter_api.work_queue import current_work, measured_slot
from converter.interactive_reuse import core_binding, rebind_solved, rebind_export

MAX_REQUEST = 3 * 1024 * 1024
MAX_RESPONSE = 64 * 1024 * 1024
ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
SHA = re.compile(r'^[a-f0-9]{64}$')


def measured(stage, function, *args):
    """Correlate private work without logging geometry, paths or owner bindings."""
    started = time.monotonic()
    succeeded = False
    try:
        result = function(*args)
        succeeded = True
        return result
    finally:
        control = current_work.get()
        logging.getLogger('standin.converter').info(json.dumps(dict(
            event='interactive_stage', stage=stage,
            duration_ms=round((time.monotonic()-started)*1000, 3),
            succeeded=succeeded,
            request_id=control.request_id if control else None)))


class SnapshotStore:
    SOLVE_RESERVE_BYTES = 512 * 1024**2

    def __init__(self, root, registry, runner, *, ttl=3600, capacity=64, byte_limit=2*1024**3):
        requested=Path(root)
        if requested.is_symlink():raise ValueError('private root cannot be a symlink')
        self.root=requested.resolve();self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        # One active service owns this private volume. Fail closed on accidental
        # multi-process deployment; restart cannot restore stale ready state.
        import fcntl
        self._volume_lock=(self.root/'.service.lock').open('a+b')
        fcntl.flock(self._volume_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=self.root/'.standin-interactive-store-v1'
        if marker.exists():
            if marker.is_symlink() or marker.read_text()!='standin-interactive-store-v1\n':raise ValueError('foreign private volume')
        else:
            if any(p.name!='.service.lock' for p in self.root.iterdir()):raise ValueError('private volume must start empty')
            atomic_write(marker,b'standin-interactive-store-v1\n')
        for old in self.root.iterdir():
            if old.name.startswith(('body-','snapshot-','variant-')) and old.is_dir() and not old.is_symlink():shutil.rmtree(old)
        self.registry=registry;self.runner=runner;self.ttl=ttl;self.capacity=capacity;self.byte_limit=byte_limit
        self.entries={};self.bases={};self.variant_cache={};self.lock=threading.RLock();self.gate=LatestRevisionGate(capacity)

    @staticmethod
    def _disk_bytes(directory):
        size = 0
        for path in directory.rglob('*'):
            try:
                info = path.lstat()
            except FileNotFoundError:
                # A running worker can rename a temporary file during the walk.
                continue
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
        return size

    def _gc(self, preferred_base=None):
        now=time.monotonic()
        latest = {}
        for e in self.entries.values():
            scope = (e['owner'], e['person'])
            latest[scope] = max(latest.get(scope, 0), e['revision'])
        for key,e in list(self.entries.items()):
            scope = (e['owner'], e['person'])
            if not e['busy'] and (now >= e['expiry'] or e['revision'] < latest[scope]):
                shutil.rmtree(e['directory']);del self.entries[key]
                if not any(other['owner']==e['owner'] and other['person']==e['person'] for other in self.entries.values()):
                    self.gate.forget(e['owner'],e['person'])
        referenced_bases = {id(e['base']) for e in self.entries.values() if 'base' in e}
        idle_bases = {key:b for key,b in self.bases.items() if not b['busy'] and id(b) not in referenced_bases}
        for key,b in list(idle_bases.items()):
            if now>=b['expiry']:
                shutil.rmtree(b['directory']);del self.bases[key]
                del idle_bases[key]
        for key,v in list(self.variant_cache.items()):
            if now>=v['expiry']:
                shutil.rmtree(v['directory']);del self.variant_cache[key]
        size = self._disk_bytes(self.root)
        busy = sum(e['busy'] for e in self.entries.values())
        reserve = self.SOLVE_RESERVE_BYTES * (1 + busy)
        entry_limit = len(self.entries) >= self.capacity
        reclaimed = 0
        if not entry_limit and size + reserve > self.byte_limit:
            # Variants are optional suggestions, and an unused body can be rebuilt.
            # Keep live snapshot files and their bases, including active workers.
            eviction_order = [
                *((self.variant_cache, key) for key in self.variant_cache),
                *((self.bases, key) for key in sorted(
                    idle_bases, key=lambda key: (key == preferred_base, idle_bases[key]['expiry'])
                )),
            ]
            for cache, key in eviction_order:
                directory = cache[key]['directory']
                removed = self._disk_bytes(directory)
                shutil.rmtree(directory)
                del cache[key]
                reclaimed += removed
                size -= removed
                if size + reserve <= self.byte_limit:
                    size = self._disk_bytes(self.root)
                    if size + reserve <= self.byte_limit:
                        break
        reason = 'entries' if entry_limit else 'disk' if size + reserve > self.byte_limit else None
        if reason or reclaimed:
            control = current_work.get()
            logging.getLogger('standin.converter').info(json.dumps(dict(
                event='interactive_capacity', result='rejected' if reason else 'reclaimed',
                reason=reason, used_bytes=size, reserved_bytes=reserve, reclaimed_bytes=reclaimed,
                byte_limit=self.byte_limit, entries=len(self.entries), entry_limit=self.capacity,
                busy=busy, request_id=control.request_id if control else None)))
        if reason:
            raise HTTPException(429,'INTERACTIVE_CAPACITY')

    def _get(self, snapshot, owner):
        if not ID.fullmatch(snapshot) or not SHA.fullmatch(owner):raise HTTPException(400,'INVALID_BINDING')
        with self.lock:
            e=self.entries.get(snapshot)
            if not e or not hmac.compare_digest(e['owner'],owner):raise HTTPException(404,'NOT_FOUND')
            if time.monotonic()>=e['expiry']:raise HTTPException(410,'EXPIRED')
            e['checkpoint']()
            return e

    def _child(self, stage, job, identity, base, destination, checkpoint):
        # Reuse the production scheduler and kill-on-deadline/disconnect fence.
        control=current_work.get()
        def check():
            checkpoint()
            if control:control.check('execution')
        # _run_child is dedicated to the final exporter; this worker has its own
        # fixed script, with the same bounded subprocess cancellation behavior.
        import subprocess
        queued=time.monotonic()
        with self.runner._process_slots, tempfile.TemporaryFile() as log:
            check();start=time.monotonic()
            queue_ms=(start-queued)*1000
            p=subprocess.Popen([self.runner.settings.blender_binary,'--background','--factory-startup',
                '--disable-autoexec','--python-exit-code','1','--python',str(Path(__file__).resolve().parents[1]/'converter/interactive_worker.py'),
                '--',stage,str(job),str(identity),str(base),str(destination)],stdout=log,stderr=log)
            try:
                while p.poll() is None:
                    check()
                    if time.monotonic()-start>self.runner.settings.shape_timeout_seconds:raise TimeoutError('interactive deadline')
                    time.sleep(.05)
                check()
                if p.returncode:
                    failure = Path(destination).parent/'hand-contact-error'
                    if stage == 'solve' and failure.is_file():
                        code = failure.read_text()
                        if code == 'HAND_CONTACT_INTERSECTION':
                            raise HTTPException(409, code)
                        if code == 'HAND_CONTACT_UNAVAILABLE':
                            raise HTTPException(503, code)
                    raise RuntimeError('interactive worker failed')
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                execution_ms=(time.monotonic()-start)*1000
                if control:control.execution_ms+=execution_ms
                logging.getLogger('standin.converter').info(json.dumps(dict(
                    event='interactive_blender', stage=stage, queue_wait_ms=round(queue_ms,3),
                    execution_ms=round(execution_ms,3), return_code=p.returncode,
                    request_id=control.request_id if control else None)))

    def _body_metadata(self,directory,binding):
        """An interrupted preparation must never poison the next hand revision."""
        try:
            metadata=decode((directory/'base.json').read_bytes());snapshot=directory/'body.blend'
            if (metadata.get('schema')!='interactive-body-base-v1' or metadata.get('binding')!=binding
                or metadata.get('runtimeRevision')!=runtime_revision() or snapshot.is_symlink()
                or metadata.get('snapshotSha256')!=digest(snapshot.read_bytes())):return None
            return metadata
        except (OSError,ValueError,KeyError,TypeError):return None

    def solve(self, payload):
        if set(payload)!={'selection','operation','revision','person','bvhBase64','expiresAt'}:raise ValueError('invalid request')
        from datetime import datetime
        deadline=datetime.fromisoformat(payload['expiresAt'].replace('Z','+00:00'))
        remaining=deadline.timestamp()-time.time()
        if deadline.tzinfo is None or not 0<remaining<=3660:raise ValueError('invalid operation deadline')
        selection=payload['selection'];operation=payload['operation'];person=payload['person'];revision=payload['revision']
        if not isinstance(selection,dict) or selection.get('schema')!='interactive-selection-v1':raise ValueError('invalid selection')
        if not ID.fullmatch(operation) or not isinstance(person,str) or not ID.fullmatch(person):raise ValueError('invalid operation')
        if type(revision) is not int or not 1<=revision<=2**53-1:raise ValueError('invalid revision')
        owner=selection['ownerBinding']
        if not SHA.fullmatch(owner):raise ValueError('invalid owner')
        raw=base64.b64decode(payload['bvhBase64'],validate=True)
        if not 0<len(raw)<=2*1024*1024 or digest(raw)!=selection['source']['outputBvhSha256']:raise ValueError('source bytes changed')
        character=selection['body'];resolved=self.registry.resolve(character['characterId'])
        if resolved.metadata.sha256!=character['assetSha256']:raise ValueError('character changed')
        recipe=resolve_shape(character['characterId'],resolved.metadata.sha256,selection['shape'],selection['shapeSource'])
        source=selection['source'];refined=source['variant']=='refined'
        if source['variant'] not in ('base','refined') or not SHA.fullmatch(source['originalBvhSha256']):raise ValueError('invalid source variant')
        lineage=source.get('refine',{}).get('lineage') if refined else None
        if refined and (not isinstance(lineage,dict) or not lineage.get('version') or not SHA.fullmatch(lineage.get('configSha256','')) or lineage.get('baseSha256')!=source['originalBvhSha256']):raise ValueError('refine policy lineage missing')
        if not refined and source['outputBvhSha256']!=source['originalBvhSha256']:raise ValueError('base lineage changed')
        if selection['camera']['source_bvh_sha256']!=source['originalBvhSha256']:raise ValueError('camera source changed')
        hands=selection['hands'];wanted=normalize_request({side:hands.get(side) for side in ('left','right')} if hands else None)
        library=load_library()
        if hands and hands.get('library_sha256')!=library.sha256:raise ValueError('hand library changed')
        eye=[float(r==c) for r in range(4) for c in range(4)]
        identity=dict(ownerBinding=owner,selection=dict(operation=operation,revision=revision,person=person,candidate=selection['candidateId']),
            source=dict(originalBvhSha256=source['originalBvhSha256'],refinedBvhSha256=digest(raw) if refined else None,
                        refineVersion=lineage['version']+':'+lineage['configSha256'] if refined else 'none'),
            body=dict(characterId=character['characterId'],characterSha256=resolved.metadata.sha256,rigProfile=resolved.metadata.rig_profile),
            shape=shape_binding(recipe),hands=dict(left=wanted.get('left') if wanted else None,right=wanted.get('right') if wanted else None,
                librarySha256=library.sha256,rulesVersion='hand-bake-v1'),scope=selection['scope'],mirror=selection['mirror'],
            camera=dict(projection='orthographic',viewMatrix=eye,projectionMatrix=eye,distanceTorso=12.,torsoReferenceSha256=resolved.metadata.sha256,baked=False),
            solver=dict(version=SOLVER_VERSION,manifestSha256=SOLVER_MANIFEST_SHA256,adapterVersion=VERSION,runtimeSha256=runtime_revision()))
        # A per-owner/base lock prevents duplicate retarget jobs and cache races.
        binding=body_binding(identity);base_key=digest(canonical(binding));request_hash=digest(canonical(payload))
        with self.lock:
            for sid,e in self.entries.items():
                if e['request']==request_hash:
                    if e['busy']:raise HTTPException(409,'SOLVE_IN_PROGRESS')
                    self._get(sid,owner)
                    return dict(snapshotId=sid,packet=json.loads((e['directory']/'solved/packet.json').read_bytes()))
            self._gc(preferred_base=base_key);revision_checkpoint=self.gate.select_binding(owner,person,revision,request_hash)
            expires=time.monotonic()+min(self.ttl,remaining)
            def checkpoint():
                revision_checkpoint()
                if time.monotonic()>=expires:raise HTTPException(410,'EXPIRED')
            sid=uuid.uuid4().hex;directory=Path(tempfile.mkdtemp(prefix='snapshot-',dir=self.root))
            entry=dict(owner=owner,request=request_hash,directory=directory,checkpoint=checkpoint,expiry=expires,busy=True,revision=revision,person=person)
            self.entries[sid]=entry
            base=self.bases.get(base_key)
            if base is None:
                base=dict(directory=Path(tempfile.mkdtemp(prefix='body-',dir=self.root)),expiry=time.monotonic()+self.ttl,busy=0,lock=threading.Lock())
                self.bases[base_key]=base
            base['busy']+=1
        try:
            job_dir=directory/'job';job_dir.mkdir();(job_dir/'input.bvh').write_bytes(raw)
            job=dict(schema_version=JOB_SCHEMA_VERSION,conversion_id=sid,temp_dir=str(job_dir),bvh_path=str(job_dir/'input.bvh'),
                character_fbx=str(resolved.path),character_id=character['characterId'],character_sha256=resolved.metadata.sha256,
                output_path=str(job_dir/'artifact.fbx'),report_path=str(job_dir/'report.json'),frame=0,mirror=identity['mirror'],output_mode='rigged_rest',
                apply_root_translation=False,embed_textures=False,force_exact_v324=False,thumbnail=None,output_scope='full',
                **({'shape_recipe':recipe} if recipe else {}))
            (job_dir/'job.json').write_text(json.dumps(job));ip=directory/'identity.json';atomic_write(ip,canonical(identity))
            with measured_slot(base['lock']):
                checkpoint()
                base_directory=base['directory']/'base'
                metadata=measured('body_metadata',self._body_metadata,base_directory,binding)
                if metadata is None:
                    if base_directory.exists():shutil.rmtree(base_directory)
                    measured('body_prepare',self._child,'prepare',job_dir/'job.json',ip,base_directory,directory/'unused',checkpoint)
                    metadata=measured('body_metadata_after_prepare',self._body_metadata,base_directory,binding)
                if metadata is None:raise ValueError('body snapshot did not publish completely')
                # This camera is immutable and never affected by hand variant bounds.
                identity['camera']=camera_for_model(selection['camera'],metadata['model'],resolved.metadata.sha256,identity['scope'])
                # Camera participates in latest revision fencing as well as state hash.
                checkpoint()
                ip.unlink();atomic_write(ip,canonical(identity))
                core_key = digest(canonical(core_binding(identity)))
                core = base['directory']/'cores'/core_key
                state = None
                if core.is_dir():
                    try:
                        state = measured('core_rebind',rebind_solved,core, directory/'solved', identity)
                    except (OSError, ValueError):
                        # Private partial/corrupt cache is never display authority.
                        shutil.rmtree(core)
                        shutil.rmtree(directory/'solved', ignore_errors=True)
                        checkpoint()
                if state is None:
                    measured('blender_solve',self._child,'solve',job_dir/'job.json',ip,base_directory,directory/'solved',checkpoint)
                    state,_=measured('validate_before_cache',load_solved,directory/'solved'); checkpoint()
                    core.parent.mkdir(exist_ok=True)
                    shutil.copytree(directory/'solved', core)
                entry['core'] = core
                entry['base'] = base
            checkpoint()
            with self.lock:
                entry['stateHash']=state.state_hash
                entry['packetSha256']=digest((directory/'solved/packet.json').read_bytes())
                entry['runtimeRevision']=runtime_revision()
                entry['busy']=False;base['expiry']=time.monotonic()+self.ttl
            return dict(snapshotId=sid,packet=state.packet())
        except BaseException:
            with self.lock:self.entries.pop(sid,None)
            shutil.rmtree(directory,ignore_errors=True)
            raise
        finally:
            with self.lock:base['busy']-=1

    def read(self,snapshot,owner,kind):
        e=self._get(snapshot,owner)
        # Export writes only its separate artifact directory. Once the immutable
        # model is published, an export must not revoke preview reads.
        if e['busy'] and (kind!='model' or not e.get('stateHash')):
            raise HTTPException(409,'NOT_READY')
        # Publication already fully validated this immutable packet. Rehash its
        # original bytes on every read instead of decoding the entire surface
        # a second time merely to deliver it. BFF independently parses it too.
        if runtime_revision()!=e['runtimeRevision']:raise ValueError('runtime changed')
        if kind=='model':
            raw=(e['directory']/'solved/packet.json').read_bytes()
            if digest(raw)!=e['packetSha256']:raise ValueError('packet changed')
        else:
            from converter.solved_state import decode
            receipt=decode((e['directory']/'export/verified.canonical.json').read_bytes())
            if receipt['stateHash']!=e['stateHash']:raise ValueError('receipt changed')
            raw=(e['directory']/'export/artifact.fbx').read_bytes()
            if digest(raw)!=receipt['artifactSha256']:raise ValueError('artifact changed')
        e['checkpoint']()
        if len(raw)>MAX_RESPONSE:raise ValueError('artifact exceeds budget')
        return raw

    def export(self,snapshot,owner):
        e=self._get(snapshot,owner)
        with self.lock:
            if e['busy']:raise HTTPException(409,'NOT_READY')
            e['busy']=True
        try:
            path=e['directory']/'export'
            if (path/'receipt.json').exists():receipt=json.loads((path/'receipt.json').read_bytes())
            else:
                if path.exists():shutil.rmtree(path)
                control=current_work.get()
                def check():
                    e['checkpoint']()
                    if control:control.check('execution')
                with measured_slot(e['base']['lock']):
                    check()
                    cached = e['core']/'verified-export'
                    receipt = None
                    if cached.is_dir():
                        try:
                            receipt = measured('export_rebind',rebind_export,cached, path, e['directory']/'solved')
                        except (OSError, ValueError):
                            shutil.rmtree(cached)
                            shutil.rmtree(path, ignore_errors=True)
                            check()
                    if receipt is None:
                        with measured_slot(self.runner._process_slots, worker=True):
                            receipt=measured('export_bundle',lambda: export_bundle(self.runner.settings.blender_binary,e['directory']/'solved',path,e['stateHash'],checkpoint=check))
                        check()
                        shutil.copytree(path, cached)
                    check()
            e['checkpoint']()
            with self.lock:
                if receipt['stateHash'] not in self.variant_cache:
                    while len(self.variant_cache)>=16:
                        oldest=next(iter(self.variant_cache));shutil.rmtree(self.variant_cache.pop(oldest)['directory'])
                    cache=Path(tempfile.mkdtemp(prefix='variant-',dir=self.root))
                    atomic_write(cache/'state.json',(e['directory']/'solved/state.canonical.json').read_bytes())
                    atomic_write(cache/'receipt.json',json.dumps(receipt,separators=(',',':')).encode())
                    self.variant_cache[receipt['stateHash']]=dict(directory=cache,owner=owner,expiry=e['expiry'])
            return receipt
        finally:
            with self.lock:e['busy']=False

    def variants(self,snapshot,owner):
        from converter.interactive_variants import asset_binding, build_table
        current=self._get(snapshot,owner)
        state,_=load_solved(current['directory']/'solved')
        binding=asset_binding(state.payload['identity']);pairs=[];keys=set();budget=0
        with self.lock:entries=list(reversed(list(self.variant_cache.values())))
        for entry in entries:
            proof=entry['directory']/'receipt.json'
            if entry['owner']!=owner or time.monotonic()>=entry['expiry']:continue
            try:
                receipt=json.loads(proof.read_bytes())
            except FileNotFoundError:
                continue
            if asset_binding(receipt['identity'])!=binding:continue
            hands=receipt['identity']['hands'];key=(hands['left'],hands['right'])
            if key in keys:continue
            from converter.solved_state import SolvedPoseState
            try:
                candidate=SolvedPoseState((entry['directory']/'state.json').read_bytes())
            except FileNotFoundError:
                continue
            # Keep the whole response bounded, including canonical JSON escaping.
            cost=len(candidate.canonical_bytes)*2
            if budget+cost>60*1024*1024:continue
            pairs.append((candidate,receipt));budget+=cost;keys.add(key)
            if len(pairs)>=8:break
        raw=build_table(pairs,registered_character_sha256={binding['body']['characterSha256']})
        self._get(snapshot,owner)
        return raw

    def cancel(self,owner,person,revision):
        # Never let a delayed cancellation revoke a newer revision.
        with self.lock:
            matches=[e for e in self.entries.values() if e['owner']==owner and e['person']==person and e['revision']==revision]
            for e in matches:
                def cancelled():raise Superseded('cancelled')
                e['checkpoint']=cancelled;e['expiry']=time.monotonic()
            self.gate.cancel_revision(owner,person,revision)


def register_interactive(app,registry,runner):
    if os.getenv('CONVERTER_INTERACTIVE_ENABLED','false').lower()!='true':return
    token=os.getenv('CONVERTER_INTERACTIVE_TOKEN','')
    root=os.getenv('CONVERTER_INTERACTIVE_ROOT','')
    if len(token)<32 or not root:raise ValueError('interactive transport needs private root and service token')
    store=SnapshotStore(root,registry,runner);app.state.interactive=store
    from converter_api.interactive_static import StaticStore
    static=StaticStore(uri=os.getenv('CONVERTER_INTERACTIVE_STATIC_ROOT',''))
    def authorized(request):
        value=request.headers.get('authorization','')
        if not hmac.compare_digest(value,'Bearer '+token):raise HTTPException(401,'UNAUTHORIZED')
    async def body(request):
        parts=[];size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>MAX_REQUEST:raise HTTPException(413,'REQUEST_TOO_LARGE')
            parts.append(chunk)
        try:return json.loads(b''.join(parts))
        except (ValueError,UnicodeError):raise HTTPException(400,'INVALID_INPUT')
    def guarded(function,*args):
        try:return function(*args)
        except HTTPException:raise
        except Superseded:raise HTTPException(409,'SUPERSEDED')
        except (ValueError,KeyError,TypeError):raise HTTPException(409,'INTERACTIVE_INTEGRITY')
        except Exception:raise HTTPException(503,'INTERACTIVE_WORKER_FAILED')
    @app.post('/interactive/v1/solve')
    async def solve(request:Request):
        authorized(request);payload=await body(request)
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(measured,'solve_total',guarded,store.solve,payload)
    @app.post('/interactive/v1/static/resolve')
    async def resolve_static(request:Request):
        authorized(request);payload=await body(request)
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(guarded,static.resolve,payload)
    @app.get('/interactive/v1/static/{sha}')
    def static_file(sha:str,request:Request):
        authorized(request)
        return Response(guarded(static.file,sha),media_type='model/gltf-binary',headers={'Cache-Control':'private, no-store'})
    @app.post('/interactive/v1/cancel')
    async def cancel(request:Request):
        authorized(request);p=await body(request);guarded(store.cancel,p['ownerBinding'],p['person'],p['revision']);return {'cancelled':True}
    @app.get('/interactive/v1/snapshots/{snapshot}/model')
    @app.get('/interactive/v1/snapshots/{snapshot}/file')
    def read(snapshot:str,request:Request):
        authorized(request)
        kind=request.url.path.rsplit('/',1)[-1]
        raw=guarded(store.read,snapshot,request.headers.get('x-standin-owner',''),kind)
        return Response(raw,media_type='application/vnd.standin.solved-pose-state-v1+json' if kind=='model' else 'application/octet-stream',headers={'Cache-Control':'private, no-store'})
    @app.post('/interactive/v1/snapshots/{snapshot}/export')
    def export(snapshot:str,request:Request):
        authorized(request);return measured('export_total',guarded,store.export,snapshot,request.headers.get('x-standin-owner',''))
    @app.get('/interactive/v1/snapshots/{snapshot}/variants')
    def variants(snapshot:str,request:Request):
        authorized(request);raw=guarded(store.variants,snapshot,request.headers.get('x-standin-owner',''))
        return Response(raw,media_type='application/vnd.standin.hand-variants-v1+json',headers={'Cache-Control':'private, no-store','X-Standin-Atlas-SHA256':digest(raw)})
