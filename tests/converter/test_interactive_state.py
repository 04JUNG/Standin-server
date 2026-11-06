import copy
import json
from pathlib import Path
import subprocess

import pytest
from converter.hand_library import load_library
from converter.interactive_adapter import VERSION, body_binding, atomic_write, runtime_revision, shape_binding
from converter.interactive_scene import compare
from converter.interactive_variants import build_table
from converter.protocol import SOLVER_VERSION, SOLVER_MANIFEST_SHA256
from converter.solved_state import SolvedPoseState, canonical, decode, digest, validate_model


def identity():
    eye = [float(r == c) for r in range(4) for c in range(4)]
    return dict(ownerBinding='a'*64, selection=dict(operation='op', revision=1, person='p', candidate='c'),
        source=dict(originalBvhSha256='b'*64, refinedBvhSha256=None, refineVersion='none'),
        body=dict(characterId='registered-fixture', characterSha256='c'*64, rigProfile='mixamo'),
        shape=shape_binding(None),
        hands=dict(left=None, right=None, librarySha256=load_library().sha256, rulesVersion='hand-bake-v1'),
        scope='full', mirror=False,
        camera=dict(projection='orthographic', viewMatrix=eye, projectionMatrix=eye, distanceTorso=4., torsoReferenceSha256='c'*64, baked=False),
        solver=dict(version=SOLVER_VERSION, manifestSha256=SOLVER_MANIFEST_SHA256, adapterVersion=VERSION, runtimeSha256=runtime_revision()))


def model():
    eye = [float(r == c) for r in range(4) for c in range(4)]
    return dict(schema='interactive-pose-model-v1', coordinates=dict(space='right-handed-y-up-metres-hip-origin',
        originBlender=[0., 0., 1.], matrixLayout='row-major', baked='rigged_rest'),
        bones=[dict(name='Hips', parent=-1, restLocal=eye[:], poseLocal=eye[:], inverseBind=eye[:])],
        meshes=[dict(name='mesh_000', positions=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
            normals=[[0.,0.,1.]]*3, normalTriangles=[[0,1,2]], triangles=[[0,1,2]], weights=[[[0,1.]]]*3)], capabilities={'localHands':False})


@pytest.mark.parametrize('variant',['approved','wrong-hash','unknown-id','shaped'])
@pytest.mark.parametrize('raises',[False,True])
def test_terminal_bone_adapter_is_hash_pinned_and_restored(monkeypatch,variant,raises):
    import sys
    from types import SimpleNamespace
    from converter import interactive_adapter as adapter, worker, shape_bake
    legacy=object();full=object();rt=SimpleNamespace(import_character=legacy)
    monkeypatch.setitem(sys.modules,'converter.retarget',rt)
    monkeypatch.setattr(shape_bake,'import_full_character',full)
    cid=next(iter(adapter.TERMINAL_BONE_ASSETS))
    job=dict(character_id=cid,character_sha256=adapter.TERMINAL_BONE_ASSETS[cid])
    if variant=='wrong-hash':job['character_sha256']='f'*64
    if variant=='unknown-id':job['character_id']='unapproved-model'
    if variant=='shaped':job['shape_recipe']={'existing':'shape path'}
    def run(value):
        assert value is job
        assert rt.import_character is (full if variant=='approved' else legacy)
        if raises:raise ValueError('existing gate rejected')
        return {'ok':True}
    monkeypatch.setattr(worker,'run_job',run)
    if raises:
        with pytest.raises(ValueError,match='existing gate rejected'):adapter._run_body_job(job)
    else:assert adapter._run_body_job(job)=={'ok':True}
    assert rt.import_character is legacy


def receipt(state):
    return dict(schema='interactive-fbx-receipt-v1', verified=True, stateHash=state.state_hash,
        identity=state.payload['identity'], packetModelSha256=digest(canonical(state.payload['model'])))


def test_immutable_restart_and_identity():
    i=identity(); m=model(); state=SolvedPoseState.create(i,m)
    i['hands']['left']='fist'; m['meshes'][0]['positions'][0][0]=9
    assert state.payload['identity']['hands']['left'] is None
    assert state.payload['model']['meshes'][0]['positions'][0][0] == 0
    assert SolvedPoseState(state.canonical_bytes).state_hash == state.state_hash
    value=state.payload;value['identity']['selection']['revision']=99
    assert state.payload['identity']['selection']['revision']==1
    with pytest.raises(ValueError): SolvedPoseState(bytearray(state.canonical_bytes))
    assert state.packet()['exportState']=='not-requested'
    assert state.payload['safety']['finalFileVerified'] is False


@pytest.mark.parametrize('field', ['source','body','shape','hands','scope','camera','solver','selection','mirror','ownerBinding'])
def test_lineage_changes_hash(field):
    original=identity(); changed=copy.deepcopy(original)
    if field=='source':changed[field]['refinedBvhSha256']='d'*64
    elif field=='body':changed[field]['characterSha256']='d'*64
    elif field=='shape':changed[field]['sha256']='d'*64
    elif field=='hands':changed[field]['left']='fist'
    elif field=='scope':changed[field]='half'
    elif field=='camera':changed[field]['projection']='perspective'
    elif field=='solver':changed[field]['adapterVersion']='next'
    elif field=='selection':changed[field]['revision']=2
    elif field=='mirror':changed[field]=True
    else:changed[field]='d'*64
    assert SolvedPoseState.create(original,model()).state_hash != SolvedPoseState.create(changed,model()).state_hash


def test_body_reuse_preserves_shape_refine_ownership_and_latest_hands():
    before=identity();after=copy.deepcopy(before);after['hands']['left']='fist';after['selection']['revision']=5
    assert body_binding(before)==body_binding(after)
    for field,key in [('shape','sha256'),('source','refinedBvhSha256'),('body','characterSha256')]:
        changed=copy.deepcopy(after);changed[field][key]='f'*64
        assert body_binding(before)!=body_binding(changed)
    after['ownerBinding']='f'*64
    assert body_binding(before)!=body_binding(after)


@pytest.mark.parametrize('scope',['full','half','bust','shoulder_head','head'])
def test_current_scope_contract(scope):
    i=identity();i['scope']=scope
    SolvedPoseState.create(i,model())


def test_camera_orbit_and_unsafe_surface_fail_closed():
    i=identity();i['camera']['orbit']=1
    with pytest.raises(ValueError):SolvedPoseState.create(i,model())
    for key,value in [('skins',[]),('animations',[]),('extensions',{})]:
        m=model();m[key]=value
        with pytest.raises(ValueError):validate_model(m)
    m=model();m['meshes'][0]['weights'][0]=[]
    with pytest.raises(ValueError):validate_model(m)
    m=model();m['meshes'][0]['positions'][0][0]=float('nan')
    with pytest.raises(ValueError):SolvedPoseState.create(identity(),m)
    m=model();m['bones'][0]['parent']=0
    with pytest.raises(ValueError):validate_model(m)


def test_pose_vertex_skin_correspondence_rejects_drift():
    m=model();assert compare(m,copy.deepcopy(m))['maxPositionErrorM']==0
    for mutate in [lambda m:m['meshes'][0]['positions'][0].__setitem__(0,.001),
                   lambda m:m['bones'][0]['poseLocal'].__setitem__(0,.99),
                   lambda m:m['meshes'][0]['triangles'][0].reverse()]:
        changed=copy.deepcopy(m);mutate(changed)
        with pytest.raises(ValueError):compare(m,changed)


def test_canonical_roundtrip_and_reject_noncanonical():
    values={'z':-0.0,'a':[1.0,1,True,'한글',1e-100]}
    assert canonical(decode(canonical(values)))==canonical(values)
    with pytest.raises(ValueError):decode(b'{"a":1.0}')
    with pytest.raises(ValueError):canonical({'a':2**54})
    with pytest.raises(ValueError):canonical({'$f64':'0000000000000000'})


def test_atomic_artifact_cannot_overwrite(tmp_path):
    path=tmp_path/'state';atomic_write(path,b'a')
    with pytest.raises(ValueError):atomic_write(path,b'b')
    assert path.read_bytes()==b'a'


def test_variants_require_registry_exact_receipt_and_binding(tmp_path):
    state=SolvedPoseState.create(identity(),model()); proof=receipt(state)
    with pytest.raises(ValueError):build_table([(state,proof)],registered_character_sha256=set())
    bad={**proof,'stateHash':'0'*64}
    with pytest.raises(ValueError):build_table([(state,bad)],registered_character_sha256={'c'*64})
    raw=build_table([(state,proof)],registered_character_sha256={'c'*64})
    (tmp_path/'atlas').write_bytes(raw)
    (tmp_path/'state').write_bytes(state.canonical_bytes)
    # Real JS consumer verifies the original Python canonical bytes and all
    # bindings; no mirrored implementation pretending to be a browser test.
    module=(Path(__file__).resolve().parents[2]/'converter/shared/interactive_pose_v1.mjs').as_uri()
    script=f'''import fs from 'node:fs';import assert from 'node:assert/strict';
      import {{readSolvedState,readHandVariants,selectHandVariant}} from {json.dumps(module)};
      const raw=new Uint8Array(fs.readFileSync(process.argv[1]));
      const s=await readSolvedState(raw,process.argv[2]);
      const atlas=await readHandVariants(new Uint8Array(fs.readFileSync(process.argv[3])),process.argv[4]);
      const v=selectHandVariant(atlas,s.identity,{{left:null,right:null}});
      assert.equal(v.previewState,'local-draft');assert.equal(v.model.meshes[0].positions[1][0],1);
      assert.equal(selectHandVariant(atlas,s.identity,{{left:'fist',right:null}}),null);
      const changed=structuredClone(s.identity);changed.shape.sha256='d'.repeat(64);
      assert.throws(()=>selectHandVariant(atlas,changed,{{}}),/stale/);
      await assert.rejects(readSolvedState(raw,'0'.repeat(64)),/hash/);
      assert(Object.isFrozen(s.model.meshes[0].positions));'''
    subprocess.run(['node','--input-type=module','-e',script,str(tmp_path/'state'),state.state_hash,str(tmp_path/'atlas'),digest(raw)],check=True)


def test_late_revision_and_exception_never_publish_ready(tmp_path,monkeypatch):
    import converter.interactive_export as exporter
    state=SolvedPoseState.create(identity(),model())
    monkeypatch.setattr('converter.interactive_adapter.load_solved',lambda _: (state,None))
    def child(binary,args,checkpoint,timeout):
        directory=Path(args[args.index('--destination')+1])
        (directory/'artifact.fbx').write_bytes(b'fixture-only')
        proof={**receipt(state),'revision':1,'artifactSha256':digest(b'fixture-only')}
        (directory/'verified.canonical.json').write_bytes(canonical(proof))
    monkeypatch.setattr(exporter,'_run_child',child)
    count=0
    def checkpoint():
        nonlocal count
        count+=1
        if count==2:raise exporter.Superseded('late hand/refine change')
    with pytest.raises(exporter.Superseded):
        exporter.export_bundle('binary','bundle',tmp_path/'stale',state.state_hash,checkpoint=checkpoint)
    assert not (tmp_path/'stale/receipt.json').exists()
    def crash(*args):raise RuntimeError('export failed')
    monkeypatch.setattr(exporter,'_run_child',crash)
    with pytest.raises(RuntimeError):exporter.export_bundle('binary','bundle',tmp_path/'failed',state.state_hash,checkpoint=lambda:None)
    assert not (tmp_path/'failed/receipt.json').exists()


def test_rapid_hands_late_refine_cancel_and_restart():
    from converter.interactive_revision import LatestRevisionGate
    from converter.interactive_export import Superseded
    gate=LatestRevisionGate(max_people=1); i=identity(); checkpoints=[]
    for revision in range(1,50):
        i['selection']['revision']=revision;i['hands']['left']='fist' if revision%2 else 'open'
        checkpoints.append(gate.select(i))
    checkpoints[-1]()
    for old in checkpoints[:-1]:
        with pytest.raises(Superseded):old()
    late=copy.deepcopy(i);late['source']['refinedBvhSha256']='e'*64
    with pytest.raises(Superseded):gate.select(late) # same rev cannot restore stale hands
    gate.cancel(i['ownerBinding'],i['selection']['person'])
    with pytest.raises(Superseded):checkpoints[-1]()
    with pytest.raises(Superseded):gate.select(i)
    i['selection']['revision']+=1;current=gate.select(i);current()
    gate.restart()
    with pytest.raises(Superseded):current()


def test_snapshot_hash_tamper_and_state_swap_rejected_on_restart(tmp_path):
    from converter.interactive_adapter import load_solved
    state=SolvedPoseState.create(identity(),model())
    (tmp_path/'state.canonical.json').write_bytes(state.canonical_bytes)
    (tmp_path/'solved.blend').write_bytes(b'local-private-snapshot-fixture')
    (tmp_path/'staged.fbx').write_bytes(b'immutable-private-file')
    metadata=dict(stagedFbxSha256=digest(b'immutable-private-file'),schema='interactive-snapshot-v1',stateHash=state.state_hash,
        runtimeRevision=runtime_revision(),snapshotSha256=digest(b'local-private-snapshot-fixture'))
    (tmp_path/'snapshot.json').write_bytes(canonical(metadata))
    assert load_solved(tmp_path)[0].state_hash==state.state_hash
    (tmp_path/'solved.blend').write_bytes(b'late-mutation')
    with pytest.raises(ValueError):load_solved(tmp_path)
    (tmp_path/'solved.blend').write_bytes(b'local-private-snapshot-fixture')
    changed=identity();changed['hands']['left']='fist'
    (tmp_path/'state.canonical.json').write_bytes(SolvedPoseState.create(changed,model()).canonical_bytes)
    with pytest.raises(ValueError):load_solved(tmp_path)


def test_cancel_running_export_terminates_child(monkeypatch):
    import converter.interactive_export as exporter
    class Process:
        returncode=None
        terminated=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self,timeout=None):return self.returncode
    process=Process()
    monkeypatch.setattr(exporter.subprocess,'Popen',lambda *args,**kw:process)
    def cancel():raise exporter.Superseded('new hand revision')
    with pytest.raises(exporter.Superseded):exporter._run_child('fixture-binary',[],cancel,10)
    assert process.terminated


def test_old_cancel_cannot_revoke_new_selection():
    from converter.interactive_revision import LatestRevisionGate
    from converter.interactive_export import Superseded
    gate=LatestRevisionGate();first=identity();old=gate.select(first)
    second=copy.deepcopy(first);second['selection']['revision']=2;current=gate.select(second)
    gate.cancel_revision(first['ownerBinding'],'p',1);current()
    with pytest.raises(Superseded):old()
    gate.cancel_revision(first['ownerBinding'],'p',2)
    with pytest.raises(Superseded):current()
