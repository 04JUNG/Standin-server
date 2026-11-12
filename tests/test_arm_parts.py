"""Arm-part assets: real serialization, different rigs, preservation and rollback."""
import copy
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('scipy')
from src.arm_refine.parts import extract_part, standalone_bvh, read_part, write_part, apply_part, search_parts
from src.bvh import parse_bvh, fk, channel_starts
from src.experimental.part_composition import world_transforms, _set_rotation
from src.bvh import _rot


def source_rig(tmp_path, *, scale=1., order='ZYX'):
    specs=[('Hips',None,(0,0,0)),('Spine','Hips',(0,.6,0)),('Head','Spine',(0,.8,0)),
        ('LeftShoulder','Spine',(-.1,.4,0)),('LeftArm','LeftShoulder',(-.25,0,0)),
        ('LeftForeArm','LeftArm',(-.5,0,0)),('LeftHand','LeftForeArm',(-.5,0,0)),
        ('LeftHandMiddle1','LeftHand',(-.1,0,0)),
        ('RightShoulder','Spine',(.1,.4,0)),('RightArm','RightShoulder',(.25,0,0)),
        ('RightForeArm','RightArm',(.5,0,0)),('RightHand','RightForeArm',(.5,0,0)),
        ('RightHandMiddle1','RightHand',(.1,0,0)),
        ('LeftUpLeg','Hips',(-.2,0,0)),('LeftLeg','LeftUpLeg',(0,-.5,0)),('LeftFoot','LeftLeg',(0,-.5,.1)),
        ('RightUpLeg','Hips',(.2,0,0)),('RightLeg','RightUpLeg',(0,-.5,0)),('RightFoot','RightLeg',(0,-.5,.1))]
    lines=['HIERARCHY'];values=[]
    def emit(name,depth=0):
        i=next(i for i,s in enumerate(specs) if s[0]==name)
        _,parent,off=specs[i];ch=(['Xposition','Yposition','Zposition'] if parent is None else [])+[a+'rotation' for a in order]
        lines.extend(['  '*depth+('ROOT ' if parent is None else 'JOINT ')+name,'  '*depth+'{',
            '  '*depth+' OFFSET '+' '.join(str(scale*x) for x in off),'  '*depth+f' CHANNELS {len(ch)} '+' '.join(ch)])
        values.extend([0.]*len(ch))
        for child,p,_ in specs:
            if p==name:emit(child,depth+1)
        lines.append('  '*depth+'}')
    emit('Hips');lines.extend(['MOTION','Frames: 1','Frame Time: 0.033333',' '.join(map(str,values))])
    path=tmp_path/(str(scale)+order+'.bvh');path.write_text('\n'.join(lines)+'\n')
    return path


def test_extract_has_real_chain_bind_and_no_fingers(tmp_path):
    path=source_rig(tmp_path)
    part=extract_part(path,'left')
    assert len(part['arms']['left']['chain'])==4
    assert part['arms']['left']['chain'][0]['name']=='LeftShoulder'
    assert part['side_convention']=='subject_relative'
    assert part['apply_policy']['fingers']=='preserve_destination_local_pose'
    assert not part['release_allowed']


@pytest.mark.parametrize('selection',['left','right','both'])
@pytest.mark.parametrize('order',['XYZ','ZXY','ZYX'])
def test_part_bvh_json_roundtrip(tmp_path,selection,order):
    source=source_rig(tmp_path,order=order)
    p=extract_part(source,selection)
    out=tmp_path/'part.json';write_part(p,out);p=read_part(out)
    bvh=tmp_path/'part.bvh';bvh.write_text(standalone_bvh(p))
    j,f=parse_bvh(str(bvh));pos=fk(j,f[0]);names={x[0]:i for i,x in enumerate(j)}
    assert all('Finger' not in n and 'Spine' not in n for n in names)
    for side in p['arms']:
        arm=p['arms'][side]
        for n,k in [('Shoulder','anchor_position_torso'),('Arm','shoulder_position_torso'),('ForeArm','elbow_position_torso'),('Hand','wrist_position_torso')]:
            np.testing.assert_allclose(pos[names[side.title()+n]],arm[k],atol=1e-8)


@pytest.mark.parametrize('selection',['left','right','both'])
def test_different_rig_units_parent_rotation_and_fingers_preserved(tmp_path,selection):
    p=extract_part(source_rig(tmp_path),'both')
    j,f=parse_bvh(str(source_rig(tmp_path,scale=100,order='XYZ')));base=f[0]
    names={r[0]:i for i,r in enumerate(j)}
    _set_rotation(j,base,names['Spine'],_rot('X',17)@_rot('Y',25))
    for side in ['Left','Right']:
        _set_rotation(j,base,names[side+'Arm'],_rot('Z',30))
        _set_rotation(j,base,names[side+'HandMiddle1'],_rot('X',40))
    original=base.copy();before=world_transforms(j,base)
    out,report=apply_part(j,base,p,selection=selection)
    assert report['accepted'],report
    assert report['protected_channels_exact']
    assert report['protected_position_max_delta']==0
    after=world_transforms(j,out)
    for side in ['Left','Right']:
        i=names[side+'Hand']
        np.testing.assert_allclose(after[i][:3,:3],before[i][:3,:3],atol=1e-12)
    np.testing.assert_array_equal(base,original)
    starts=channel_starts(j)
    for side in ['Left','Right']:
        i=names[side+'HandMiddle1'];np.testing.assert_array_equal(out[starts[i]:starts[i]+3],base[starts[i]:starts[i]+3])


def test_pair_requires_atomic_both(tmp_path):
    p=extract_part(source_rig(tmp_path),'both',relation='paired')
    j,f=parse_bvh(str(source_rig(tmp_path)))
    with pytest.raises(ValueError,match='atomically'):apply_part(j,f[0],p,selection='left')


def test_gate_rejection_returns_exact_base(tmp_path,monkeypatch):
    from src.collision import CollisionMeasure
    import src.arm_refine.parts as module
    p=extract_part(source_rig(tmp_path),'both')
    j,f=parse_bvh(str(source_rig(tmp_path)))
    calls=iter([CollisionMeasure(True,depth=0),CollisionMeasure(True,depth=.2)]*2)
    monkeypatch.setattr(module,'arm_torso_penetration',lambda *a:next(calls))
    out,report=apply_part(j,f[0],p)
    assert not report['accepted']
    np.testing.assert_array_equal(out,f[0])


def test_search_self_match_and_does_not_mutate_query(tmp_path):
    p=extract_part(source_rig(tmp_path),'both');query={s:np.array(a['directions_torso'])*2 for s,a in p['arms'].items()}
    original=copy.deepcopy(query)
    hits=search_parts([p],query,'both')
    assert hits[0]['mean_direction_error_deg']<1e-6
    for side in query:np.testing.assert_array_equal(query[side],original[side])


def test_bad_side_directions_and_topology_rejected(tmp_path):
    source=source_rig(tmp_path)
    with pytest.raises(ValueError):extract_part(source,'screen_left')
    p=extract_part(source,'left');p['arms']['left']['directions_torso'][0]=[0,0,0]
    with pytest.raises(ValueError,match='directions'):write_part(p,tmp_path/'bad.json')
    source.write_text(source.read_text().replace('LeftForeArm','UnknownElbow'))
    with pytest.raises(ValueError,match='unsupported rig'):extract_part(source,'left')


def test_folded_contact_hold_rejected_before_apply(tmp_path):
    p=extract_part(source_rig(tmp_path),'both',relation='paired')
    p['semantics']={'action':'fold_arms','pair_quality':{'eligible_strict_fold':False}}
    j,f=parse_bvh(str(source_rig(tmp_path)))
    with pytest.raises(ValueError,match='contact quality is on hold'):
        apply_part(j,f[0],p)
