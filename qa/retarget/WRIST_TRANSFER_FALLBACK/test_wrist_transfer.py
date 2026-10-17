"""palm-roll fallback 손목 전달 재현·회귀 테스트 (Blender 5.2 bpy 필요).

실행: PY=<bpy 5.2 python> MESH_ROOT=... W_ROOT=... OLD_CONV=<수정 전 server> NEW_CONV=<수정 후 server> python -m pytest -q test_wrist_transfer.py
- 재현: 수정 전 변환기에서 cmu_135_01_04368(+미러)의 손이 전완과 rest 관계(0°)로 고정된다.
- 수정: 수정 후에는 원본 손목 상대회전(L 67.7° / R 12.0°)이 전달되고(전달량 = 원본 채널), 미러는 좌우 대칭.
- 끄기: STANDIN_WRIST_TRANSFER=0 이면 수정 전과 같은 결과.
- 회귀: 손바닥 기준점이 있는 프로필(mixamo, 생성 포즈 등)은 수정 전후 손 회전·관통 값이 동일.
"""
import json, os, subprocess, tempfile
from pathlib import Path
import pytest

HERE = Path(__file__).parent; FIX = HERE / 'fixtures'
PY = os.environ.get('PY', 'python'); OLD = os.environ.get('OLD_CONV'); NEW = os.environ.get('NEW_CONV')
FALLBACK = ['cmu_135_01_04368', 'cmu_135_01_04368_mirror']
CONTROLS = ['Baseball Hit_00033', 'Baseball Hit_00033_mirror', 'Baseball Hit_00054_mirror', 'Female Laying Pose_01',
            'gen_ac_brace__mg_orb_overhead__x__h0', 'gen_ac_fight_side__mg_wrist_hold__x__h1',
            'gen_ac_jump_split__mg_beam__x-j0__h5', 'gen_fl_one_knee__bag_strap__knee__h0', 'gen_sq_deep__selfie__forearm__h1']

def run(conv, names, extra_env=None):
    out = Path(tempfile.mkdtemp()); lst = out / 'list.txt'
    lst.write_text(''.join(f'k{i:02d} {FIX / (n + ".bvh")}\n' for i, n in enumerate(names)))
    env = dict(os.environ, CONV_ROOT=conv, **(extra_env or {}))
    subprocess.run([PY, str(HERE / 'jtool.py'), str(out), str(lst)], env=env, check=True, capture_output=True,
                   cwd=os.environ.get('MESH_ROOT', '.'))
    return {n: json.load(open(out / f'k{i:02d}.json')) for i, n in enumerate(names)}

@pytest.fixture(scope='module')
def res():
    if not (OLD and NEW): pytest.skip('OLD_CONV/NEW_CONV 미지정')
    names = FALLBACK + CONTROLS
    return dict(old=run(OLD, names), new=run(NEW, names), off=run(NEW, FALLBACK, {'STANDIN_WRIST_TRANSFER': '0'}))

def rel(d, s): return d['target'][s]['target_hand_rel_from_char_rest_deg']

def test_repro_fallback_drops_wrist(res):
    for n in FALLBACK:
        d = res['old'][n]
        assert set(d['palm_roll_mode'].values()) == {'fallback'}
        assert rel(d, 'Left') < 0.5 and rel(d, 'Right') < 0.5
        assert max(d['source_wrist_rel_deg'].values()) > 10

def test_fix_transfers_source_wrist(res):
    for n in FALLBACK:
        d = res['new'][n]; src = d['source_wrist_rel_deg']
        assert abs(d['wrist_transfer']['hand.L'] - src['Left']) < 0.1
        assert abs(d['wrist_transfer']['hand.R'] - src['Right']) < 0.1
        # 비틀림 절반은 전완이 나눠 가지므로 손-전완 상대각은 원본보다 작거나 같다
        for s in ('Left', 'Right'):
            assert rel(d, s) <= src[s] + 0.1

def test_mirror_symmetry(res):
    a, b = res['new'][FALLBACK[0]], res['new'][FALLBACK[1]]
    assert abs(rel(a, 'Left') - rel(b, 'Right')) < 0.5 and abs(rel(a, 'Right') - rel(b, 'Left')) < 0.5

def test_switch_off_reproduces_old(res):
    for n in FALLBACK:
        assert abs(rel(res['off'][n], 'Left') - rel(res['old'][n], 'Left')) < 0.01
        assert abs(rel(res['off'][n], 'Right') - rel(res['old'][n], 'Right')) < 0.01

def test_controls_unchanged(res):
    for n in CONTROLS:
        o, w = res['old'][n], res['new'][n]
        assert not w['wrist_transfer']
        for s in ('Left', 'Right'):
            assert abs(rel(o, s) - rel(w, s)) < 0.01
        assert abs(max([c['d'] for c in o['cols']], default=0) - max([c['d'] for c in w['cols']], default=0)) < 0.5
