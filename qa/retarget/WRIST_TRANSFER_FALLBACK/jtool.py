"""Convert BVHs with a chosen converter tree and measure wrists + self-collision on the real character.
usage: CONV_ROOT=/path/to/server SCAN_CHAR=... python jtool.py OUTDIR LIST   (LIST lines: key<space>bvh_path)
"""
import sys, os, json, math, tempfile, hashlib
from pathlib import Path
CONV = os.environ.get('CONV_ROOT', '/home/claude/mesh/server')
H = Path(os.environ.get('MESH_ROOT', '/home/claude/mesh'))
sys.path[:0] = [str(H / 's3'), CONV, str(H / 'complement'), os.environ.get('W_ROOT', '/home/claude/w')]
import bpy, numpy as np
from converter import retarget
from converter.convert import convert
from pose_curation.rendering.fingers import rest_bases, apply_channels
from self_collision import inspect_self
from src.bvh import parse_bvh, channel_starts
from scipy.spatial.transform import Rotation as Rot
CHAR = H / 'assets' / os.environ.get('SCAN_CHAR', 'standin-master-v2.fbx')
SEAM = {frozenset(p) for p in [("LeftForearm","LeftUpperArm"),("RightForearm","RightUpperArm"),("LeftShin","LeftThigh"),("RightShin","RightThigh"),("Head","Torso"),("LeftUpperArm","Torso"),("RightUpperArm","Torso"),("LeftThigh","Torso"),("RightThigh","Torso")]}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
bases = rest_bases(str(CHAR))
retarget.reset_scene(); tmpl, _ = retarget.import_character(str(CHAR))
tb = {b.name.rsplit(':', 1)[-1]: b for b in tmpl.pose.bones}
def _restpack(s):
    W = tmpl.matrix_world.to_3x3()
    hd = lambda n: tmpl.matrix_world @ tb[s + n].head
    return dict(F0=(W @ tb[s + 'ForeArm'].matrix.to_3x3()).normalized(), H0=(W @ tb[s + 'Hand'].matrix.to_3x3()).normalized(),
                hand=hd('Hand'), mid=hd('HandMiddle1'), idx=hd('HandIndex1'), pk=hd('HandPinky1'))
RP = {s: _restpack(s) for s in ('Left', 'Right')}
REST = {s: tb[s + 'ForeArm'].matrix.to_3x3().normalized().inverted() @ tb[s + 'Hand'].matrix.to_3x3().normalized() for s in ('Left', 'Right')}

def src_wrist(path):
    joints, frames = parse_bvh(path); st = channel_starts(joints)
    def W(frame):
        Wm, ci = {}, 0
        for i, (n, p, off, ch, e) in enumerate(joints):
            order = ''; vals = {}
            for c in ch:
                v = frame[ci]; ci += 1
                if c.endswith('rotation'): order += c[0]; vals[c[0]] = v
            R = Rot.from_euler(order, [vals[a] for a in order], degrees=True).as_matrix() if order else np.eye(3)
            Wm[i] = R if p == -1 else Wm[p] @ R
        return Wm
    Wp, W0 = W(frames[0]), W(np.zeros_like(frames[0]))
    res = {}
    for s in ('Left', 'Right'):
        f = next((i for i, j in enumerate(joints) if j[0].split(':')[-1] == s + 'ForeArm'), None)
        h = next((i for i, j in enumerate(joints) if j[0].split(':')[-1] == s + 'Hand'), None)
        if f is None or h is None: continue
        rel_p = Wp[f].T @ Wp[h]; rel_0 = W0[f].T @ W0[h]
        d = Rot.from_matrix(rel_0.T @ rel_p)
        res[s] = round(math.degrees(d.magnitude()), 2)
    return res

def ang(a, b):
    return round(math.degrees(a.angle(b)), 2) if a.length > 1e-8 and b.length > 1e-8 else None

tmp = tempfile.mkdtemp()

def wrist_anat(arm, bones, s):
    """전완 local 기준 손목 신전(+)/굴곡(-), 요측(+)/척측(-) 편위, 전완축 둘레 비틀림 — 캐릭터 rest 대비(deg).
    dorsal 기준 = 캐릭터 rest에서 world +Z(손바닥이 아래인 T포즈의 손등)."""
    from mathutils import Vector
    R = RP[s]; W = arm.matrix_world.to_3x3()
    F = (W @ bones[s + 'ForeArm'].matrix.to_3x3()).normalized(); H = (W @ bones[s + 'Hand'].matrix.to_3x3()).normalized()
    F0, H0 = R['F0'], R['H0']
    hd = lambda n: arm.matrix_world @ bones[s + n].head
    h = F.inverted() @ (hd('HandMiddle1') - hd('Hand')); h0 = F0.inverted() @ (R['mid'] - R['hand'])
    f0 = Vector((0, 1, 0)); n0 = F0.inverted() @ Vector((0, 0, 1)); n0 = (n0 - f0 * n0.dot(f0)).normalized(); l0 = f0.cross(n0)
    if l0.dot(F0.inverted() @ (R['idx'] - R['pk'])) < 0: l0 = -l0
    A = lambda v, ax: math.degrees(math.atan2(v.dot(ax), v.dot(f0)))
    q = ((F.inverted() @ H) @ (F0.inverted() @ H0).inverted()).to_quaternion()
    tw = q.to_swing_twist('Y')[1]
    return dict(ext_deg=round(A(h, n0) - A(h0, n0), 1), radial_dev_deg=round(A(h, l0) - A(h0, l0), 1), twist_deg=round(math.degrees(tw), 1),
                delta_q=[round(x, 6) for x in q], h0=[round(x, 5) for x in h0], n0=[round(x, 5) for x in n0], l0=[round(x, 5) for x in l0])

for line in open(sys.argv[2]):
    key, bvh = line.rstrip('\n').split(' ', 1)
    if (out / f'{key}.json').exists(): continue
    rec = {'key': key, 'bvh': bvh, 'bvh_sha256': hashlib.sha256(open(bvh, 'rb').read()).hexdigest(), 'conv_root': CONV}
    try:
        retarget.reset_scene()
        art = Path(tmp) / f'{key}.fbx'
        rep = convert(bvh_path=bvh, character_fbx=str(CHAR), out_path=str(art), src_profile=None, dst_profile='mixamo', embed_textures=False)
        rec['fbx_sha256'] = hashlib.sha256(open(art, 'rb').read()).hexdigest()
        for f in ('palm_roll_mode', 'palm_fallback_reason', 'terminal_follow_bones', 'palm_roll_applied_deg'):
            rec[f] = getattr(rep, f, None) if not isinstance(getattr(rep, f, None), (list, dict)) else json.loads(json.dumps(getattr(rep, f), default=str))
        rec['wrist_transfer'] = json.loads(json.dumps(getattr(rep, 'wrist_transfer', {}), default=str))
        retarget.reset_scene()
        arm, _ = retarget.import_character(str(art))
        try: apply_channels(arm, bvh, bases); rec['fingers'] = 'applied'
        except ValueError as e: rec['fingers'] = f'not applied: {e}'
        bpy.context.view_layer.update()
        bones = {b.name.rsplit(':', 1)[-1]: b for b in arm.pose.bones}
        m = {}
        for s in ('Left', 'Right'):
            sh, el, wr, mid, idx, pk = [arm.matrix_world @ bones[s + n].head for n in ('Arm', 'ForeArm', 'Hand', 'HandMiddle1', 'HandIndex1', 'HandPinky1')]
            rel = bones[s + 'ForeArm'].matrix.to_3x3().normalized().inverted() @ bones[s + 'Hand'].matrix.to_3x3().normalized()
            dq = (rel @ REST[s].inverted()).to_quaternion(); dd = math.degrees(dq.angle); dd = min(dd, 360 - dd)
            m[s] = dict(target_hand_rel_from_char_rest_deg=round(dd, 2), wrist_dir_change_deg=ang(wr - el, mid - wr),
                        elbow_bend_deg=ang(el - sh, wr - el), hand_pos=[round(v * 1000) for v in wr], mid_pos=[round(v * 1000) for v in mid])
            m[s].update(wrist_anat(arm, bones, s))
        rec['target'] = m
        rec['source_wrist_rel_deg'] = src_wrist(bvh)
        wb = inspect_self(arm)
        rec['cols'] = [{'r': c['regions'], 'd': c['max_depth_mm'], 'w': c.get('witness_world')} for c in wb['collisions'] if frozenset(c['regions']) not in SEAM]
        g = bpy.context.evaluated_depsgraph_get()
        obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
        me = obj.evaluated_get(g).to_mesh()
        P = np.array([obj.matrix_world @ v.co for v in me.vertices])
        if not (out / 'tris.bin').exists():
            me.calc_loop_triangles(); np.array([t.vertices[:] for t in me.loop_triangles], dtype=np.uint16).tofile(out / 'tris.bin')
        obj.evaluated_get(g).to_mesh_clear()
        np.round(P * 1000).astype(np.int16).tofile(out / f'{key}.bin')
        rec['status'] = 'ok'
    except Exception as e:
        import traceback; rec['status'] = 'error'; rec['error'] = f'{type(e).__name__}: {e}'; rec['tb'] = traceback.format_exc()[-800:]
    json.dump(rec, open(out / f'{key}.json', 'w'), ensure_ascii=False, default=str)
    print('DONE', key, rec['status'], flush=True)
