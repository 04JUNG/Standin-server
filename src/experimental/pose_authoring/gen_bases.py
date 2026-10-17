"""Lower-body / trunk bases for the combinatorial generator.

Each base: group (decides which upper-body actions and free-arm fillers fit),
spec (root, legs, props, support, base trunk), and search-v2 tag pieces.
"""
from .presets import merge, stand, chair, desk, leg, wall_behind, SEAT_H, DESK_H

B = {}


def base(bid, group, spec, posture, desc, label, actions="", objects="", domain="", activity=""):
    B[bid] = {"group": group, "spec": spec, "posture": posture, "desc": desc, "label": label,
              "actions": actions, "objects": objects, "domain": domain, "activity": activity}


# ------------------------------------------------------------------ standing
base("st_neutral", "stand", stand(), "standing", "똑바로 서서", "서서")
base("st_wide", "stand", stand(width=15, toe_out=12), "standing", "다리를 어깨너비보다 넓게 벌리고 서서",
     "다리 벌리고 서서")
base("st_narrow", "stand", stand(width=6.5, toe_out=4), "standing", "두 발을 모으고 서서", "발 모으고 서서")
base("st_lead", "stand", stand(lead=9), "standing", "한 발을 살짝 앞에 두고 서서", "한 발 앞에 두고")
base("st_contra", "stand", merge(stand(width=9, lead=5), {"root": {"roll": 4}, "spine": {"roll": -3}}),
     "standing", "한쪽 다리에 체중을 싣고 짝다리로 서서", "짝다리로 서서")
base("st_walk", "stand", stand(lead=16, y=89.5), "standing", "걸어가면서", "걸으며", actions="walk")
base("st_stride", "stand", merge(stand(lead=21, y=87.5), {"root": {"pitch": 4}, "spine": {"pitch": 3}}),
     "standing", "성큼성큼 큰 걸음으로 걸으며", "큰 걸음으로", actions="walk,step")
base("st_back", "stand", merge(stand(lead=-12, y=88.5), {"root": {"pitch": -3}}), "standing",
     "한 발 뒤로 물러서며", "뒷걸음치며", actions="retreat,step")
base("st_lean_fwd", "stand", merge(stand(width=10), {"root": {"pitch": 8, "z": -3}, "spine": {"pitch": 12}}),
     "standing", "상체를 앞으로 숙이고 서서", "앞으로 숙이고", actions="lean,bend")
base("st_lean_back", "stand", merge(stand(width=10, lead=-3), {"spine": {"pitch": -8}}), "standing",
     "상체를 살짝 뒤로 젖히고 서서", "뒤로 젖히고", actions="lean_back")
base("st_bent_knees", "stand", merge(stand(width=14, y=85, toe_out=12), {"root": {"pitch": 10}, "spine": {"pitch": 4}}),
     "standing", "무릎을 굽혀 자세를 낮추고 서서", "자세 낮추고", actions="prepare")
base("st_lunge", "stand", merge(stand(width=12, lead=26, y=80), {"root": {"pitch": 6}}), "standing",
     "한 발을 크게 앞으로 내디뎌 런지하듯", "크게 내디디며", actions="step,dash")
base("st_twist", "stand", merge(stand(width=10, lead=4), {"spine": {"yaw": 22}}), "standing",
     "허리를 옆으로 틀고 서서", "허리 틀고", actions="turn")
base("st_wall", "stand", merge(stand(width=10, y=89.5), wall_behind(z=-17.5), {
    "root": {"z": 0, "pitch": -7}, "spine": {"pitch": -2},
    "legs": {"L": leg(10, 10.4, 22, [0.1, 0, 1]), "R": leg(-10, 10.4, 17, [-0.1, 0, 1])}}),
     "standing", "벽에 등을 기대고 서서", "벽에 기대", actions="lean", objects="wall")
base("st_stairs", "stand", merge({
    "root": {"y": 94, "pitch": 8}, "spine": {"pitch": 4},
    "props": [{"kind": "step", "min": [-40, 0, 10], "max": [40, 18, 40]}],
    "legs": {"L": leg(10, 28.4, 26, [0.1, 0.2, 1]),
             "R": leg(-10, 12.4, -16, [-0.1, 0, 1], pitch=20, toe=30, planted=False)},
    "support": "floor"}), "standing", "계단을 한 칸 오르며", "계단 오르며", actions="climb,step")
base("st_tiptoe", "stand", merge({
    "root": {"y": 94.5},
    "legs": {"L": leg(8, 15, -1, [0.1, 0, 1], pitch=0, toe=50, planted=False),
             "R": leg(-8, 15, -1, [-0.1, 0, 1], pitch=0, toe=50, planted=False)}}),
     "standing", "발뒤꿈치를 들고 까치발로 서서", "까치발로", actions="balance")

# ------------------------------------------------------------------ chair
base("ch_normal", "chair", chair(), "sitting", "의자에 바르게 앉아", "의자에 앉아", actions="sit", objects="chair")
base("ch_fwd", "chair", merge(chair(), {"spine": {"pitch": 16}}), "sitting", "의자에 앉아 상체를 앞으로 숙이고",
     "의자에 앉아 숙이고", actions="sit,lean", objects="chair")
base("ch_wide", "chair", chair(feet_spread=19, feet_z=44), "sitting", "의자에 다리를 넓게 벌리고 앉아",
     "다리 벌리고 앉아", actions="sit", objects="chair")
base("ch_feet_out", "chair", merge(chair(feet_z=60), {"root": {"pitch": -6}}), "sitting",
     "의자에 다리를 앞으로 쭉 뻗고 앉아", "다리 뻗고 앉아", actions="sit", objects="chair")
base("ch_feet_tuck", "chair", chair(feet_z=26, feet_spread=9), "sitting", "의자에 발을 뒤로 당겨 모으고 앉아",
     "발 모으고 앉아", actions="sit", objects="chair")
base("ch_desk", "desk", merge(chair(), desk(z0=26)), "sitting", "책상 앞 의자에 앉아", "책상에 앉아",
     actions="sit", objects="chair,desk", activity="study")
base("ch_desk_fwd", "desk", merge(chair(), desk(z0=26), {"spine": {"pitch": 14}}), "sitting",
     "책상 앞에 앉아 몸을 앞으로 기울이고", "책상에 숙여 앉아", actions="sit,lean", objects="chair,desk",
     activity="study")
base("ch_slouch", "chair", {
    "root": {"y": SEAT_H + 8.0, "z": 4, "pitch": -24}, "spine": {"pitch": 4},
    "legs": {"L": leg(14, 9.0, 72, [0.2, 0.4, 1], planted=False, mode="flat", pitch=-25),
             "R": leg(-12, 9.0, 66, [-0.2, 0.4, 1], planted=False, mode="flat", pitch=-25)},
    "props": [{"kind": "seat", "min": [-23, 0, -26], "max": [23, SEAT_H, 12], "snap": "y+"},
              {"kind": "backrest", "min": [-23, 30, -30], "max": [23, SEAT_H + 50, -26], "wall": True,
               "snap": "z-"}],
    "support": "seat"}, "sitting", "의자 등받이에 깊게 기대 늘어지게 앉아", "의자에 늘어져",
     actions="sit,lean_back", objects="chair", activity="rest")
base("ch_bench_low", "chair", merge(chair(seat=38, feet_z=36, feet_spread=15), {"spine": {"pitch": 8}}), "sitting",
     "낮은 벤치에 걸터앉아", "벤치에 걸터앉아", actions="sit", objects="chair")

# ------------------------------------------------------------------ floor sit
FLOOR = {"root": {"y": 9.0}, "support": "floor"}
base("fl_cross", "floor", merge(FLOOR, {
    "root": {"pitch": 6}, "spine": {"pitch": 6},
    "legs": {"L": leg(-24, 8, 38, [1, 0.3, 0.3], planted=False, mode="free", pitch=20),
             "R": leg(5.5, 12, 35, [-1, 0.45, 0.3], planted=False, mode="free", pitch=20)}}),
     "sitting", "바닥에 양반다리로 앉아", "양반다리로", actions="sit")
base("fl_knees_up", "floor", merge(FLOOR, {
    "root": {"pitch": -22}, "spine": {"pitch": 18},
    "legs": {"L": leg(12, 9.5, 54, [0.15, 1, 0.3], planted=True),
             "R": leg(-12, 9.5, 54, [-0.15, 1, 0.3], planted=True)}}),
     "sitting", "바닥에 무릎을 세우고 앉아", "무릎 세우고 앉아", actions="sit")
base("fl_legs_out", "floor", merge(FLOOR, {
    "root": {"pitch": -10}, "spine": {"pitch": 10},
    "legs": {"L": leg(11, 9, 82, [0.1, 1, 0.1], mode="free", pitch=-20, planted=False),
             "R": leg(-12, 9, 78, [-0.1, 1, 0.1], mode="free", pitch=-20, planted=False)}}),
     "sitting", "바닥에 두 다리를 앞으로 쭉 뻗고 앉아", "다리 뻗고 바닥에", actions="sit")
base("fl_one_knee", "floor", merge(FLOOR, {
    "root": {"pitch": -18}, "spine": {"pitch": 16},
    "legs": {"L": leg(13, 9.5, 54, [0.2, 1, 0.3], planted=True),
             "R": leg(-12, 9, 80, [-0.1, 1, 0.1], mode="free", pitch=-20, planted=False)}}),
     "sitting", "바닥에 한쪽 무릎만 세우고 앉아", "한 무릎 세우고 앉아", actions="sit")

# ------------------------------------------------------------------ kneel
base("kn_upright", "kneel", {
    "root": {"y": 40, "z": 0},
    "legs": {"L": leg(11, 16, -36, [0, -0.3, 1], planted=False, mode="free", pitch=0, toe=60),
             "R": leg(-11, 16, -36, [0, -0.3, 1], planted=False, mode="free", pitch=0, toe=60)},
    "support": "floor"}, "kneeling", "두 무릎을 꿇고 상체를 세워", "무릎 꿇고 서서", actions="kneel")
base("kn_seiza", "kneel", {
    "root": {"y": 30, "z": -6, "pitch": 4},
    "legs": {"L": leg(10, 6, -22, [0.05, 0.25, 1], planted=False, mode="free", pitch=0, toe=60),
             "R": leg(-10, 6, -22, [-0.05, 0.25, 1], planted=False, mode="free", pitch=0, toe=60)},
    "support": "floor"}, "kneeling", "무릎을 꿇고 발뒤꿈치 위에 앉아", "꿇어앉아", actions="kneel,sit")
base("kn_one", "kneel", {
    "root": {"y": 42, "z": 0},
    "legs": {"L": leg(14, 7.2, 34, [0.2, 0.2, 1]),
             "R": leg(-11, 16, -32, [0, -0.3, 1], planted=False, mode="free", pitch=0, toe=60)},
    "support": "floor"}, "kneeling", "한쪽 무릎을 꿇고 다른 발은 세워", "한 무릎 꿇고", actions="kneel")

# ------------------------------------------------------------------ squat / crouch
base("sq_deep", "squat", {
    "root": {"y": 34, "z": -10, "pitch": 20}, "spine": {"pitch": 26},
    "legs": {"L": leg(15, 10.4, 12, [0.45, 0.1, 1], planted=True),
             "R": leg(-15, 10.4, 12, [-0.45, 0.1, 1], planted=True)}},
     "crouching", "깊게 쪼그려 앉아", "쪼그려 앉아", actions="crouch")
base("sq_half", "squat", merge(stand(width=14, y=66, toe_out=14), {
    "root": {"pitch": 26, "z": -10}, "spine": {"pitch": 14},
    "legs": {"L": leg(14, 10.4, 10, [0.35, 0.1, 1]), "R": leg(-14, 10.4, 10, [-0.35, 0.1, 1])}}),
     "crouching", "무릎을 반쯤 굽혀 몸을 웅크리고", "반쯤 웅크려", actions="crouch,prepare")

# ------------------------------------------------------------------ lying
base("ly_back", "lie_back", {"root": {"y": 12, "pitch": -90}, "place": "all", "support": "floor", "settle": True,
                            "legs": {"L": leg(10, -84, 2, [0.1, 0, 1], frame="p", mode="free", pitch=-20,
                                              planted=False),
                                     "R": leg(-10, -84, 2, [-0.1, 0, 1], frame="p", mode="free", pitch=-20,
                                              planted=False)}},
     "lying", "바닥에 등을 대고 반듯이 누워", "반듯이 누워", actions="rest")
base("ly_back_knee", "lie_back", {"root": {"y": 12, "pitch": -90}, "place": "all", "support": "floor",
                                 "settle": True,
                                 "legs": {"L": leg(10, -84, 2, [0.1, 0, 1], frame="p", mode="free", pitch=-20,
                                                   planted=False),
                                          "R": leg(-11, -50, 38, [-0.1, 0, 1], frame="p", mode="free", pitch=0,
                                                   planted=False)}},
     "lying", "바닥에 누워 한쪽 무릎을 세우고", "누워 한 무릎 세우고", actions="rest")
base("ly_front", "lie_front", {"root": {"y": 12, "pitch": 74}, "place": "all", "settle": True, "spine": {"pitch": -24},
                              "head": {"pitch": 18},
                              "legs": {"L": leg(10, 7, -82, [0, -1, 0], mode="free", pitch=40, planted=False),
                                       "R": leg(-10, 30, -64, [0, -1, 0.2], mode="free", pitch=30,
                                                planted=False)}},
     "lying", "배를 깔고 엎드려 팔꿈치로 상체를 받치고", "엎드려", actions="rest")
base("ly_side", "lie_side", {"root": {"y": 18, "roll": -88}, "place": "all", "settle": True,
                            "head": {"roll": 12, "pitch": 6}, "spine": {"pitch": 10},
                            "legs": {"L": leg(-4, -48, 46, [0, 0.3, 1], frame="p", mode="free", pitch=20,
                                              planted=False),
                                     "R": leg(-10, -58, 30, [0, 0.3, 1], frame="p", mode="free", pitch=20,
                                              planted=False)}},
     "lying", "옆으로 누워 무릎을 굽히고", "옆으로 누워", actions="rest")
