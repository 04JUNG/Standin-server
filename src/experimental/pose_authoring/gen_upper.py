"""Upper-body actions for the combinatorial generator (actor-right = -x is the active arm).

Each action: groups it fits, arms (both, or only R when one=True and the left
arm takes a group filler), hand shapes, optional head/spine/clavicle offsets,
and search-v2 tag pieces. Coordinates are in the chest frame "c" (origin
Spine2, +y up, +z forward) unless another frame is given.
"""
from .presets import arm, touch, folded, DESK_H
from .presets import palm_at as _palm_at


def palm_at(x, y, z, frame="c", **kw):
    return _palm_at(x, y, z, frame, **kw)

UP = {"stand", "chair", "desk", "floor", "kneel", "squat"}
SEATED = {"chair", "desk", "floor", "kneel"}
STAND = {"stand"}
STAND_KNEEL = {"stand", "kneel"}
NO_SQUAT = UP - {"squat"}

U = {}


def ub(uid, groups, desc, label, arms, hands, one=False, head=None, spine=None, clav=None,
       acts="", objs="", domain="", activity="", emo="", intent=""):
    U[uid] = {"groups": set(groups), "desc": desc, "label": label, "arms": arms, "hands": hands,
              "one": one, "head": head or {}, "spine": spine or {}, "clav": clav,
              "acts": acts, "objs": objs, "domain": domain, "activity": activity, "emo": emo, "intent": intent}


def sym(x, y, z, frame="c", pole=(0.4, -1, -0.2), **kw):
    """Same target for both arms, mirrored in x."""
    px, py, pz = pole
    return {"L": arm(x, y, z, frame, pole=[px, py, pz], **kw),
            "R": arm(-x, y, z, frame, pole=[-px, py, pz], **kw)}


def sym_palm(x, y, z, face, frame="c", pole=(0.4, -1, -0.2), **kw):
    fx, fy, fz = face
    px, py, pz = pole
    return {"L": palm_at(x, y, z, frame, face=[fx, fy, fz], pole=[px, py, pz], **kw),
            "R": palm_at(-x, y, z, frame, face=[-fx, fy, fz], pole=[-px, py, pz], **kw)}


def sym_touch(nameL, nameR, offL=(0, 0, 0), pole=(0.5, -1, 0), **kw):
    px, py, pz = pole
    offR = (-offL[0], offL[1], offL[2])
    return {"L": touch(nameL, offL, pole=[px, py, pz], **kw), "R": touch(nameR, offR, pole=[-px, py, pz], **kw)}


# ================================================================== communication
ub("wave_high", UP, "한 손을 머리 위로 높이 들어 흔들며 인사하기", "손 높이 흔들기",
   {"R": arm(-30, 44, 10, pole=[-1, -0.3, -0.4], pron=-70)}, {"R": "open"}, one=True,
   head={"pitch": -4}, acts="wave,gesture", emo="happy", intent="greeting")
ub("wave_low", UP, "손바닥을 앞으로 보이며 가볍게 손 인사하기", "가볍게 손 인사",
   {"R": palm_at(-24, 8, 22, face=[0, 0, 1], pole=[-0.5, -1, -0.2])}, {"R": "open"}, one=True,
   acts="wave", intent="greeting")
ub("beckon", UP, "손바닥을 위로 하고 손짓해 이리 오라고 부르기", "손짓해 부르기",
   {"R": arm(-14, -8, 34, pole=[-0.4, -1, -0.2], pron=-75)}, {"R": "soft"}, one=True,
   acts="gesture", intent="invite")
ub("point_fwd", UP, "한 팔을 앞으로 곧게 뻗어 손가락으로 가리키기", "앞을 가리키기",
   {"R": arm(-22, 6, 50, pole=[-0.3, -1, 0], pron=30)}, {"R": "point"}, one=True,
   spine={"yaw": -5}, acts="point", intent="communicate")
ub("point_up", UP, "검지로 하늘 위를 가리키기", "위를 가리키기",
   {"R": arm(-20, 50, 16, pole=[-1, 0, -0.3], pron=0)}, {"R": "point"}, one=True,
   head={"pitch": -14}, acts="point", intent="request_attention")
ub("point_side", UP, "고개를 돌려 옆쪽을 손가락으로 가리키기", "옆을 가리키기",
   {"R": arm(-52, 8, 18, pole=[-0.2, -1, -0.4], pron=20)}, {"R": "point"}, one=True,
   head={"yaw": -40}, acts="point,look", intent="communicate")
ub("raise_hand", UP, "손을 번쩍 들어 질문하거나 발표하기", "손 들기",
   {"R": arm(-18, 62, 4, pole=[-1, 0, -0.3], pron=-80)}, {"R": "open"}, one=True,
   acts="raise_hand", activity="study", intent="request_attention")
ub("thumbs_up", UP, "엄지를 치켜세워 보이기", "엄지 척",
   {"R": arm(-14, -8, 34, pole=[-0.5, -1, -0.2], pron=0)}, {"R": "thumb"}, one=True,
   acts="gesture", emo="happy", intent="communicate")
ub("ok_sign", UP, "손으로 OK 표시하기", "OK 표시",
   {"R": palm_at(-14, 6, 34, face=[0, 0, 1], pole=[-0.5, -1, -0.2])}, {"R": "ok"}, one=True,
   acts="gesture", intent="communicate")
ub("v_face", UP, "얼굴 옆에서 브이 하기", "얼굴 옆 브이",
   {"R": palm_at(-10, 6, 15, "h", face=[0, 0, 1], pole=[-0.6, -1, 0])}, {"R": "v"}, one=True,
   head={"roll": 10, "yaw": 6}, acts="gesture", emo="happy", intent="express_performance")
ub("v_double", UP, "양손으로 브이를 만들어 들기", "양손 브이",
   sym_palm(20, 8, 26, (0, 0, 1), pole=(0.5, -1, -0.2)), {"L": "v", "R": "v"},
   head={"roll": 8}, acts="gesture", emo="happy", intent="express_performance")
ub("shh", UP, "검지를 입술에 대고 쉿 하기", "쉿",
   {"R": touch("mouth", (0, 0.5, 0.8), pole=[-0.3, -1, 0], grip="tip", face=None, pron=0)}, {"R": "shh"},
   one=True, head={"pitch": 4}, acts="gesture", intent="communicate")
ub("salute", STAND_KNEEL | {"chair"}, "오른손을 이마에 대고 거수경례하기", "거수경례",
   {"R": touch("forehead", (-6, 0, -1), pole=[-1, -0.2, 0.1], face=[0.4, -0.2, -1])}, {"R": "flat"},
   one=True, acts="gesture", intent="respect")
ub("handshake", UP, "악수하려고 손을 앞으로 내밀기", "악수 손 내밀기",
   {"R": arm(-10, -10, 46, pole=[-0.4, -1, -0.1], pron=0)}, {"R": "soft"}, one=True,
   acts="reach,give", intent="greeting")
ub("stop_palm", UP, "손바닥을 앞으로 내밀어 멈추라는 손짓", "멈춰 손바닥",
   {"R": palm_at(-12, 6, 48, face=[0, 0, 1], pole=[-0.4, -1, 0])}, {"R": "open"}, one=True,
   acts="gesture", intent="defense")
ub("wave_no", UP, "양손을 가슴 앞에서 내저으며 손사래 치기", "손사래",
   {"L": palm_at(11, 4, 30, face=[0, 0, 1], pole=[0.5, -1, -0.2]),
    "R": palm_at(-13, 1, 31, face=[0, 0, 1], pole=[-0.5, -1, -0.2])}, {"L": "open", "R": "open"},
   head={"yaw": -10}, acts="gesture", emo="shy", intent="communicate")
ub("shrug", UP, "어깨를 으쓱하며 양손바닥을 위로 펼치기", "어깨 으쓱",
   sym_palm(30, -18, 18, (0, 1, 0), pole=(0.2, -1, -0.5)), {"L": "open", "R": "open"},
   head={"roll": 10}, clav={"L": [12, 0], "R": [12, 0]}, acts="gesture", intent="communicate")
ub("explain_open", UP, "양팔을 벌려 손바닥을 위로 보이며 설명하기", "양팔 벌려 설명",
   sym(40, -18, 24, pole=(0.3, -1, -0.3), pron=-75), {"L": "flat", "R": "flat"},
   acts="talk,gesture", activity="conversation", intent="communicate")
ub("talk_hand", UP, "한 손을 가슴 높이로 들고 이야기하기", "손 들고 말하기",
   {"R": arm(-17, -8, 25, pole=[-0.4, -1, -0.2], pron=-30)}, {"R": "relaxed"}, one=True,
   acts="talk,gesture", activity="conversation", intent="communicate")
ub("accuse", UP, "몸을 내밀며 손가락으로 상대를 삿대질하기", "삿대질",
   {"R": arm(-16, 22, 48, pole=[-0.4, -1, 0], pron=10)}, {"R": "point"}, one=True,
   spine={"pitch": 6, "yaw": -6}, head={"pitch": -6}, acts="point", emo="angry", intent="provoke")

# ================================================================== daily / props
ub("phone_one", UP, "한 손으로 휴대폰을 들고 내려다보기", "한 손 폰 보기",
   {"R": arm(-6, -6, 27, pole=[-0.4, -1, -0.2], pron=-60, flex=-10)}, {"R": "phone"}, one=True,
   head={"pitch": 28}, spine={"pitch": 4}, acts="hold,look", objs="phone", intent="focus_task")
ub("phone_both", UP, "양손으로 휴대폰을 쥐고 화면 보기", "양손 폰 보기",
   sym(10, -12, 28, pole=(0.5, -1, -0.2), pron=-65, flex=-12), {"L": "phone", "R": "phone"},
   head={"pitch": 30}, spine={"pitch": 6}, acts="hold,look,type", objs="phone", intent="focus_task")
ub("phone_call", UP, "휴대폰을 귀에 대고 통화하기", "전화 통화",
   {"R": touch("earR", (-4, -3, 4), pole=[-0.4, -1, 0.3], face=None, pron=-40)}, {"R": "phone"}, one=True,
   head={"roll": -6}, acts="hold,talk", objs="phone", activity="conversation", intent="communicate")
ub("selfie", UP, "한 팔을 앞으로 높이 뻗어 셀카 찍기", "셀카",
   {"R": palm_at(-20, 26, 46, face=[0, -0.2, -1], pole=[-0.6, -1, 0])}, {"R": "phone"}, one=True,
   head={"roll": 10, "pitch": -8, "yaw": -6}, acts="hold,look", objs="phone", emo="happy")
ub("drink_cup", UP, "컵을 입에 대고 마시기", "컵으로 마시기",
   {"R": touch("mouth", (-1, -1, 7.5), pole=[-0.6, -1, -0.1], grip=True, face=None, pron=0)}, {"R": "cup"},
   one=True, head={"pitch": -10}, acts="drink,hold", objs="cup", activity="rest")
ub("cup_chest", UP, "컵을 가슴 앞에 들고 있기", "컵 들고",
   {"R": arm(-9, -14, 26, pole=[-0.5, -1, -0.2], pron=-15)}, {"R": "cup"}, one=True,
   acts="hold", objs="cup", activity="rest", intent="relax")
ub("cup_both", UP, "양손으로 따뜻한 컵을 감싸 쥐기", "양손으로 컵",
   sym_palm(5, -14, 26, (-1, 0, 0), pole=(0.6, -1, -0.2)), {"L": "cup", "R": "cup"},
   head={"pitch": 14}, acts="hold", objs="cup", activity="rest", intent="relax")
ub("read_book", UP, "양손으로 책을 펼쳐 들고 읽기", "책 읽기",
   {"L": palm_at(12, -12, 26, face=[-0.3, 1, -0.2], pole=[0.4, -1, -0.2]),
    "R": palm_at(-12, -12, 26, face=[0.3, 1, -0.2], pole=[-0.4, -1, -0.2])}, {"L": "soft", "R": "soft"},
   head={"pitch": 30}, spine={"pitch": 4}, acts="read,hold", objs="book", activity="study", intent="focus_task")
ub("book_hug", UP, "책을 가슴에 꼭 끌어안기", "책 끌어안기",
   {"L": arm(-4, -6, 22, pole=[0.6, -1, -0.2], pron=70),
    "R": arm(4, -11, 24, pole=[-0.6, -1, -0.2], pron=70)}, {"L": "flat", "R": "flat"},
   acts="hold", objs="book", activity="study", emo="shy")
ub("carry_box", STAND | {"kneel"}, "상자를 양손으로 받쳐 들고 있기", "상자 들기",
   sym_palm(20, -22, 30, (-1, 0.3, 0), pole=(0.4, -1, -0.3)), {"L": "open", "R": "open"},
   head={"pitch": 8}, acts="carry,hold", objs="box", intent="focus_task")
ub("bag_strap", UP, "어깨에 멘 가방끈을 가슴 앞에서 잡기", "가방끈 잡기",
   {"R": touch("chestR", (-2, 3, 4.0), pole=[-0.6, -1, -0.2], grip=True, face=None, pron=30)}, {"R": "grip"},
   one=True, acts="hold", objs="bag")
ub("backpack", STAND_KNEEL, "백팩 양쪽 끈을 두 손으로 잡기", "백팩 끈 잡기",
   {"L": touch("chestL", (2, 3, 4.0), pole=[0.6, -1, -0.2], grip=True, face=None, pron=30),
    "R": touch("chestR", (-2, 3, 4.0), pole=[-0.6, -1, -0.2], grip=True, face=None, pron=30)},
   {"L": "grip", "R": "grip"}, acts="hold", objs="bag", activity="study")
ub("bag_carry", STAND, "손에 가방을 들고 있기", "가방 들고",
   {"R": arm(-22, -2, 7, "p", pole=[-0.3, -0.3, -1], pron=30)}, {"R": "grip"}, one=True,
   acts="carry,hold", objs="bag")
ub("umbrella", UP, "우산 손잡이를 가슴 앞에서 쥐고 쓰기", "우산 쓰기",
   {"R": arm(-12, 2, 24, pole=[-0.5, -1, -0.2], pron=-5)}, {"R": "grip"}, one=True,
   acts="hold", objs="umbrella")
ub("watch_check", UP, "손목을 들어 시계를 확인하기", "시계 보기",
   {"L": arm(4, -6, 30, pole=[1, -1, -0.1], pron=80)}, {"L": "relaxed"}, one="L",
   head={"pitch": 26, "yaw": 8}, acts="look", intent="wait_turn")
ub("glasses", UP, "검지로 안경을 밀어 올리기", "안경 고쳐 쓰기",
   {"R": touch("nose", (0, 3.0, 2.0), pole=[-0.4, -1, 0], grip="tip", face=None, pron=0)}, {"R": "point"},
   one=True, acts="groom", intent="focus_task")
ub("eat_bowl", SEATED | {"stand"}, "그릇을 들고 젓가락으로 음식을 입에 넣기", "그릇 들고 먹기",
   {"L": arm(6, -14, 26, pole=[0.5, -1, -0.2], pron=-80),
    "R": arm(-4, -9, 14, "h", pole=[-0.7, -1, 0], pron=-35)}, {"L": "cup", "R": "pinch"},
   head={"pitch": 10}, acts="eat,hold", objs="cup")
ub("tray", STAND, "식판을 양손으로 수평으로 들고 서기", "식판 들기",
   sym(18, -16, 24, pole=(0.3, -1, -0.3), pron=-80), {"L": "open", "R": "open"},
   head={"pitch": 14}, acts="carry,hold", objs="box")
ub("cook", STAND, "한 손으로 팬을 잡고 다른 손으로 뒤집개를 쥐고 요리하기", "요리하기",
   {"L": arm(12, -20, 32, pole=[0.5, -1, -0.2], pron=10),
    "R": arm(-6, -16, 36, pole=[-0.5, -1, -0.2], pron=20)}, {"L": "grip", "R": "grip"},
   head={"pitch": 26}, spine={"pitch": 6}, acts="cook,hold", activity="cooking", intent="focus_task")
ub("broom", STAND, "빗자루를 양손으로 쥐고 바닥 쓸기", "빗자루질",
   {"L": arm(-2, -6, 26, pole=[0.6, -1, -0.2], pron=10),
    "R": arm(-14, -30, 32, pole=[-0.4, -1, -0.2], pron=10)}, {"L": "grip", "R": "grip"},
   head={"pitch": 24}, spine={"pitch": 8}, acts="clean,hold", objs="broom", intent="focus_task")
ub("rope_pull", STAND, "밧줄을 양손으로 잡고 뒤로 당기기", "밧줄 당기기",
   {"L": arm(2, -8, 46, pole=[0.4, -1, -0.2], pron=0), "R": arm(-4, -10, 36, pole=[-0.4, -1, -0.2], pron=0)},
   {"L": "grip", "R": "grip"}, spine={"pitch": -8}, acts="pull,hold", objs="rope", intent="focus_task")
ub("push", STAND, "양손바닥으로 앞을 힘껏 밀기", "앞으로 밀기",
   sym_palm(14, 6, 46, (0, 0, 1), pole=(0.4, -1, -0.2)), {"L": "open", "R": "open"},
   spine={"pitch": 12}, acts="push", intent="focus_task")
ub("lantern", UP, "랜턴을 앞으로 들어 비추기", "랜턴 들기",
   {"R": arm(-20, 10, 38, pole=[-0.5, -1, -0.2], pron=10)}, {"R": "grip"}, one=True,
   head={"pitch": -4}, acts="hold,look", objs="lantern", intent="observe")
ub("torch_high", STAND_KNEEL, "횃불을 머리 위로 높이 치켜들기", "횃불 치켜들기",
   {"R": arm(-24, 48, 18, pole=[-1, -0.3, -0.3], pron=10)}, {"R": "grip"}, one=True,
   head={"pitch": -10}, acts="hold,raise_hand", objs="torch", domain="fantasy")
ub("reach_up", STAND_KNEEL, "위쪽 선반에 손을 쭉 뻗기", "위로 손 뻗기",
   {"R": arm(-16, 62, 20, pole=[-1, 0, -0.3], pron=-20)}, {"R": "open"}, one=True,
   head={"pitch": -22}, acts="reach,stretch")
ub("reach_fwd", UP, "앞에 있는 것을 잡으려 손을 뻗기", "앞으로 손 뻗기",
   {"R": arm(-15, -6, 46, pole=[-0.3, -1, 0], pron=40)}, {"R": "open"}, one=True,
   spine={"pitch": 10}, acts="reach")
ub("stretch_up", UP, "깍지 낀 양팔을 머리 위로 뻗어 기지개 켜기", "기지개",
   sym(6, 62, 4, pole=(1, 0, -0.3), pron=-80), {"L": "open", "R": "open"},
   head={"pitch": -10}, acts="stretch", intent="release_tension")
ub("yawn", UP, "한 손으로 하품하는 입을 가리고 다른 팔은 위로 뻗기", "하품",
   {"R": touch("mouth", (0, -1, 0), pole=[-0.8, -1, -0.1]), "L": arm(10, 64, -2, pole=[1, 0.2, -0.3], pron=0)},
   {"R": "flat", "L": "open"}, head={"pitch": -12}, acts="stretch", emo="tired", intent="release_tension")
ub("behind_back", STAND, "두 손을 등 뒤로 모아 뒷짐 지기", "뒷짐",
   sym(8, -36, -16, pole=(0.4, -0.2, -1), pron=20), {"L": "relaxed", "R": "relaxed"},
   acts="wait", intent="observe")
ub("tie_hair", NO_SQUAT, "양손을 뒤통수로 올려 머리를 묶기", "머리 묶기",
   sym_touch("backhead", "backhead", (4, 3, 0), pole=(0.8, 0.3, 0.6)), {"L": "pinch", "R": "pinch"},
   head={"pitch": 10}, acts="groom", intent="prepare_action")
ub("hands_head_back", UP, "깍지 낀 두 손을 뒤통수에 대고 느긋하게 있기", "깍지 끼고 뒤통수",
   sym_touch("backhead", "backhead", (5, 2, 0), pole=(1, 0.2, 0.3)), {"L": "flat", "R": "flat"},
   head={"pitch": -6}, acts="rest", intent="relax")

# ================================================================== desk only
ub("desk_write", {"desk"}, "노트에 필기하기", "필기",
   {"R": palm_at(-8, DESK_H + 1.5, 42, "w", pole=[-1, -0.5, -0.2]),
    "L": palm_at(16, DESK_H + 0.4, 42, "w", pole=[1, -0.5, -0.2])}, {"R": "pinch", "L": "flat"},
   spine={"pitch": 10}, head={"pitch": 28}, acts="write", objs="book", activity="study", intent="focus_task")
ub("desk_type", {"desk"}, "책상 위 노트북 키보드로 타이핑하기", "타이핑",
   {"L": palm_at(10, DESK_H + 1.2, 40, "w", pole=[1, -0.5, -0.2]),
    "R": palm_at(-10, DESK_H + 1.2, 40, "w", pole=[-1, -0.5, -0.2])}, {"L": "relaxed", "R": "relaxed"},
   head={"pitch": 16}, acts="type", objs="computer", activity="work", intent="focus_task")
ub("desk_slump", {"desk"}, "책상에 팔을 포개고 엎드리기", "책상에 엎드리기",
   {"L": arm(-5, DESK_H + 3.8, 60, "w", pole=[1, -0.5, -0.3], pron=80),
    "R": arm(5, DESK_H + 3.8, 60, "w", pole=[-1, -0.5, -0.3], pron=80)}, {"L": "relaxed", "R": "relaxed"},
   spine={"pitch": 30}, head={"pitch": 26}, acts="rest,lean", activity="study", emo="tired", intent="rest_recover")
ub("desk_lean_arm", {"desk"}, "한 팔을 책상 위에 접어 기대고 이야기하기", "책상에 팔 걸치고",
   {"L": arm(-4, DESK_H + 3.2, 34, "w", pole=[1, -0.6, -0.4], pron=80)}, {"L": "relaxed"}, one="L",
   spine={"pitch": 8, "yaw": 6}, acts="lean,talk", activity="conversation", intent="communicate")
ub("desk_both_palms", {"desk"}, "양손을 책상 위에 가지런히 올려놓기", "책상에 양손",
   {"L": palm_at(16, DESK_H + 0.4, 38, "w", pole=[1, -0.5, -0.2]),
    "R": palm_at(-16, DESK_H + 0.4, 38, "w", pole=[-1, -0.5, -0.2])}, {"L": "flat", "R": "flat"},
   acts="wait", activity="work", intent="wait_turn")

# ================================================================== emotion
ub("scratch_head", UP, "한 손으로 뒷머리를 긁으며 멋쩍어하기", "뒷머리 긁기",
   {"R": touch("backhead", (-4, 3, 0), pole=[-0.6, 0.4, 1])}, {"R": "soft"}, one=True,
   head={"pitch": 6, "roll": -8, "yaw": 8}, acts="gesture", emo="shy")
ub("hand_on_head", UP, "한 손으로 정수리를 짚기", "머리 짚기",
   {"R": touch("crownR", (0, 0, 0), pole=[-1, 0.2, 0.3])}, {"R": "soft"}, one=True,
   head={"pitch": 10, "roll": -6}, acts="gesture", emo="tired")
ub("facepalm", UP, "손바닥으로 이마를 짚으며 고개 숙이기", "이마 짚기",
   {"R": touch("forehead", (-1, 0, 0), pole=[-0.5, -1, 0])}, {"R": "flat"}, one=True,
   head={"pitch": 20}, acts="gesture", emo="tired", intent="release_tension")
ub("head_in_hands", NO_SQUAT, "양손으로 머리를 감싸 쥐고 괴로워하기", "머리 감싸 쥐기",
   sym_touch("templeL", "templeR", (0, 1, -1), pole=(0.6, -1, 0.5)), {"L": "soft", "R": "soft"},
   head={"pitch": 22}, spine={"pitch": 8}, acts="gesture", emo="sad")
ub("cover_face", UP, "양손으로 얼굴을 가리고 고개 숙이기", "얼굴 가리기",
   sym_touch("eyeL", "eyeR", (1, -1, 0), pole=(0.5, -1, 0)), {"L": "flat", "R": "flat"},
   head={"pitch": 24}, spine={"pitch": 6}, acts="gesture", emo="shy")
ub("cover_mouth_both", UP, "양손으로 입을 막고 놀라기", "양손 입 막기",
   sym(4.5, -10, 15, "h", pole=(0.5, -1, 0), pron=-80, flex=15), {"L": "flat", "R": "flat"},
   head={"pitch": 6}, acts="gesture", emo="surprised")
ub("cover_mouth_one", UP, "한 손으로 입을 가리고 놀라기", "한 손 입 막기",
   {"R": arm(-2, -10, 15, "h", pole=[-0.5, -1, 0], pron=-80, flex=15)}, {"R": "flat"}, one=True,
   head={"pitch": 2}, acts="gesture", emo="surprised")
ub("ears_cover", UP, "양손으로 귀를 꽉 막기", "귀 막기",
   sym_touch("earL", "earR", (0, 0, 0), pole=(0.6, -1, 0.2)), {"L": "flat", "R": "flat"},
   head={"pitch": 12}, acts="gesture", emo="afraid", intent="avoid_danger")
ub("cheeks", UP, "양손으로 두 볼을 감싸 쥐기", "양볼 감싸기",
   sym_touch("cheekL", "cheekR", (0, 0, 0), pole=(0.5, -1, 0)), {"L": "soft", "R": "soft"},
   head={"roll": 6}, acts="gesture", emo="shy")
ub("wipe_tears", UP, "손등으로 눈물을 훔치기", "눈물 닦기",
   {"R": touch("eyeR", (-1.5, -1.5, 5.5), pole=[-0.4, -1, 0], face=[0, 0, 1])}, {"R": "soft"}, one=True,
   head={"pitch": 14, "roll": -6}, acts="gesture", emo="sad")
ub("hand_chest", UP, "한 손을 가슴에 얹고 안도하기", "가슴에 손",
   {"R": touch("chestR", (2, 0, 0), pole=[-0.6, -1, -0.2])}, {"R": "flat"}, one=True,
   head={"pitch": 6}, acts="gesture", emo="relieved", intent="respect")
ub("pray", UP, "양손을 가슴 앞에 모아 기도하기", "두 손 모아 기도",
   sym_palm(1.6, -2, 24, (-1, 0, 0), pole=(0.6, -1, -0.2)), {"L": "flat", "R": "flat"},
   head={"pitch": 14}, acts="pray", intent="respect")
ub("clap", UP, "가슴 앞에서 손뼉을 치기", "박수",
   sym_palm(1.6, 2, 32, (-1, 0, 0.2), pole=(0.6, -1, -0.2)), {"L": "flat", "R": "flat"},
   head={"pitch": -4}, acts="clap", emo="happy", intent="express_performance")
ub("banzai", UP, "두 팔을 번쩍 들어 만세 하기", "만세",
   sym(22, 58, 10, pole=(1, 0, -0.3), pron=-60), {"L": "open", "R": "open"},
   head={"pitch": -14}, acts="raise_hand", emo="happy", intent="express_performance")
ub("fists_up", UP, "두 주먹을 높이 치켜들고 환호하기", "두 주먹 환호",
   sym(24, 50, 14, pole=(1, -0.4, -0.3), pron=0), {"L": "fist", "R": "fist"},
   head={"pitch": -12}, acts="raise_hand", emo="happy", intent="express_performance")
ub("fist_pump", UP, "한 주먹을 높이 치켜들며 기뻐하기", "주먹 치켜들기",
   {"R": arm(-22, 60, 12, pole=[-1, 0, -0.3], pron=0)}, {"R": "fist"}, one=True,
   head={"pitch": -10}, acts="raise_hand", emo="happy")
ub("fist_small", UP, "가슴 앞에서 주먹을 불끈 쥐며 다짐하기", "주먹 불끈",
   {"R": arm(-14, -14, 22, pole=[-0.5, -1, -0.2], pron=10)}, {"R": "fist"}, one=True,
   acts="gesture", intent="confident")
ub("fists_anger", STAND_KNEEL, "주먹을 꽉 쥐고 어깨를 웅크린 채 화내기", "주먹 쥐고 화내기",
   sym(24, -2, 12, "p", pole=(0.6, -0.3, -1), pron=40), {"L": "fist", "R": "fist"},
   head={"pitch": 10}, spine={"pitch": 6}, clav={"L": [9, 4], "R": [9, 4]}, acts="gesture", emo="angry")
ub("surprised_open", UP, "깜짝 놀라 양팔을 벌리기", "놀라 양팔 벌리기",
   sym(36, -2, -6, "p", pole=(0.2, -0.3, -1), pron=-30), {"L": "open", "R": "open"},
   head={"pitch": -6}, acts="recoil", emo="surprised")
ub("startled_hands", UP, "놀라서 양손을 앞으로 들어 막기", "놀라 손 들기",
   {"L": palm_at(14, 8, 30, face=[0, 0, 1], pole=[0.5, -1, -0.3]),
    "R": palm_at(-14, 5, 32, face=[0, 0, 1], pole=[-0.5, -1, -0.3])}, {"L": "open", "R": "open"},
   spine={"pitch": -4}, acts="recoil", emo="afraid", intent="defense")
ub("dejected", UP, "어깨를 축 늘어뜨리고 고개를 떨구기", "낙담",
   sym(19, -3, 13, "p", pole=(0.4, -0.3, -1), pron=30), {"L": "relaxed", "R": "relaxed"},
   head={"pitch": 32}, spine={"pitch": 14}, clav={"L": [-3, 6], "R": [-3, 6]}, acts="wait", emo="sad")
ub("self_hug", UP, "양팔로 몸을 감싸 안고 움츠리기", "몸 감싸 안기",
   folded("L", y=-2, z=-2), {"L": "soft", "R": "soft"}, head={"pitch": 12},
   acts="hug", emo="afraid", intent="protect")
ub("neck_hold", UP, "한 손으로 뒷목을 잡기", "뒷목 잡기",
   {"R": touch("backhead", (-3, -7, -1), pole=[-1, 0.3, 0.5])}, {"R": "soft"}, one=True,
   head={"pitch": -10}, acts="gesture", emo="tired", intent="release_tension")
ub("cough", UP, "주먹을 입가에 대고 기침하기", "기침",
   {"R": arm(-1, -9, 14, "h", pole=[-0.5, -1, 0], pron=-30)}, {"R": "fist"}, one=True,
   head={"pitch": 18}, spine={"pitch": 10}, acts="gesture", emo="tired")
ub("chin_think", UP, "한 손으로 턱을 괴고 다른 팔로 팔꿈치를 받치며 생각하기", "턱 괴고 생각",
   {"R": touch("chin", (0, -2.5, 3.5), pole=[-0.1, -1, 0.3], grip=True, face=None, pron=-10),
    "L": palm_at(-7, -21, 22, face=[0, 1, 0], pole=[1, -1, -0.2])}, {"R": "fist", "L": "relaxed"},
   head={"pitch": 6, "roll": 6}, acts="gesture", intent="observe")
ub("chin_touch", UP, "검지와 엄지로 턱을 만지며 고민하기", "턱 만지기",
   {"R": touch("chin", (0, -3.0, 4.0), pole=[-0.3, -1, 0.2], grip=True, face=None, pron=-10)}, {"R": "pinch"},
   one=True, head={"pitch": -4, "yaw": 10}, acts="gesture", intent="observe")
ub("cross_arms", UP, "팔짱을 끼기", "팔짱",
   folded("L", y=-12), {"L": "relaxed", "R": "relaxed"}, acts="fold_arms", intent="observe")
ub("cross_arms_hi", UP, "팔짱을 높게 끼고 당당하게 버티기", "높은 팔짱",
   folded("R", y=-6), {"L": "relaxed", "R": "relaxed"}, head={"pitch": -6},
   acts="fold_arms", intent="confident")
ub("hands_hips", STAND_KNEEL, "양손을 허리에 짚고 당당한 자세 취하기", "양손 허리",
   sym_touch("hipL", "hipR", pole=(1, 0, -0.4)), {"L": "flat", "R": "flat"}, head={"pitch": -4},
   acts="gesture", intent="confident")
ub("hand_hip_talk", STAND, "한 손은 허리에 짚고 다른 손을 내밀며 말하기", "한 손 허리 말하기",
   {"L": touch("hipL", pole=[1, 0, -0.4]),
    "R": palm_at(-20, -12, 30, face=[0, 1, 0], pole=[-0.4, -1, -0.3])}, {"L": "flat", "R": "relaxed"},
   acts="talk,gesture", activity="conversation", intent="communicate")
ub("bow", {"stand"}, "두 팔을 몸에 붙이고 허리를 숙여 인사하기", "허리 숙여 인사",
   sym(21.5, -6, 1, "p", pole=(0.2, -0.2, -1), pron=20), {"L": "soft", "R": "soft"},
   spine={"pitch": 28}, head={"pitch": 10}, acts="bow", intent="respect")
ub("attention", {"stand"}, "팔을 몸에 붙이고 차렷하기", "차렷",
   sym(21.5, -6, 1, "p", pole=(0.2, -0.2, -1), pron=20), {"L": "soft", "R": "soft"},
   acts="wait", intent="respect")

# ================================================================== romance
ub("heart_big", NO_SQUAT, "두 팔을 머리 위로 올려 큰 하트 만들기", "머리 위 하트",
   sym_touch("top", "top", (5, 1, 0), pole=(1, 0.2, 0)), {"L": "soft", "R": "soft"},
   head={"roll": 6}, acts="gesture", domain="romance", emo="happy", intent="affection")
ub("finger_heart", UP, "가슴 앞에서 손가락 하트 만들기", "손가락 하트",
   {"R": arm(-10, -2, 30, pole=[-0.5, -1, -0.2], pron=-10)}, {"R": "pinch"}, one=True,
   head={"roll": 8}, acts="gesture", domain="romance", emo="happy", intent="affection")
ub("blow_kiss", UP, "입에 댄 손을 앞으로 보내며 손키스하기", "손키스",
   {"R": palm_at(-6, -4, 22, "h", face=[0, 0, 1], pole=[-0.6, -1, 0])}, {"R": "flat"}, one=True,
   head={"roll": 8}, acts="gesture", domain="romance", emo="happy", intent="affection")
ub("flowers", UP, "꽃다발을 양손으로 모아 쥐기", "꽃다발 들기",
   {"L": arm(3, -14, 26, pole=[0.5, -1, -0.2], pron=0), "R": arm(-3, -8, 27, pole=[-0.5, -1, -0.2], pron=0)},
   {"L": "grip", "R": "grip"}, head={"pitch": 8}, acts="hold", objs="flower", domain="romance",
   emo="shy", intent="affection")
ub("give_gift", UP, "선물을 양손으로 앞으로 내밀기", "선물 내밀기",
   sym_palm(10, -8, 44, (0, 1, 0), pole=(0.4, -1, -0.2)), {"L": "open", "R": "open"},
   spine={"pitch": 8}, acts="give,hold", objs="box", domain="romance", emo="shy", intent="affection")
ub("hug_open", UP, "안아 주려고 양팔을 앞으로 벌리기", "안으려 팔 벌리기",
   sym(22, 2, 40, pole=(1, -0.6, 0), pron=-30), {"L": "open", "R": "open"},
   acts="hug,reach", domain="romance", emo="happy", intent="affection")
ub("hair_tuck", UP, "흘러내린 머리카락을 귀 뒤로 넘기기", "머리카락 넘기기",
   {"R": touch("earR", (-1, 3, 2), pole=[-0.8, -0.6, 0.4], face=None, pron=-20)}, {"R": "soft"}, one=True,
   head={"pitch": 8, "roll": -6}, acts="groom", domain="romance", emo="shy")

# ================================================================== action / fantasy
ub("boxing_guard", STAND | {"kneel", "squat"}, "주먹을 턱 앞으로 올려 복싱 가드 자세 잡기", "복싱 가드",
   {"L": arm(9, 20, 30, pole=[0.6, -1, 0.2], pron=40), "R": arm(-7, 16, 24, pole=[-0.6, -1, 0.2], pron=40)},
   {"L": "fist", "R": "fist"}, head={"pitch": 10}, spine={"pitch": 6}, acts="guard", domain="combat",
   activity="boxing", intent="defense")
ub("punch", STAND, "한 팔로 곧게 스트레이트를 뻗기", "스트레이트 펀치",
   {"R": arm(-4, 20, 56, pole=[-0.5, -1, 0], pron=70), "L": arm(9, 20, 28, pole=[0.6, -1, 0.2], pron=40)},
   {"L": "fist", "R": "fist"}, spine={"yaw": 18, "pitch": 4}, head={"yaw": -14}, acts="punch,strike",
   domain="combat", activity="boxing", intent="attack")
ub("sword_mid", STAND_KNEEL, "양손으로 검을 쥐고 앞으로 겨누기", "양손 검 겨누기",
   {"L": arm(3, -16, 32, pole=[0.6, -1, 0.2], pron=-10), "R": arm(-2, -10, 34, pole=[-0.6, -1, 0.2], pron=-10)},
   {"L": "grip", "R": "grip"}, head={"pitch": 4}, acts="hold,guard", objs="sword", domain="combat",
   activity="swordsmanship", intent="prepare_action")
ub("sword_over", STAND, "양손으로 검을 머리 위로 치켜들기", "검 머리 위로",
   {"L": arm(4, 52, 8, pole=[1, -0.2, -0.2], pron=-10), "R": arm(-2, 58, 6, pole=[-1, -0.2, -0.2], pron=-10)},
   {"L": "grip", "R": "grip"}, head={"pitch": -4}, acts="hold,prepare", objs="sword", domain="combat",
   activity="swordsmanship", intent="attack")
ub("sword_side", STAND, "검을 한 손으로 비스듬히 아래로 늘어뜨려 쥐기", "검 내려 쥐기",
   {"R": arm(-26, -4, 12, "p", pole=[-0.3, -0.5, -1], pron=20)}, {"R": "grip"}, one=True,
   head={"pitch": 4}, acts="hold", objs="sword", domain="combat", activity="swordsmanship", intent="guard")
ub("pistol_aim", STAND | {"kneel"}, "양손으로 권총을 쥐고 앞으로 조준하기", "권총 조준",
   {"R": arm(-6, 18, 52, pole=[-0.5, -1, 0], pron=0), "L": arm(-2, 16, 46, pole=[0.6, -1, 0], pron=-20)},
   {"R": "grip", "L": "cup"}, head={"pitch": 6}, acts="aim,hold", objs="pistol", domain="combat",
   activity="shooting", intent="attack")
ub("pistol_up", STAND_KNEEL, "권총을 얼굴 옆에 세워 들고 경계하기", "권총 세워 들기",
   {"R": arm(-12, 10, 16, pole=[-0.4, -1, 0], pron=0)}, {"R": "grip"}, one=True,
   head={"yaw": 20}, acts="hold,guard", objs="pistol", domain="combat", activity="shooting", intent="guard")
ub("cast_fwd", UP, "양손바닥을 앞으로 내밀어 마법을 쏘기", "마법 쏘기",
   {"R": palm_at(-12, 8, 48, face=[0, 0, 1], pole=[-0.4, -1, 0]),
    "L": palm_at(12, 4, 42, face=[0, 0, 1], pole=[0.4, -1, 0])}, {"L": "open", "R": "open"},
   spine={"pitch": 6}, acts="cast_spell", domain="fantasy", activity="magic", intent="attack")
ub("cast_up", UP, "한 손을 하늘로 들어 마법을 모으기", "한 손 들어 마법",
   {"R": arm(-24, 46, 18, pole=[-1, -0.3, -0.3], pron=-80)}, {"R": "open"}, one=True,
   head={"pitch": -16}, acts="cast_spell,raise_hand", domain="fantasy", activity="magic", intent="prepare_action")
ub("cast_orb", UP, "가슴 앞에서 양손으로 마력 구슬을 감싸기", "마력 구슬",
   sym_palm(10, -6, 30, (-1, 0, 0), pole=(0.5, -1, -0.2)), {"L": "cup", "R": "cup"},
   head={"pitch": 16}, acts="cast_spell", domain="fantasy", activity="magic", intent="prepare_action")
ub("shield_block", STAND_KNEEL, "팔뚝에 낀 방패를 앞으로 세우고 검을 뒤로 빼기", "방패 막기",
   {"L": arm(10, 8, 30, pole=[1, -1, 0.3], pron=70), "R": arm(-26, -4, -6, pole=[-0.3, -0.6, -1], pron=20)},
   {"L": "fist", "R": "grip"}, spine={"pitch": 6}, acts="block,defend", objs="shield,sword", domain="combat",
   intent="defense")
ub("block_arms", UP, "양팔을 얼굴 앞에 세워 막기", "양팔 막기",
   {"L": arm(8, 20, 28, pole=[0.3, -1, 0.3], pron=10), "R": arm(-8, 14, 30, pole=[-0.3, -1, 0.3], pron=10)},
   {"L": "fist", "R": "fist"}, head={"pitch": 14}, spine={"pitch": 8}, acts="block,defend", domain="combat",
   intent="defense")
ub("duck_head", {"squat", "kneel", "floor"}, "양팔로 머리를 감싸 보호하기", "머리 감싸 보호",
   sym_touch("crownL", "crownR", pole=(0.6, 0, 1)), {"L": "soft", "R": "soft"}, head={"pitch": 30},
   spine={"pitch": 16}, acts="defend", emo="afraid", intent="protect")
ub("surrender", UP, "두 손을 들어 항복하기", "손 들고 항복",
   sym(30, 40, 4, pole=(1, -0.4, -0.4), pron=-80), {"L": "open", "R": "open"},
   acts="surrender", emo="afraid", intent="surrender")
ub("fight_ready", STAND | {"squat"}, "양손을 펴 앞으로 내밀고 무술 겨루기 자세", "무술 겨루기",
   {"L": palm_at(12, 14, 40, face=[0.3, 0, 1], pole=[0.5, -1, 0]),
    "R": palm_at(-8, 4, 26, face=[0.3, 0, 1], pole=[-0.5, -1, 0])}, {"L": "flat", "R": "flat"},
   spine={"pitch": 6}, acts="guard,prepare", domain="combat", activity="martial_arts", intent="prepare_action")
ub("spear_hold", STAND, "창을 양손으로 쥐고 앞으로 겨누기", "창 겨누기",
   {"L": arm(10, -10, 40, pole=[0.5, -1, 0], pron=10), "R": arm(-14, -20, 10, pole=[-0.4, -1, -0.4], pron=10)},
   {"L": "grip", "R": "grip"}, acts="hold,guard", objs="spear", domain="combat", activity="spear_fighting",
   intent="prepare_action")

# ================================================================== sports / performance
ub("basket_shoot", STAND, "공을 머리 위로 들어 슛 하기", "농구 슛",
   {"R": arm(-10, 46, 18, pole=[-1, -0.3, -0.2], pron=-60), "L": palm_at(4, 50, 26, face=[-1, 0, 0],
                                                                             pole=[1, -0.3, -0.2])},
   {"R": "cup", "L": "open"}, head={"pitch": -16}, acts="shoot_ball", objs="ball", domain="sports",
   activity="basketball", intent="focus_task")
ub("ball_chest", UP, "공을 양손으로 가슴 앞에 들기", "공 들기",
   sym_palm(13, -6, 28, (-1, 0, 0), pole=(0.5, -1, -0.2)), {"L": "cup", "R": "cup"},
   acts="hold", objs="ball", domain="sports", intent="prepare_action")
ub("ball_hip", STAND, "공을 한쪽 옆구리에 끼고 서기", "공 옆구리에 끼기",
   {"R": arm(-24, -24, 14, pole=[-0.5, -0.6, -1], pron=-60)}, {"R": "cup"}, one=True,
   acts="hold,carry", objs="ball", domain="sports", intent="relax")
ub("throw_ready", STAND, "공을 머리 뒤로 들어 던질 준비하기", "던지기 준비",
   {"R": arm(-34, 26, -14, pole=[-0.5, -1, -0.6], pron=-20), "L": arm(24, 10, 40, pole=[0.4, -1, 0], pron=20)},
   {"R": "cup", "L": "open"}, spine={"yaw": -14, "roll": -4}, head={"yaw": 12}, acts="throw,prepare",
   objs="ball", domain="sports", activity="baseball", intent="prepare_action")
ub("bat_ready", STAND, "배트를 어깨 위로 들고 타격 준비하기", "배트 타격 준비",
   {"R": arm(-14, 8, 14, pole=[-0.4, -1, -0.2], pron=-10), "L": arm(-8, 4, 18, pole=[0.6, -1, -0.2], pron=-10)},
   {"R": "grip", "L": "grip"}, head={"yaw": 30}, acts="swing,prepare", objs="bat", domain="sports",
   activity="baseball", intent="prepare_action")
ub("racket_ready", STAND, "라켓을 앞에 들고 공을 기다리기", "라켓 준비",
   {"R": arm(-22, -8, 30, pole=[-0.5, -1, -0.2], pron=10), "L": arm(10, -10, 34, pole=[0.5, -1, -0.2], pron=-30)},
   {"R": "grip", "L": "open"}, spine={"pitch": 8}, acts="prepare,hold", objs="racket", domain="sports",
   activity="tennis", intent="prepare_action")
ub("guitar", NO_SQUAT, "기타를 메고 코드를 잡으며 연주하기", "기타 연주",
   {"L": arm(30, -10, 30, pole=[0.5, -1, -0.2], pron=-60), "R": arm(-8, -26, 18, pole=[-0.5, -1, -0.2], pron=60)},
   {"L": "soft", "R": "pinch"}, head={"pitch": 16, "yaw": 14}, acts="play_music,hold", objs="guitar",
   domain="performance", activity="music", intent="express_performance")
ub("violin", STAND | {"chair"}, "바이올린을 어깨에 얹고 활을 그어 연주하기", "바이올린 연주",
   {"L": arm(24, 10, 34, pole=[0.6, -1, -0.2], pron=-60), "R": arm(-26, 4, 22, pole=[-0.6, -1, -0.4], pron=60)},
   {"L": "soft", "R": "pinch"}, head={"roll": 14, "yaw": 16, "pitch": 8}, acts="play_music,hold",
   objs="violin", domain="performance", activity="music", intent="express_performance")
ub("mic_sing", UP, "마이크를 입에 대고 노래하며 다른 팔을 뻗기", "마이크 노래",
   {"R": touch("mouth", (0, -3, 8), pole=[-0.5, -1, 0], grip=True, face=None, pron=-10),
    "L": arm(38, 10, 20, pole=[0.3, -1, -0.3], pron=-60)}, {"R": "grip", "L": "open"},
   head={"pitch": -10}, acts="play_music,hold", domain="performance", activity="music", emo="happy",
   intent="express_performance")
ub("dance_diag", STAND, "한 팔은 위로, 한 팔은 아래로 사선으로 뻗어 춤추기", "사선 댄스",
   {"L": arm(40, 38, 6, pole=[0, -1, -0.4], pron=-40), "R": arm(-42, -30, 8, pole=[0, -1, -0.4], pron=-40)},
   {"L": "open", "R": "open"}, head={"yaw": 14, "pitch": -6}, acts="dance", domain="performance",
   activity="dance", emo="happy", intent="express_performance")
ub("dance_point", STAND, "한 손을 허리에 짚고 다른 손으로 위를 찌르는 댄스", "포인트 댄스",
   {"L": touch("hipL", pole=[1, 0, -0.4]), "R": arm(-26, 54, 10, pole=[-1, 0, -0.3], pron=0)},
   {"L": "flat", "R": "point"}, head={"pitch": -10, "yaw": -10}, acts="dance,point", domain="performance",
   activity="dance", intent="express_performance")
ub("ballet_arms", STAND, "두 팔을 둥글게 머리 위로 올린 발레 자세", "발레 팔",
   sym(10, 50, 14, pole=(1, -0.2, 0.4), pron=-40), {"L": "soft", "R": "soft"}, head={"pitch": -6},
   acts="dance,balance", domain="performance", activity="ballet", intent="express_performance")
ub("balance_arms", STAND, "양팔을 옆으로 뻗어 균형 잡기", "양팔 벌려 균형",
   sym(50, 6, 2, pole=(0, -1, -0.4), pron=-20), {"L": "open", "R": "open"},
   acts="balance", intent="maintain_balance")
ub("warmup_arm", STAND, "한 팔을 반대쪽 팔로 당겨 어깨를 스트레칭하기", "어깨 스트레칭",
   {"R": arm(14, 6, 30, pole=[0.2, -1, -0.4], pron=60), "L": arm(-2, -6, 24, pole=[0.8, -1, -0.2], pron=60)},
   {"R": "relaxed", "L": "grip"}, acts="stretch", domain="sports", activity="fitness", intent="warm_up")
ub("yoga_tree", STAND, "양손을 가슴 앞에 합장하고 요가 자세", "합장 요가",
   sym_palm(1.6, -2, 22, (-1, 0, 0), pole=(0.8, -1, -0.2)), {"L": "flat", "R": "flat"},
   acts="balance", domain="sports", activity="yoga", intent="maintain_balance")

# ================================================================== lying only
ub("lie_arms_side", {"lie_back"}, "두 팔을 몸 옆에 두고 쉬기", "팔 옆에 두고",
   sym(25, -40, -3, pole=(0.3, 0, -1)), {"L": "relaxed", "R": "relaxed"}, acts="rest", intent="rest_recover")
ub("lie_belly_hands", {"lie_back"}, "두 손을 배 위에 포개 얹기", "배 위에 손",
   sym_touch("belly", "belly", (4, 4, 0), pole=(1, -0.5, 0.2)), {"L": "flat", "R": "flat"},
   acts="rest", intent="relax")
ub("lie_head_pillow", {"lie_back"}, "깍지 낀 손을 머리 밑에 받치기", "손깍지 베개",
   sym_touch("backhead", "backhead", (5, 2, 0), pole=(1, 0.2, 0.3)), {"L": "flat", "R": "flat"},
   acts="rest", intent="relax")
ub("lie_phone_up", {"lie_back"}, "양손으로 휴대폰을 얼굴 위로 들고 보기", "누워 폰 보기",
   sym_palm(5, 16, 30, (0, 0, -1), pole=(0.8, -0.5, -1)), {"L": "phone", "R": "phone"},
   head={"pitch": 12}, acts="hold,look", objs="phone", intent="focus_task")
ub("lie_arm_eyes", {"lie_back"}, "한 팔을 이마 위에 얹어 눈을 가리기", "팔로 눈 가리기",
   {"R": touch("forehead", (0, 2, 0), pole=[-1, 0.2, 0.3], face=None, pron=80),
    "L": arm(25, -40, -3, pole=[0.3, 0, -1])}, {"R": "relaxed", "L": "relaxed"}, acts="rest", emo="tired",
   intent="rest_recover")
ub("lie_starfish", {"lie_back"}, "팔을 양옆으로 활짝 뻗기", "팔 벌려 대자",
   sym(54, 26, -9, pole=(0, -1, 0.2), pron=-60), {"L": "relaxed", "R": "relaxed"}, acts="rest,stretch",
   intent="release_tension")
ub("lie_book_up", {"lie_back"}, "책을 얼굴 위로 들고 읽기", "누워 책 읽기",
   sym_palm(12, 14, 28, (-0.3, 0, -1), pole=(0.8, -0.5, -1)), {"L": "soft", "R": "soft"},
   head={"pitch": 14}, acts="read,hold", objs="book", activity="study", intent="focus_task")
ub("prone_phone", {"lie_front"}, "양손으로 휴대폰을 들고 보기", "엎드려 폰",
   sym_palm(7, 6, 26, (0, -0.3, -1), pole=(0.3, -0.3, 1)), {"L": "phone", "R": "phone"},
   acts="hold,look", objs="phone", intent="focus_task")
ub("prone_chin", {"lie_front"}, "양손으로 턱을 괴기", "엎드려 턱 괴기",
   sym_touch("chin", "chin", (4, -2.5, 2), pole=(0.3, -0.6, 1), grip=True, face=None, pron=-10),
   {"L": "soft", "R": "soft"}, head={"pitch": -6}, acts="rest,look", emo="happy", intent="relax")
ub("prone_book", {"lie_front"}, "앞에 펼친 책을 읽기", "엎드려 책 읽기",
   {"L": palm_at(12, -6, 32, face=[0, -1, 0], pole=[0.3, -0.3, 1]),
    "R": palm_at(-12, -6, 32, face=[0, -1, 0], pole=[-0.3, -0.3, 1])}, {"L": "flat", "R": "flat"},
   head={"pitch": 20}, acts="read", objs="book", activity="study", intent="focus_task")
ub("side_hand_pillow", {"lie_side"}, "손을 뺨 밑에 베기", "손 베개",
   {"R": touch("cheekR", (-3, 0, -2), pole=[0, -1, 0.6]), "L": touch("thighL", (0, 6, 0), pole=[0.8, 0, -0.4])},
   {"L": "relaxed", "R": "soft"}, acts="rest", emo="tired", intent="rest_recover")
ub("side_phone", {"lie_side"}, "휴대폰을 얼굴 앞에 들고 보기", "옆으로 누워 폰",
   {"R": touch("cheekR", (-3, 0, -2), pole=[0, -1, 0.6]),
    "L": arm(-2, 4, 30, pole=[0.6, -1, 0.2], pron=-60)}, {"L": "phone", "R": "soft"},
   acts="hold,look", objs="phone", intent="focus_task")

# ------------------------------------------------------------------ free-arm fillers (left arm)
FILL = {
    "stand": [("hang", arm(24, -4, 6, "p"), "relaxed", "다른 팔은 자연스럽게 내림"),
              ("side", arm(21.5, -6, 1, "p", pron=20), "soft", "다른 팔은 몸에 붙임"),
              ("hip", touch("hipL", pole=[1, 0, -0.4]), "flat", "다른 손은 허리에 짚음"),
              ("pocket", touch("thighL", (2.5, 16, 0), pole=[0.6, -0.2, -1]), "flat", "다른 손은 주머니에 넣음")],
    "chair": [("thigh", touch("thighL", (0, 4, 0)), "flat", "다른 손은 허벅지 위에 올림"),
              ("knee", touch("kneeL", (0, 3, 0)), "relaxed", "다른 손은 무릎 위에 올림")],
    "desk": [("thigh", touch("thighL", (0, 4, 0)), "flat", "다른 손은 허벅지 위에 올림"),
             ("desk", palm_at(16, DESK_H + 0.4, 40, "w", pole=[1, -0.5, -0.2]), "flat", "다른 손은 책상 위에 둠")],
    "floor": [("knee", touch("kneeL", (0, 2, 0)), "relaxed", "다른 손은 무릎 위에 올림"),
              ("floor", palm_at(28, 1.4, -34, "w", pole=[0.4, 0.1, -1], fdir=[0.3, 0, -1]), "flat",
               "다른 손은 뒤로 바닥을 짚음")],
    "kneel": [("thigh", touch("thighL", (0, 4, 0)), "flat", "다른 손은 허벅지 위에 올림"),
              ("hang", arm(24, -4, 6, "p"), "relaxed", "다른 팔은 자연스럽게 내림")],
    "squat": [("knee", touch("kneeL", (0, 3, 0)), "relaxed", "다른 손은 무릎 위에 올림"),
              ("forearm", arm(10, 14, 52, "p", pole=[0.2, -1, 0], pron=60), "relaxed", "다른 팔은 무릎에 걸침")],
}

HEADS = [("", {}), ("고개를 숙임", {"pitch": 16}), ("위를 올려다봄", {"pitch": -18}),
         ("옆을 돌아봄", {"yaw": 35}), ("반대쪽을 돌아봄", {"yaw": -35}), ("고개를 갸웃함", {"roll": 12})]
