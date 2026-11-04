"""Fresh scene evidence. No saved model answers, per-cut labels or pose IDs."""

from pathlib import Path
import hashlib, json, math, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURFACE = """각 인물 P의 몸통과 머리를 따로 관찰하세요. 원본 그림에서 실제로 보이는 가슴·등·어깨·옷의 단서를 설명하세요. 가슴 정면이 보이는지, 옆면인지, 등인지 먼저 판단하고 그 다음 방향을 쓰세요. 얼굴 옆모습과 몸통 정면을 혼동하지 마세요. JSON people 배열로 답하세요. 각 항목에는 person_index, torso_surface(front/back/side), torso_heading(front/front_left/front_right/left/right/back_left/back_right/back/unknown), torso_evidence(실제 관찰한 문장), head_heading(같은 방향 종류), head_pitch(level/down/up/unknown), head_bbox({left,top,right,bottom}, 머리카락~턱의 보이는 전체 머리 사각형, 이미지 0~1000 좌표), lowest_anatomy(그 인물의 가장 아래 보이는 신체 부위), scope_evidence(실제 옷·다리·신체 근거)를 넣으세요. 신체가 잘렸으면 정직하게 설명하세요."""
ROLES = """말풍선은 읽지 말고 세부 동작을 보세요. 각 사람의 몸·시선·팔이 누구를 향하는지 설명하세요. 가운데 사람이 두 사람 사이에 서서 팔로 막는 동작이 있는지도 실제로 확인하세요. 팔을 뻗은 사람이 물건을 쥐고 있다면 제지 동작으로 바꾸지 마세요. JSON으로 scene_action(관찰한 장면 한 문장), people 배열(person_index, action, target_person_indices), relations 배열(actor, target, type, evidence)을 반환하세요. type은 opposing/restraining/reaching_toward/looking_at/leaving/unknown 중 근거가 맞는 것을 고르세요. 실제로 안 보이는 관계는 만들지 마세요."""
ACTOR = """Analyze only P{index}, identified by its red HEAD marker, using the ENTIRE viewport. Trace shoulder, sleeve and hand ownership through occlusion. Detached foreground hands can belong to a foreground head. Judge chest/back independently of nose direction. Do not use dialogue as pose evidence. Return JSON only with person_index, torso_heading, torso_surface, torso_confidence, head_heading, head_pitch, head_confidence, posture(standing/sitting/kneeling/crouching/unknown), torso_lean, lowest_visible_anatomy(head/chest/waist/hips/thighs/knees/shins/feet/unknown), visible_regions, body_scope(full/half/bust/head/unknown), body_bbox(object with left,top,right,bottom, viewport 0..1000 coordinates), arms(array), search_query_ko. For each visible arm include shoulder_side_in_image, shoulder, elbow, hand(0..1000 pairs or null), visible, hand_state, action, target_person_index, confidence. Include only anatomy owned by this person. A clipped head does not imply that its detached hand should be cropped away."""
HAND = """Find the largest OPEN HUMAN HAND in this crop, not a face, sleeve, bag, rectangle or text. Return JSON visible, open_palm, x,y(palm center 0..1000 in THIS crop), bbox(object with left,top,right,bottom for fingers and palm), wrist_x,wrist_y,description. If no open hand is drawn, visible=false."""
CLASSES = {
    "surface": {
        "front": "the front of a person, chest and front of shoulders facing the viewer",
        "back": "the back of a person, rear shoulders and back of shirt facing the viewer",
        "left": "a person seen from the side, facing left",
        "right": "a person seen from the side, facing right",
    },
    "action": {
        "phone": "a person looking down and holding a mobile phone with both hands",
        "reach": "a person extending an open hand toward someone walking away",
        "restrain": "a person standing between two other people with arms spread to stop both of them",
        "hip": "a person standing with hands on hips facing someone",
        "neutral": "a person standing upright with arms resting down",
        "away": "a person walking away viewed from behind holding a bag",
    },
    "range": {
        "full": "a whole person whose hips and two legs and feet are drawn",
        "half": "a partial person whose head torso and arms are drawn without visible legs",
        "bust": "a cropped head neck and shoulder without arms or hands",
    },
}


class Reader:
    def __init__(self, provider="gemini", model=None):
        self.provider = provider
        if provider == "mlx":
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            self.model_name = model or "mlx-community/Qwen3-VL-8B-Instruct-4bit"
            self.model, self.processor = load(self.model_name)
            self.cfg = load_config(self.model_name)
        elif provider in {"gemini", "openai"}:
            from src.vlm.client import GeminiVLMClient, OpenAIVLMClient

            adapter = OpenAIVLMClient if provider == "openai" else GeminiVLMClient
            self.adapter = adapter(model=model)
            self.model_name = self.adapter._model
        else:
            raise ValueError("scene evidence requires a real configured provider")

    def ask(self, path, prompt, schema=None):
        if self.provider == "mlx":
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            raw = generate(
                self.model,
                self.processor,
                apply_chat_template(self.processor, self.cfg, prompt, num_images=1),
                [str(path)],
                max_tokens=1500,
                temperature=0.0,
                repetition_penalty=1.10,
                verbose=False,
            ).text
        else:
            from google.genai import types
            from src.vlm.client import (
                _gemini_attempt_options,
                _http_status,
                _is_timeout,
            )

            for attempt in range(2):
                left = (
                    getattr(self, "deadline", time.monotonic() + 120) - time.monotonic()
                )
                if left <= 0:
                    raise TimeoutError("scene evidence budget exhausted")
                try:
                    answer = self.adapter._client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, self.adapter._to_part(str(path))],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_json_schema=schema,
                            http_options=_gemini_attempt_options(
                                types, min(60000, max(1, int(left * 1000)))
                            ),
                        ),
                    )
                    raw = answer.text or ""
                    break
                except Exception as exc:
                    if attempt or not (
                        _http_status(exc) in (500, 502, 503, 504) or _is_timeout(exc)
                    ):
                        raise
                    print(
                        "SCENE_EVIDENCE_TRANSIENT_RETRY", type(exc).__name__, flush=True
                    )
                    time.sleep(0.5)
        starts = [raw.find(c) for c in ("{", "[") if raw.find(c) >= 0]
        if not starts:
            raise ValueError("model returned no JSON")
        value, _ = json.JSONDecoder().raw_decode(raw[min(starts) :])
        if isinstance(value, list):
            value = {"people": value}
        if not isinstance(value, dict):
            raise ValueError("model returned invalid JSON object")

        def boxes(v):
            if isinstance(v, list):
                return [boxes(x) for x in v]
            if isinstance(v, dict):
                return {
                    k: (
                        [x["left"], x["top"], x["right"], x["bottom"]]
                        if k in ("head_bbox", "body_bbox", "bbox")
                        and isinstance(x, dict)
                        and set(("left", "top", "right", "bottom")) <= x.keys()
                        else boxes(x)
                    )
                    for k, x in v.items()
                }
            return v

        value = boxes(value)
        return dict(
            analysis=value,
            coordinate_contract="xyxy-1000",
            raw=raw,
            prompt=prompt,
            model=self.model_name,
            created_ns=time.time_ns(),
            manual_inputs=False,
        )


class VisualEvidence:
    def __init__(self, pose_manifest, clip_model="openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from src.pose_humanart import HumanArtPoseModel

        self.torch = torch
        self.clip = CLIPModel.from_pretrained(
            clip_model, use_safetensors=False, local_files_only=True
        ).eval()
        self.processor = CLIPProcessor.from_pretrained(
            clip_model, local_files_only=True
        )
        self.pose = HumanArtPoseModel(
            str(pose_manifest), calibration_owner="manifest-rescue"
        )

    def features(self, **args):
        with self.torch.inference_mode():
            result = (
                self.clip.get_text_features
                if "text" in args
                else self.clip.get_image_features
            )(
                **self.processor(
                    **args,
                    return_tensors="pt",
                    **({"padding": True} if "text" in args else {}),
                )
            )
            if not isinstance(result, self.torch.Tensor):
                result = result.pooler_output
            return result / result.norm(dim=-1, keepdim=True)

    def contrastive(self, image, people):
        text = {
            k: self.features(
                text=[
                    "A black and white storyboard sketch of " + s + "."
                    for s in labels.values()
                ]
            )
            for k, labels in CLASSES.items()
        }
        rows = []
        for p in people:
            crop = image.crop(tuple(round(x) for x in p["bbox"]))
            canvas = Image.new("RGB", (max(crop.size),) * 2, "white")
            canvas.paste(
                crop,
                ((canvas.width - crop.width) // 2, (canvas.height - crop.height) // 2),
            )
            z = self.features(images=[canvas])
            rankings = {}
            for k, labels in CLASSES.items():
                scores = (z @ text[k].T).mean(0).tolist()
                rankings[k] = sorted(
                    [dict(label=a, similarity=b) for a, b in zip(labels, scores)],
                    key=lambda x: -x["similarity"],
                )
            rows.append(dict(person_index=p["person_index"], rankings=rankings))
        return dict(people=rows, classes=CLASSES)

    def proposals(self, image, viewport):
        captions = [
            "a close-up of a human hand with open fingers and a wrist",
            "a human head or face or hair",
            "a sleeve or arm with no hand",
            "a human torso with clothing",
            "a mobile phone held in both hands",
            "a door or a bag or furniture",
            "printed Korean words and a speech balloon",
        ]
        text = self.features(
            text=["A black and white line drawing of " + s + "." for s in captions]
        )
        scene = image.crop(viewport)
        w, h = scene.size
        boxes = []
        for fraction in (0.22, 0.33, 0.45):
            side = round(w * fraction)
            for y in np.linspace(0, max(0, h - side), 7):
                for x in np.linspace(0, max(0, w - side), 7):
                    boxes.append(
                        [int(x), int(y), min(w, int(x) + side), min(h, int(y) + side)]
                    )
        found = []
        for offset in range(0, len(boxes), 24):
            subset = boxes[offset : offset + 24]
            scores = (
                self.features(images=[scene.crop(b) for b in subset]) @ text.T
            ).numpy()
            for box, scores in zip(subset, scores):
                margin = float(scores[0] - max(scores[1:]))
                if margin > 0.02:
                    found.append(
                        dict(
                            bbox=(np.array(box) + np.tile(viewport[:2], 2)).tolist(),
                            margin=margin,
                        )
                    )

        def overlap(a, b):
            size = np.maximum(
                np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0
            ).prod()
            return size / max(
                1,
                min(
                    np.prod(np.subtract(a[2:], a[:2])),
                    np.prod(np.subtract(b[2:], b[:2])),
                ),
            )

        picked = []
        for p in sorted(found, key=lambda x: -x["margin"]):
            if all(overlap(p["bbox"], q["bbox"]) < 0.45 for q in picked):
                picked.append(p)
            if len(picked) == 4:
                break
        return picked

    def joints(self, image, people, actor_answers, viewport):
        import cv2

        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        w, h = image.size
        vp = np.array(viewport)
        rows = []
        for p in people:
            i = p["person_index"]
            a = actor_answers[i]["analysis"]
            boxes = [np.array(p["bbox"], float)]
            if isinstance(a.get("body_bbox"), list) and len(a["body_bbox"]) == 4:
                b = np.array(a["body_bbox"], float) * np.tile(
                    vp[2:] - vp[:2], 2
                ) / 1000 + np.tile(vp[:2], 2)
                if np.isfinite(b).all() and (b[2:] > b[:2]).all():
                    boxes.append(b)
            candidates = []
            for box in boxes:
                pose = self.pose.estimate_with_detections(
                    im, np.array([box], np.float32), w, h
                )[0]
                kp = pose.keypoints
                sc = pose.scores
                head = (
                    np.mean(kp[:5][sc[:5] > 0.35], 0)
                    if (sc[:5] > 0.35).any()
                    else kp[0]
                )
                candidates.append(
                    dict(
                        bbox=box.tolist(),
                        keypoints=kp.tolist(),
                        scores=sc.tolist(),
                        head_distance_pixels=float(
                            np.linalg.norm(head - p["head_center"])
                        ),
                    )
                )
            chosen = min(candidates, key=lambda x: x["head_distance_pixels"])
            chosen.update(
                person_index=i,
                owned_head_match=bool(
                    chosen["head_distance_pixels"]
                    < 0.20 * np.linalg.norm(np.array(p["bbox"][2:]) - p["bbox"][:2])
                ),
            )
            rows.append(chosen)
        return dict(people=rows, model=self.pose.runtime_identity())


def run(ctx, reader, visual):
    reader.deadline = time.monotonic() + 300
    for item in json.loads((ctx.work / "inputs.json").read_text()):
        folder = ctx.work / item["cut_id"]
        folder.mkdir(exist_ok=True)
        image = Image.open(item["image_path"]).convert("RGB")
        assert (
            hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
            == item["image_sha256"]
        )
        people = json.loads(
            (ctx.baseline / item["cut_id"] / "people.json").read_text()
        )["people"]
        vp = item.get("viewport", [0, 0, *image.size])
        view = image.crop(vp)
        sx, sy = view.size
        draw = ImageDraw.Draw(view)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                max(22, round(max(view.size) * 0.035)),
            )
        except OSError:
            try:
                font = ImageFont.load_default(size=28)
            except TypeError:
                font = ImageFont.load_default()
        identities = []
        for p in people:
            x, y = np.array(p["head_center"]) - vp[:2]
            label = "P" + str(p["person_index"])
            draw.rectangle((x - 10, y - 10, x + 10, y + 10), outline="red", width=3)
            draw.text(
                (x + 12, y - 12),
                label,
                fill="red",
                font=font,
                stroke_width=2,
                stroke_fill="white",
            )
            identities.append(
                dict(
                    person_index=p["person_index"],
                    head_marker=(np.array([x / sx, y / sy]) * 1000).tolist(),
                    body_bbox=(
                        (np.array(p["bbox"]) - np.tile(vp[:2], 2))
                        / np.tile(np.subtract(vp[2:], vp[:2]), 2)
                        * 1000
                    ).tolist(),
                )
            )
        identity = (
            "Use exactly these person identities, with the integer IDs shown. Do not renumber from 1. Do not add people. "
            + json.dumps(identities)
            + "\n"
        )
        surface_schema = {
            "type": "object",
            "required": ["people"],
            "properties": {
                "people": {
                    "type": "array",
                    "minItems": len(people),
                    "maxItems": len(people),
                    "items": {
                        "type": "object",
                        "required": [
                            "person_index",
                            "torso_surface",
                            "torso_heading",
                            "head_heading",
                            "head_bbox",
                            "lowest_anatomy",
                            "scope_evidence",
                        ],
                        "properties": {
                            "person_index": {
                                "type": "integer",
                                "enum": [p["person_index"] for p in people],
                            },
                            "torso_surface": {
                                "type": "string",
                                "enum": ["front", "back", "side"],
                            },
                            "torso_heading": {
                                "type": "string",
                                "enum": [
                                    "front",
                                    "front_left",
                                    "front_right",
                                    "left",
                                    "right",
                                    "back_left",
                                    "back_right",
                                    "back",
                                    "unknown",
                                ],
                            },
                            "head_heading": {
                                "type": "string",
                                "enum": [
                                    "front",
                                    "front_left",
                                    "front_right",
                                    "left",
                                    "right",
                                    "back_left",
                                    "back_right",
                                    "back",
                                    "unknown",
                                ],
                            },
                            "torso_evidence": {"type": "string"},
                            "head_pitch": {"type": "string"},
                            "head_bbox": {
                                "type": "object",
                                "required": ["left", "top", "right", "bottom"],
                                "properties": {
                                    k: {"type": "number", "minimum": 0, "maximum": 1000}
                                    for k in ("left", "top", "right", "bottom")
                                },
                            },
                            "lowest_anatomy": {"type": "string"},
                            "scope_evidence": {"type": "string"},
                        },
                    },
                }
            },
        }
        view = view.resize(
            (round(sx * 1300 / max(sx, sy)), round(sy * 1300 / max(sx, sy)))
        )
        view.save(folder / "scene-context.png")
        actors = {}
        for name, prompt in [("surfaces-review", SURFACE), ("roles-review", ROLES)]:
            result = reader.ask(
                folder / "scene-context.png",
                identity + prompt,
                surface_schema if name == "surfaces-review" else None,
            )
            result["image_sha256"] = item["image_sha256"]
            (folder / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            print("FRESH", item["cut_id"], name, flush=True)
        actor_schema = {
            "type": "object",
            "required": [
                "person_index",
                "posture",
                "lowest_visible_anatomy",
                "body_bbox",
            ],
            "properties": {
                "person_index": {
                    "type": "integer",
                    "enum": [p["person_index"] for p in people],
                },
                "posture": {
                    "type": "string",
                    "enum": ["standing", "sitting", "kneeling", "crouching", "unknown"],
                },
                "lowest_visible_anatomy": {
                    "type": "string",
                    "enum": [
                        "head",
                        "chest",
                        "waist",
                        "hips",
                        "thighs",
                        "knees",
                        "shins",
                        "feet",
                        "unknown",
                    ],
                },
                "body_bbox": {
                    "type": "object",
                    "required": ["left", "top", "right", "bottom"],
                    "properties": {
                        k: {"type": "number", "minimum": 0, "maximum": 1000}
                        for k in ("left", "top", "right", "bottom")
                    },
                },
                "action_evidence": {"type": "string"},
            },
        }
        for p in people:
            i = p["person_index"]
            result = reader.ask(
                folder / "scene-context.png",
                identity + ACTOR.format(index=i),
                actor_schema,
            )
            actors[i] = result
            (folder / f"person_{i}-analysis.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            print("FRESH", item["cut_id"], "actor", i, flush=True)
        (folder / "identities.json").write_text(
            json.dumps(
                dict(
                    analysis_viewport_bbox=vp,
                    people=[
                        dict(
                            person_index=p["person_index"], head_center=p["head_center"]
                        )
                        for p in people
                    ],
                )
            )
        )
        (folder / "contrastive.json").write_text(
            json.dumps(visual.contrastive(image, people), indent=2)
        )
        joint_people = [dict(p) for p in people]
        surface = json.loads((folder / "surfaces-review.json").read_text())["analysis"][
            "people"
        ]
        from .plan import pid, rect, mid

        by_id = {pid(a["person_index"]): a for a in surface}
        for p in joint_people:
            if not p.get("head_reference_reliable", True):
                p["head_center"] = mid(
                    rect(by_id[p["person_index"]]["head_bbox"], np.array(vp))
                ).tolist()
        (folder / "fresh-keypoints.json").write_text(
            json.dumps(visual.joints(image, joint_people, actors, vp), indent=2)
        )
        hands = []
        for j, proposal in enumerate(visual.proposals(image, vp)):
            b = proposal["bbox"]
            pad = 0.8 * max(b[2] - b[0], b[3] - b[1])
            crop = [
                max(0, round(b[0] - pad)),
                max(0, round(b[1] - pad)),
                min(image.width, round(b[2] + pad)),
                min(image.height, round(b[3] + pad)),
            ]
            hand = image.crop(crop)
            hand = hand.resize(
                tuple(round(v * 1000 / max(hand.size)) for v in hand.size)
            )
            path = folder / f"hand-{j}.png"
            hand.save(path)
            hand_schema = {
                "type": "object",
                "required": ["visible"],
                "properties": {
                    "visible": {"type": "boolean"},
                    "open_palm": {"type": "boolean"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "bbox": {
                        "type": "object",
                        "required": ["left", "top", "right", "bottom"],
                        "properties": {
                            k: {"type": "number", "minimum": 0, "maximum": 1000}
                            for k in ("left", "top", "right", "bottom")
                        },
                    },
                    "description": {"type": "string"},
                },
            }
            result = reader.ask(path, HAND, hand_schema)
            result["crop_bbox"] = crop
            hands.append(result)
        (folder / "localized-hands.json").write_text(
            json.dumps(
                dict(
                    hands=hands, image_sha256=item["image_sha256"], manual_inputs=False
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        print("FRESH", item["cut_id"], "visual and hands", len(hands), flush=True)
