"""04J-286: Gemini 스켈레톤 보완(Hybrid Pose v3.2, PR #82)을 /analyze 경로에 연결.

위치: 슬롯 배정 → crop 재추론 → Human-Art 폴백 **다음**, 라우팅(관절 충분 판정) **앞**.
정상(valid)이 아닌 VLM 슬롯이 하나라도 있으면 컷당 Gemini를 **최대 1회** 부른다.
모두 정상이면 부르지 않는다(시간 0).

규칙(#82 계약을 그대로 따른다):
- 입력은 원본 러프 + 현재 관절 표시 이미지 + 관절 JSON(0~1000 좌표). 응답은
  `merge_single_response`로 검증하고, 지적된 관절과 같은 팔/다리 사슬만 바꾼다.
- 수정하지 않은 관절·점수는 그대로. 실패/형식 오류면 기존 관절을 그대로 쓴다(재시도 없음).
- Gemini가 "누락 인물"로 새로 그린 사람은 스켈레톤이 없는 슬롯(VLM 박스)과 겹칠 때만 붙인다.
- 보완된 슬롯은 `skeleton_source="vlm_repair"` → refine 금지(구조 정책이 full_image만 허용),
  자동 high 신뢰도 금지. 검색 Top-5 계산식·임계값은 바꾸지 않는다.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from .config import CFG
from .schema import COCO17, Skeleton
from .skeleton_extraction import finalize_slot

REPAIR_SOURCE = "vlm_repair"
VERSION = "skeleton-repair-v1(hybrid-v3.2-single)"


def _env_mode() -> str:
    value = os.getenv("SKELETON_REPAIR", "auto").strip().lower()
    return value if value in {"auto", "off", "all"} else "auto"


@dataclass(frozen=True)
class RepairConfig:
    # auto: 정상 아닌 VLM 슬롯이 있을 때만 / all: 항상 / off: 안 함
    mode: str = "auto"
    model: str = ""
    thinking_level: str = "low"
    timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "RepairConfig":
        thinking = os.getenv("SKELETON_REPAIR_THINKING", "low").strip().lower()
        return cls(
            mode=_env_mode(),
            model=os.getenv("SKELETON_REPAIR_MODEL", "").strip() or (
                CFG.openai_model if CFG.vlm_provider == "openai" else CFG.gemini_model),
            thinking_level=thinking if thinking in {"low", "default"} else "low",
            timeout_s=float(os.getenv("SKELETON_REPAIR_TIMEOUT_S", "30")),
        )


def build_repairer(vlm_client=None, config: Optional[RepairConfig] = None):
    """Gemini 분석을 쓰는 운영 구성에서만 켠다. mock VLM이면 None(기존 경로 그대로)."""
    config = config or RepairConfig.from_env()
    provider = CFG.vlm_provider.lower()
    if config.mode == "off" or provider not in {"gemini", "openai"}:
        return None
    if vlm_client is not None and "mock" in type(vlm_client).__name__.lower():
        return None
    key = os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY", "")
    if not key:
        return None
    from .hybrid_pose.gemini import GeminiReviewer
    if provider == "openai":
        from .hybrid_pose.openai import OpenAIReviewer
        reviewer_class = OpenAIReviewer
    else:
        reviewer_class = GeminiReviewer
    reviewer = reviewer_class(key, config.model, timeout=config.timeout_s,
                              thinking_level=config.thinking_level)
    return SkeletonRepairer(reviewer, config)


def _to_pil(image):
    from PIL import Image
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as src:
            return src.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]          # 파이프라인 ndarray 규약은 BGR
        return Image.fromarray(np.ascontiguousarray(arr).astype(np.uint8)).convert("RGB")
    return image.convert("RGB")


def _box1000(box, w, h):
    if box is None:
        return None
    x1, y1, x2, y2 = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
    clip = lambda v, s: float(np.clip(v / s * 1000.0, 0, 1000))  # noqa: E731
    return {"x1": clip(x1, w), "y1": clip(y1, h), "x2": clip(x2, w), "y2": clip(y2, h)}


def slots_to_prediction(slots, img_w, img_h, threshold):
    """슬롯 순서 그대로 person_index를 매긴다(스켈레톤 없는 슬롯은 빈 관절 + VLM 박스)."""
    people = []
    for index, slot in enumerate(slots):
        joints = []
        if slot.skeleton is not None:
            pts = np.asarray(slot.skeleton.keypoints, float).reshape(17, 2)
            scores = np.asarray(slot.skeleton.scores, float).reshape(17)
        else:
            pts, scores = np.full((17, 2), np.nan), np.zeros(17)
        for j, name in enumerate(COCO17):
            ok = bool(np.isfinite(pts[j]).all() and np.isfinite(scores[j]) and scores[j] >= threshold)
            joints.append({
                "name": name,
                "x": float(np.clip(pts[j, 0] / img_w * 1000, 0, 1000)) if ok else None,
                "y": float(np.clip(pts[j, 1] / img_h * 1000, 0, 1000)) if ok else None,
                "state": "visible" if ok else "not_present",
                "score": float(scores[j]) if np.isfinite(scores[j]) else None,
            })
        located = [p for p in joints if p["state"] != "not_present"]
        if located:
            bbox = {"x1": min(p["x"] for p in located), "y1": min(p["y"] for p in located),
                    "x2": max(p["x"] for p in located), "y2": max(p["y"] for p in located)}
        else:
            bbox = _box1000(slot.vlm_box, img_w, img_h) or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        people.append({"person_index": index, "bbox": bbox, "keypoints": joints})
    return {"people": people}


def _iou(a, b):
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    area = lambda r: max(0.0, r["x2"] - r["x1"]) * max(0.0, r["y2"] - r["y1"])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _skeleton_from(person, old: Optional[Skeleton], img_w, img_h, repaired_score):
    pts = (np.asarray(old.keypoints, float).reshape(17, 2).copy() if old is not None
           else np.zeros((17, 2)))
    scores = (np.asarray(old.scores, float).reshape(17).copy() if old is not None
              else np.zeros(17))
    changed = []
    for j, joint in enumerate(person["keypoints"]):
        if "score" in joint and old is not None:
            continue                       # 수정 안 된 관절(점수 보존) → 그대로
        if joint["state"] == "not_present":
            if old is not None and scores[j] > 0:
                scores[j] = 0.0
                changed.append(COCO17[j])
            continue
        pts[j] = (joint["x"] / 1000.0 * img_w, joint["y"] / 1000.0 * img_h)
        scores[j] = repaired_score
        changed.append(COCO17[j])
    return Skeleton(keypoints=pts.astype(np.float32), scores=scores.astype(np.float32)), changed


class SkeletonRepairer:
    def __init__(self, reviewer, config: RepairConfig):
        self.reviewer = reviewer
        self.config = config

    def needs(self, slots) -> list[int]:
        if self.config.mode == "all":
            return [i for i, s in enumerate(slots) if s.slot_origin == "vlm"]
        return [i for i, s in enumerate(slots)
                if s.slot_origin == "vlm" and s.state != "valid"]

    def repair(self, image, slots, img_w, img_h, cfg=CFG) -> dict:
        """슬롯을 제자리에서 보완한다. 반환: 기록용 trace(dict)."""
        from .hybrid_pose.contracts import merge_single_response
        from .hybrid_pose.render import context, overlay, png_bytes

        targets = self.needs(slots)
        trace = {"version": VERSION, "mode": self.config.mode, "model": self.config.model,
                 "thinking_level": self.config.thinking_level, "called": False,
                 "targets": targets, "status": "skipped", "elapsed_ms": 0.0,
                 "repaired_slots": [], "added_people_matched": 0, "notes": []}
        if not targets:
            return trace
        started = perf_counter()
        threshold = float(cfg.skeleton_kpt_threshold)
        # 보완 관절 점수: 임계값 바로 위(사용은 하되 '높은 확신'으로 보이지 않게).
        repaired_score = float(max(threshold, getattr(cfg, "min_skeleton_score", 0.0)) + 0.05)
        try:
            pil = _to_pil(image)
            w, h = pil.size
            sx, sy = w / float(img_w), h / float(img_h)
            if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
                pil = pil.resize((int(img_w), int(img_h)))
            base = slots_to_prediction(slots, img_w, img_h, threshold)
            trace["called"] = True
            response, _raw = self.reviewer.request(
                "single_review_repair", png_bytes(pil), png_bytes(overlay(pil, base)),
                context(base, img_w, img_h))
            try:
                final, audit = merge_single_response(base, response)
                decision = response["evaluation"]["decision"]
            except ValueError as whole_error:
                # 응답 전체가 계약에 안 맞으면, 지적(issue)마다 같은 엄격 규칙으로 따로 검증해
                # 통과한 것만 쓴다. 규칙 자체는 완화하지 않는다(통과 못 한 지적은 버림).
                final, audit, kept, dropped = salvage_by_issue(base, response)
                trace["whole_response_rejected"] = str(whole_error)[:120]
                trace["issues_kept"], trace["issues_dropped"] = kept, dropped
                if not kept:
                    raise
                decision = "repair_partial"
        except Exception as error:  # noqa: BLE001 - 실패 시 기존 관절 유지
            from .hybrid_pose.gemini import GeminiStageError
            # 계약 검증 오류 문구는 코드에 고정된 문자열이라 기록해도 안전하다.
            safe = isinstance(error, (GeminiStageError, ValueError)) and not isinstance(
                error, UnicodeError)
            trace.update(status="error", error=type(error).__name__,
                         message=str(error)[:160] if safe else None,
                         elapsed_ms=round((perf_counter() - started) * 1000, 1))
            return trace
        trace["decision"] = decision
        trace["issues"] = [{"type": i.get("type"), "person_index": i.get("person_index"),
                            "joints": i.get("joint_names")}
                           for i in (response.get("evaluation") or {}).get("issues") or []
                           if isinstance(i, dict)]
        trace["unresolved"] = len(audit.get("unresolved") or [])
        by_index = {p["person_index"]: p for p in final["people"]}
        n = len(slots)
        # 1) 기존 인물 관절 수정
        edited = {c["person_index"] for c in audit["changes"] if c.get("kind") == "joint"}
        for index in sorted(edited):
            if index >= n or index not in by_index:
                continue
            self._apply(slots[index], by_index[index], img_w, img_h, repaired_score, cfg, trace,
                        "edit")
        # 2) 새 인물 → 스켈레톤 없는 슬롯에 배정. VLM 박스가 있으면 겹침(IoU≥0.1),
        #    박스가 없는 빈 슬롯(분석이 박스를 못 준 컷)은 남은 새 인물을 왼쪽부터 차례로 붙인다.
        boxed = [i for i, s in enumerate(slots) if s.skeleton is None and s.vlm_box is not None]
        unboxed = [i for i, s in enumerate(slots) if s.skeleton is None and s.vlm_box is None]
        adds = [by_index.get(c["final_person_index"]) for c in audit["changes"]
                if c.get("kind") == "add_person"]
        adds = sorted((a for a in adds if a is not None), key=lambda a: a["bbox"]["x1"])
        for person in adds:
            slot_index = None
            if boxed:
                best, cand = max((_iou(person["bbox"], _box1000(slots[i].vlm_box, img_w, img_h)), i)
                                 for i in boxed)
                if best >= 0.1:
                    slot_index = cand
                    boxed.remove(cand)
            if slot_index is None and unboxed:
                slot_index = unboxed.pop(0)
            if slot_index is None:
                trace["notes"].append("added_person_unmatched")
                continue
            fresh = copy.deepcopy(person)
            for joint in fresh["keypoints"]:
                joint.pop("score", None)
            self._apply(slots[slot_index], fresh, img_w, img_h, repaired_score, cfg, trace, "add")
            trace["added_people_matched"] += 1
        removed = [c["person_index"] for c in audit["changes"] if c.get("kind") == "remove_person"]
        if removed:
            # VLM 슬롯 수는 분석 결과가 정한다. 지우지 않고 기록만 한다.
            trace["notes"].append("remove_suggested:" + ",".join(map(str, removed)))
        trace.update(status="ok", elapsed_ms=round((perf_counter() - started) * 1000, 1))
        return trace

    @staticmethod
    def _apply(slot, person, img_w, img_h, repaired_score, cfg, trace, kind):
        before = slot.state
        skeleton, changed = _skeleton_from(person, slot.skeleton, img_w, img_h, repaired_score)
        if not changed:
            return
        saved = {name: getattr(slot, name) for name in _SLOT_FIELDS}
        saved["reasons"] = list(slot.reasons)
        slot.skeleton = skeleton
        slot.skeleton_box = None
        slot.evidence = None
        slot.state = "missing"            # finalize가 새 관절로 상태를 다시 계산
        slot.skeleton_source = REPAIR_SOURCE
        slot.reasons.append(f"vlm_repair:{kind}")
        finalize_slot(slot, cfg)
        if _RANK.get(slot.state, 9) > _RANK.get(before, 9):
            # 좋아지거나, 그대로: 보완 후 상태가 더 나빠지면 원래 관절로 되돌린다.
            would_be = slot.state
            for name, value in saved.items():
                setattr(slot, name, value)
            trace.setdefault("reverted_slots", []).append(
                {"slot_id": slot.slot_id, "state_before": before, "would_be": would_be})
            return
        trace["repaired_slots"].append({"slot_id": slot.slot_id, "kind": kind,
                                        "joints": changed, "state_before": before,
                                        "state_after": slot.state})


_SLOT_FIELDS = ("skeleton", "skeleton_box", "evidence", "state", "skeleton_source", "reasons")
_RANK = {"valid": 0, "partial": 1, "suspect": 2, "invalid": 3, "missing": 4}


def salvage_by_issue(base, response):
    """지적마다 `merge_repair`(엄격 규칙 그대로)를 따로 돌려 통과한 것만 순서대로 합친다."""
    from .hybrid_pose.contracts import merge_repair
    evaluation = response.get("evaluation") if isinstance(response, dict) else None
    patch = response.get("patch") if isinstance(response, dict) else None
    issues = (evaluation or {}).get("issues") if isinstance(evaluation, dict) else None
    current, changes, kept, dropped = base, [], [], []
    if not isinstance(issues, list) or not isinstance(patch, dict):
        return current, {"changes": [], "unresolved": [], "uncertainties": []}, kept, dropped
    for issue in issues:
        iid = issue.get("issue_id") if isinstance(issue, dict) else None
        sub_patch = {k: [x for x in (patch.get(k) or []) if isinstance(x, dict) and x.get("issue_id") == iid]
                     for k in ("edits", "added_people", "removed_people", "unresolved")}
        try:
            merged, audit = merge_repair(
                current, {"decision": "repair", "issues": [issue], "uncertainties": []}, sub_patch)
        except Exception as error:  # noqa: BLE001 - 이 지적만 버림
            dropped.append({"issue_id": str(iid)[:40], "reason": str(error)[:80]
                            if isinstance(error, ValueError) else type(error).__name__})
            continue
        current = merged
        changes.extend(audit["changes"])
        kept.append(str(iid)[:40])
    return current, {"changes": changes, "unresolved": [], "uncertainties": []}, kept, dropped
