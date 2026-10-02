"""Read-only checks shared by the CLI, review API, and publication gate."""

from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np

from src.bvh import parse_bvh
from ..audit import inspect_bvh
from ..anatomy import inspect as inspect_capsules
from ..storage import sha256, read_json, contained_path
from ..review.selection import decision
from . import policy


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def preview_issues(pose):
    """Verify actual pixels and their renderer's BVH/character identity."""
    meta = pose.metadata
    issues = []
    if meta.get("preview_kind") != "character" or set(pose.thumbnails) != set(
        policy.VIEWS
    ):
        return ["4방향 캐릭터 미리보기가 필요합니다."]
    for view, path in pose.thumbnails.items():
        if not path.is_file() or sha256(path) != meta.get("thumbnail_versions", {}).get(
            view
        ):
            issues.append(f"{view} 미리보기 파일이 누락되거나 변경됐습니다.")
    preview = meta.get("preview", {})
    if preview.get("character_sha256") != policy.CHARACTER_SHA256:
        issues.append("등록된 Master V2 캐릭터의 렌더 증거가 필요합니다.")
    # The catalog supplies the actual manifest directory, including when the
    # BVH is nested under hands/<preset>/. Keep reports inside that directory.
    try:
        report_path = contained_path(Path(meta["batch_directory"]), preview["report"])
        report = read_json(report_path)
        if not report.get("ok") or report.get("bvh_sha256") != pose.content_hash:
            issues.append("미리보기와 현재 BVH 버전이 일치하지 않습니다.")
        if report.get("render_fingerprint") != preview.get("fingerprint"):
            issues.append("미리보기 렌더 버전이 일치하지 않습니다.")
        if (meta.get("prop_guides") or meta.get("support")) and report.get(
            "scene_spec"
        ) != {k: meta.get(k, "") for k in ("prop_guides", "support")}:
            issues.append(
                "소품·좌석 배치가 바뀌었습니다. 미리보기를 다시 생성해 주세요."
            )
        if report.get("fingers", {}).get("applied_joints") != 30:
            issues.append("미리보기에 적용된 손가락 30관절 기록이 필요합니다.")
        if {
            k: v["sha256"] for k, v in report.get("thumbnails", {}).items()
        } != meta.get("thumbnail_versions"):
            issues.append("렌더 결과와 현재 미리보기 해시가 일치하지 않습니다.")
    except (OSError, KeyError, ValueError, TypeError):
        issues.append("현재 미리보기의 렌더 결과 파일을 확인할 수 없습니다.")
    return issues


def visual_evidence(pose, review):
    """Retain old pixel-bound approvals; all new approvals use the checklist."""
    evidence = review.get("evidence") or {}
    legacy = False
    if not evidence:
        evidence = pose.metadata.get("quality_review") or {}
        legacy = evidence.get("status") == "accepted" and set(
            evidence.get("views", [])
        ) == set(policy.VIEWS)
    bound = evidence.get("bvh_sha256") == pose.content_hash and evidence.get(
        "thumbnail_sha256"
    ) == pose.metadata.get("thumbnail_versions")
    complete = legacy or (
        set(evidence.get("visual_checks", [])) == set(policy.VISUAL_CHECKS)
        and evidence.get("policy_fingerprint") == policy.fingerprint()
    )
    return bool(bound and complete), evidence, legacy


def inspect_pose(pose, reviews, *, bases=None):
    findings = []
    metrics = {}

    def add(code, message, severity="block"):
        findings.append({"code": code, "severity": severity, "message": message})

    review = decision(pose, reviews)
    meta = pose.metadata
    try:
        if sha256(pose.bvh) != pose.content_hash:
            raise ValueError("manifest의 BVH 해시와 실제 파일이 다릅니다.")
        joints, frames = parse_bvh(str(pose.bvh))
        if len(frames) != 1 or not np.isfinite(frames).all():
            raise ValueError("유한한 수치의 단일 프레임이어야 합니다.")
        names = {j[0].split(":")[-1]: i for i, j in enumerate(joints) if not j[4]}
        if len(names) != sum(not j[4] for j in joints):
            raise ValueError("중복된 관절 이름이 있습니다.")
        for i, j in enumerate(joints):
            if j[1] >= i or (i and j[1] < 0) or not np.isfinite(j[2]).all():
                raise ValueError("유효하지 않은 관절 계층 또는 offset입니다.")
        motion = re.split(
            r"Frame\s+Time:\s*([^\s]+)", pose.bvh.read_text(encoding="utf-8-sig")
        )
        if (
            len(motion) != 3
            or not 0 < float(motion[1]) < 1
            or len(motion[2].split()) != frames.size
        ):
            raise ValueError("프레임 시간 또는 채널 수와 실제 데이터 길이가 다릅니다.")
        for side in ["Left", "Right"]:
            for finger in policy.FINGERS:
                parent = names.get(side + "Hand", names.get(side + "Wrist"))
                for n in [1, 2, 3]:
                    index = names.get(f"{side}Hand{finger}{n}")
                    if index is None or parent is None or joints[index][1] != parent:
                        raise ValueError(f"{side} {finger}의 3관절 연결이 필요합니다.")
                    if (
                        set(joints[index][3]) != {"Xrotation", "Yrotation", "Zrotation"}
                        or np.linalg.norm(joints[index][2]) < 1e-8
                    ):
                        raise ValueError(
                            "손가락 회전 채널 또는 뼈 길이가 유효하지 않습니다."
                        )
                    parent = index
        metrics["geometry"] = inspect_bvh(pose.bvh)
        if metrics["geometry"]["finger_joints"] != 30:
            raise ValueError("손가락 관절이 정확히 30개여야 합니다.")
        for name, value in metrics["geometry"]["angles"].items():
            limit = policy.WRIST_BEND if name.endswith("Wrist") else policy.LIMB_BEND
            if value > limit:
                add(
                    "angle." + name,
                    f"{name} {value:.1f}°: 실루엣·관절 방향 재확인 필요",
                    "review",
                )
        metrics["capsules"] = inspect_capsules(pose.bvh)
        for side, value in metrics["capsules"]["arms"].items():
            if value.get("available") and value["depth"] > policy.CAPSULE_DEPTH:
                add(
                    "capsule." + side,
                    "팔/몸통 캡슐 중첩: 실제 메시와 4방향 이미지를 함께 확인",
                    "review",
                )
    except (OSError, ValueError, IndexError, KeyError, TypeError, OverflowError) as exc:
        add("bvh.invalid", str(exc))
    if meta.get("scenario"):
        from ..scenarios.validation import reach_error

        if error := reach_error(meta.get("checks", {})):
            add("scenario.reach", error)
    for field in ["source", "author", "license", "source_url", "source_sha256"]:
        if not meta.get(field):
            add("source." + field, f"출처 기록 누락: {field}")
    if meta.get("source_sha256") and not re.fullmatch(
        r"[0-9a-f]{64}", meta["source_sha256"]
    ):
        add("source.hash", "원본 파일 SHA-256 형식이 잘못됐습니다.")
    if not meta.get("hand_augmentation") and meta.get("source_hand_joints") != 30:
        add("hands.provenance", "원본 손 관절 또는 합성 손 프리셋 기록이 필요합니다.")
    if (
        meta.get("hand_augmentation")
        and meta["hand_augmentation"].get("captured_from_source") is not False
    ):
        add("hands.synthetic", "합성 손을 원본 캡처와 명확히 구분해야 합니다.")
    if meta.get("near_duplicate"):
        add(
            "duplicate",
            "기존 포즈 또는 다른 후보와 유사합니다. 추가 가치 확인 필요",
            "review",
        )
    for message in preview_issues(pose):
        add("preview.invalid", message)
    anatomy = meta.get("anatomy_check") or {}
    metrics["mesh"] = anatomy
    if (
        anatomy.get("bvh_sha256") != pose.content_hash
        or anatomy.get("policy_fingerprint") != policy.fingerprint()
    ):
        add("mesh.stale", "현재 BVH와 검수 기준에 맞춘 캐릭터 검사가 필요합니다.")
    for field, keys, limit in [
        (
            "torso_segment_rotation_degrees",
            ["Spine", "Spine1", "Spine2"],
            policy.SPINE_ROTATION,
        ),
        ("hinge_mismatch_degrees", ["Left", "Right"], policy.ELBOW_HINGE),
        ("skin_intersections", ["Left", "Right"], 0),
    ]:
        values = anatomy.get(field, {})
        for key in keys:
            value = values.get(key)
            straight = (
                field == "hinge_mismatch_degrees" and key in values and value is None
            )
            if not straight and (not _finite_number(value) or value < 0):
                add("mesh.missing", f"검사 수치 누락/무효: {field}/{key}")
            elif not straight and value > limit:
                add(
                    "mesh." + field + "." + key,
                    f"{key}: {field} = {value} (허용 {limit} 이하)",
                )
    for flag in anatomy.get("flags", []):
        add("mesh.flag", flag)
    if bases is not None and meta.get("composition_variant"):
        dependency = meta["composition_variant"]
        base = bases.get(dependency["base_pose_id"])
        if base is None or dependency["base_sha256"] != base.content_hash:
            add(
                "composition.stale",
                "기준 포즈가 변경됐습니다. 구도 변형을 재생성해야 합니다.",
            )
    visual, evidence, legacy = visual_evidence(pose, review)
    resolved = (
        set(evidence.get("resolved_findings", []))
        if visual and review.get("note", "").strip()
        else set()
    )
    unresolved = [
        f["code"]
        for f in findings
        if f["severity"] == "review" and f["code"] not in resolved
    ]
    blocked = any(f["severity"] == "block" for f in findings)
    publishable = (
        review["status"] == "accepted" and visual and not blocked and not unresolved
    )
    status = (
        "excluded"
        if review["status"] == "rejected"
        else "blocked" if blocked else "approved" if publishable else "visual_review"
    )
    return {
        "pose_id": pose.pose_id,
        "key": pose.key,
        "batch": pose.batch,
        "bvh_sha256": pose.content_hash,
        "policy_fingerprint": policy.fingerprint(),
        "status": status,
        "publishable": publishable,
        "automatic_ready": not blocked,
        "visual_evidence_current": visual,
        "legacy_visual_evidence": legacy,
        "unresolved_findings": unresolved,
        "findings": findings,
        "metrics": metrics,
        "review_url": f"http://127.0.0.1:8765/?pose={pose.key}",
    }
