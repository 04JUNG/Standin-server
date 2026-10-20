"""Authoring constraints for scenario intent, separate from anatomy approval."""

MAX_TARGET_ERROR_CM = 4.0


def reach_error(checks):
    errors = checks.get("ik_target_adjustments_cm", {})
    if set(errors) != {"Leftarm", "Rightarm", "Leftleg", "Rightleg"}:
        return "손·발의 목표 도달 검사 기록이 없습니다."
    import math

    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in errors.values()
    ):
        return "손·발의 목표 도달 검사 수치가 유효하지 않습니다."
    if max(errors.values()) > MAX_TARGET_ERROR_CM:
        return f"손·발 목표 오차가 {MAX_TARGET_ERROR_CM:g}cm를 넘습니다. 자세를 다시 설계해야 합니다."
    return None
