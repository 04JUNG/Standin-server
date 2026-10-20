"""Reproduce articulated bust exports and package actual FBX renders for CSP review."""

import argparse
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pose_curation.orientation import Orientation
from pose_curation.review.framing import FramedPreviews, NativeReference, ORIENTED_FILES
from pose_curation.review.head_exports import with_body
from pose_curation.storage import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "data/dev/bust-direction-20261003"
    )
    destination = parser.parse_args().output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    service = FramedPreviews(PROJECT / "data/curation")
    reference = NativeReference(service.character, sha256(service.character))
    report = {
        "reference_sha256": reference.content_hash,
        "csp_verified": False,
        "cases": [],
    }
    cases = [
        ("bust_left", (30, 10, -8), (-5, 0, 12)),
        ("bust_right", (-30, 10, 8), (5, 0, -12)),
        ("bust_down", (15, 35, 10), (0, 0, 0)),
        ("bust_back", (180, 0, 0), (150, 0, 0)),
    ]
    try:
        for name, face_values, body_values in cases:
            face, body = Orientation(*face_values), Orientation(*body_values)
            pose = with_body(reference, face, body)
            state = service.wait_ready(pose, "bust", orientation=face)
            row = {
                "name": name,
                "face": face.public(),
                "body": body.public(),
                "version": state["version"],
                "files": {},
            }
            for kind in ORIENTED_FILES:
                source = service.artifact(pose, "bust", face, kind, state["version"])
                target = destination / (name + source.suffix)
                shutil.copyfile(source, target)
                row["files"][kind] = {"file": target.name, "sha256": sha256(target)}
            settings = read_json(destination / (name + ".json"))
            row["validation"] = settings["fbx_validation"]
            row["articulation"] = settings["bust_articulation"]
            report["cases"].append(row)
    finally:
        service.close()
    write_json(destination / "verification.json", report)
    (destination / "README.txt").write_text(
        "머리·몸통 방향을 분리한 흉상 FBX 4종\n"
        "각 JPG는 내보낸 FBX를 Blender에서 다시 불러와 렌더한 결과입니다.\n"
        "몸통은 body, 머리는 face 각도입니다. 설정 JSON에 별도로 기록합니다.\n"
        "메시를 흉상으로 자르고 52개 뼈는 유지합니다. 목·머리 관절에 회전을 분산했습니다.\n"
        "CSP에서는 정면·평행 투영 카메라로 JPG와 비교하세요. 기존 레이어 카메라는 바꾸지 않습니다.\n"
        "이 새 흉상 보정 파일의 CSP 검증은 아직 받지 않았습니다.\n"
        "특정 러프의 몸통·표정·어깨 들썩임을 추정한 파일이 아닌 재현 가능한 방향 시험입니다.\n",
        encoding="utf-8",
    )
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        # Only the files generated for these cases; never include other local roughs.
        for filename in ["README.txt", "verification.json"] + [
            c[0] + ext for c in cases for ext in (".fbx", ".jpg", ".json")
        ]:
            output.write(destination / filename, filename)
    print({"archive": str(archive), "cases": len(report["cases"])})


if __name__ == "__main__":
    main()
