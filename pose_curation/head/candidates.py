"""Revision-bound face candidates, built offline and selected explicitly in review."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .fitting import assess, canonical
from .queries import HeadQueries
from .extract import MODEL_SHA256
from .anime_experiment import WEIGHT_HASHES
from ..orientation import Orientation
from ..storage import contained_path, read_json, sha256, write_json

VERSION = "head-candidates-v1"
OBSERVATIONS = ("crop-observations", "anime-observations", "recovery-observations")


def code_revision():
    package = Path(__file__).parent
    files = [
        package / "candidates.py",
        package / "fitting.py",
        package / "extract.py",
        package / "queries.py",
        package / "regions.py",
        package / "crop_experiment.py",
        package / "anime_experiment.py",
        package / "recovery_experiment.py",
        package.parent / "orientation.py",
        package.parent / "scoped/matching.py",
    ]
    return hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()


def overlap(first, second):
    a, b = np.asarray(first), np.asarray(second)
    intersection = np.prod(
        np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    )
    return float(
        intersection
        / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection, 1e-8)
    )


def rotation_difference(first, second):
    def matrix(value):
        return Orientation(**{k: value[k] for k in ("yaw", "pitch", "roll")}).matrix()

    relative = matrix(first).T @ matrix(second)
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def candidate_groups(rows):
    """Merge same face observations; conflicting good fits must not win by cherry-picking."""
    groups = []
    for row in rows:
        group = next(
            (g for g in groups if overlap(g[0]["bbox"], row["bbox"]) > 0.5), None
        )
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    output = []
    for group in groups:
        good = [r for r in group if r["status"] == "suggested"]
        best = min(good or group, key=lambda r: r.get("error", float("inf")))
        merged = {
            **best,
            "reasons": list(best["reasons"]),
            "sources": [r["source"] for r in group],
            "observation_count": len(group),
        }
        if any(r.get("excluded_reference") for r in group):
            merged["status"] = "manual"
            merged["reasons"] = ["검수 대상 그림이 아닌 참고 사진입니다."]
        elif len(good) > 1:
            deviation = max(
                rotation_difference(a["orientation"], b["orientation"])
                for a in good
                for b in good
            )
            merged["crop_deviation_degrees"] = deviation
            if deviation > 15:
                merged["status"] = "manual"
                merged["reasons"].append(
                    "얼굴 영역을 바꿨을 때 추정 각도가 일관되지 않습니다."
                )
        if merged["status"] != "suggested":
            merged.pop("orientation", None)
        output.append(merged)
    return sorted(output, key=lambda r: (r["bbox"][0], r["bbox"][1]))


def build(curation: Path):
    root = curation / "head-direction"
    queries = HeadQueries(curation)
    template = canonical(root / "models/canonical_face_model.obj")
    reviews_path = root / "source-face-reviews.json"
    reviews = read_json(reviews_path) if reviews_path.exists() else {}
    images = {}
    summary = {
        "images": 0,
        "detected_images": 0,
        "candidate_faces": 0,
        "suggested_faces": 0,
        "suggested_images": 0,
    }
    for item in queries.list():
        if item["excluded"]:
            continue
        row = queries.get(item["key"])
        observations = []
        inputs = {}
        for kind in OBSERVATIONS:
            relative = f"{kind}/{row['key']}.json"
            path = root / relative
            if not path.exists():
                continue
            raw_hash = sha256(path)
            raw = read_json(path)
            if raw.get("image_sha256") != row["content_hash"]:
                continue
            config = raw.get("config", {})
            if kind == "crop-observations":
                valid = (
                    config.get("version") == "body-crop-mediapipe-v1"
                    and config.get("model_sha256") == MODEL_SHA256
                    and config.get("regions_sha256")
                    == sha256(Path(__file__).with_name("regions.py"))
                )
            elif kind == "anime-observations":
                valid = (
                    config.get("version") == "anime-box-mediapipe-v1"
                    and config.get("weight_hashes") == WEIGHT_HASHES
                    and config.get("face_model_sha256") == MODEL_SHA256
                    and config.get("anime_face_detector") == "0.1.0"
                )
            else:
                valid = (
                    config.get("version") == "crop-recovery-mediapipe-v1"
                    and config.get("model_sha256") == MODEL_SHA256
                    and config.get("code_sha256")
                    == sha256(Path(__file__).with_name("recovery_experiment.py"))
                    and bool(raw.get("inputs"))
                    and set(raw["inputs"]).issubset(
                        {"anime-observations", "crop-observations"}
                    )
                    and all(
                        sha256(root / kind / f"{row['key']}.json") == digest
                        for kind, digest in raw.get("inputs", {}).items()
                    )
                )
            if not valid:
                raise ValueError(f"Stale observation recipe: {relative}")
            inputs[relative] = raw_hash
            for i, face in enumerate(raw["faces"]):
                result = assess(
                    face["points"], face["flipped_faces"], face["crop_size"], template
                )
                if "bbox" not in result:
                    continue
                offset = np.array(face["region"]["bbox"][:2])
                result["bbox"] = (
                    (np.asarray(result["bbox"]).reshape(2, 2) + offset).ravel().tolist()
                )
                for name in ("points", "projected"):
                    if name in result:
                        result[name] = (np.array(result[name]) + offset).tolist()
                if "translation" in result:
                    result["translation"] = (
                        np.array(result["translation"]) + offset
                    ).tolist()
                source = f"{relative}#{i}@{raw_hash}"
                observations.append(
                    {
                        **result,
                        "source": source,
                        "excluded_reference": reviews.get(source, {}).get(
                            "excluded", False
                        ),
                        "detector": {
                            "anime-observations": "그림 얼굴 검출",
                            "crop-observations": "몸 관측 기반 얼굴 확대",
                            "recovery-observations": "회전·확대 재검출",
                        }[kind],
                    }
                )
            if sha256(path) != raw_hash:
                raise ValueError("Observations changed during build")
        candidates = candidate_groups(observations)
        for candidate in candidates:
            identity = [VERSION, row["content_hash"], candidate["sources"]]
            candidate["id"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()[:24]
            candidate["requires_target_selection"] = True
            candidate["confidence"] = "unvalidated"
        images[row["key"]] = {
            "image_sha256": row["content_hash"],
            "size": row["size"],
            "observations": inputs,
            "candidates": candidates,
        }
        summary["images"] += 1
        summary["detected_images"] += bool(candidates)
        summary["candidate_faces"] += len(candidates)
        good = sum(c["status"] == "suggested" for c in candidates)
        summary["suggested_faces"] += good
        summary["suggested_images"] += bool(good)
    manifest = {
        "version": VERSION,
        "code_revision": code_revision(),
        "summary": summary,
        "images": images,
        "source_review_sha256": sha256(reviews_path) if reviews_path.exists() else None,
    }
    write_json(root / "candidates.json", manifest)
    return summary


class HeadCandidates:
    def __init__(self, curation: Path, queries: HeadQueries):
        self.root = curation / "head-direction"
        self.queries = queries
        self._cache = None
        self._signature = None

    def manifest(self):
        path = self.root / "candidates.json"
        signature = (path.stat().st_mtime_ns, path.stat().st_size, code_revision())
        if signature != self._signature:
            value = read_json(path)
            if (
                value.get("version") != VERSION
                or value.get("code_revision") != signature[2]
            ):
                raise ValueError(
                    "얼굴 후보 코드가 변경되었습니다. 후보를 다시 생성해 주세요."
                )
            self._cache = value
            self._signature = signature
        review = self.root / "source-face-reviews.json"
        if (sha256(review) if review.exists() else None) != self._cache.get(
            "source_review_sha256"
        ):
            raise ValueError(
                "얼굴 후보 검수 기록이 바뀌었습니다. 후보를 다시 생성해 주세요."
            )
        return self._cache

    def counts(self):
        try:
            manifest = self.manifest()
            return {
                key: sum(c["status"] == "suggested" for c in row["candidates"])
                for key, row in manifest["images"].items()
            }
        except (OSError, ValueError):
            return {}

    def get(self, key, content_hash=None):
        query = self.queries.get(key, content_hash)
        row = self.manifest()["images"].get(key)
        if row is None or row["image_sha256"] != query["content_hash"]:
            raise ValueError("이 이미지의 얼굴 후보가 아직 없습니다.")
        for relative, digest in row["observations"].items():
            if sha256(contained_path(self.root, relative)) != digest:
                raise ValueError(
                    "얼굴 관측이 변경되었습니다. 후보를 다시 생성해 주세요."
                )
        return row

    def public(self, key):
        row = self.get(key)
        return {
            "image_sha256": row["image_sha256"],
            "items": [
                {
                    k: v
                    for k, v in candidate.items()
                    if k not in {"source", "sources", "excluded_reference"}
                }
                for candidate in row["candidates"]
            ],
        }

    def selected(self, key, content_hash, identity):
        row = self.get(key, content_hash)
        candidate = next((c for c in row["candidates"] if c["id"] == identity), None)
        if (
            candidate is None
            or candidate["status"] != "suggested"
            or "orientation" not in candidate
        ):
            raise ValueError("현재 검증 조건을 통과한 얼굴 후보를 선택해 주세요.")
        return {
            **{
                k: v
                for k, v in candidate.items()
                if k not in {"source", "sources", "excluded_reference"}
            },
            "image_sha256": content_hash,
            "query": key,
            "source": "detected_face_user_selected",
            "version": VERSION,
            "requires_visual_review": True,
        }


if __name__ == "__main__":
    print(build(Path("data/curation")))
