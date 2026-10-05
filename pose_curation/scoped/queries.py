"""Allowlisted local roughs and their unchanged, real RTMPose observations."""
import re
from pathlib import Path

import numpy as np

from src.partial_pose import observed_upper_body
from src.config import CFG
from ..storage import contained_path, read_json, sha256
from .matching import observed


class RoughQueries:
    def __init__(self, curation: Path):
        roots = sorted((curation / "coverage").glob("*/inputs.json"))
        self.root = roots[-1].parent if roots else None
        self.rows = {}
        if self.root is None:
            return
        files = read_json(self.root / "report.json").get("files", {})
        for item in read_json(self.root / "inputs.json"):
            identity = item["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", identity) or identity not in files:
                continue
            extraction = self.root / "extraction" / (identity + ".json")
            if not extraction.exists():
                continue
            record = read_json(extraction)
            if record.get("image_sha256") != item["sha256"] or record.get("model", {}).get("manual_query_coordinates") is not False:
                continue
            for person in record.get("people", []):
                try:
                    observed(person["keypoints"], person["scores"], item["size"])
                except ValueError:
                    continue
                points, scores = np.array(person["keypoints"]), np.array(person["scores"])
                upper = observed_upper_body(points, scores >= .3, owner_box=None, peer_boxes=(), cfg=CFG) is not None
                key = f"{identity}:{person['person']}"
                self.rows[key] = {"key": key, "image_id": identity, "person": person["person"],
                                  "origin": "제공 러프" if item["origin"] == "supplied" else "사용자 러프",
                                  "upper_only": upper, "size": item["size"], "sha256": item["sha256"],
                                  "path": contained_path(self.root, files[identity]),
                                  "keypoints": person["keypoints"], "scores": person["scores"],
                                  "observed_joints": np.flatnonzero(observed(points, scores, item["size"])).tolist(),
                                  "model": record["model"]}

    def list(self):
        fields = ("key", "image_id", "person", "origin", "upper_only", "size")
        return [{k: row[k] for k in fields} for row in self.rows.values()]

    def get(self, key):
        row = self.rows.get(key)
        if row is None:
            raise ValueError("현재 관절 조건에 맞는 러프가 없습니다.")
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("입력 이미지가 바뀌었습니다. 관절을 다시 추출해 주세요.")
        return row

    def public(self, key):
        return {k: v for k, v in self.get(key).items() if k != "path"}
