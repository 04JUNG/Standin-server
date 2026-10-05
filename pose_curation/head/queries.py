"""All approved local roughs, independent of whether body/face detection succeeds."""
import re
import threading
from pathlib import Path
from ..storage import contained_path, read_json, sha256, write_json, utc_now


class HeadQueries:
    def __init__(self, curation: Path):
        self.review_path = curation / 'head-direction/query-reviews.json'
        self.lock = threading.RLock()
        roots = sorted((curation / "coverage").glob("*/inputs.json"))
        self.rows = {}
        if not roots:
            return
        root = roots[-1].parent
        files = read_json(root / "report.json").get("files", {})
        for item in read_json(root / "inputs.json"):
            key = item["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", key) or key not in files:
                continue
            self.rows[key] = {"key": key, "size": item["size"], "content_hash": item["sha256"],
                              "origin": "제공 러프" if item["origin"] == "supplied" else "사용자 러프",
                              "path": contained_path(root, files[key])}

    def reviews(self):
        return read_json(self.review_path) if self.review_path.exists() else {}

    def list(self):
        with self.lock:
            reviews = self.reviews()
            return [{**{k: v for k, v in row.items() if k != "path"},
                     'excluded': reviews.get(row['content_hash'], {}).get('excluded', False),
                     'exclusion_reason': reviews.get(row['content_hash'], {}).get('reason', '')}
                    for row in self.rows.values()]

    def review(self, key, content_hash, excluded, reason):
        with self.lock:
            row = self.get(key, content_hash, allow_excluded=True)
            reviews = self.reviews()
            reviews[row['content_hash']] = {'excluded': excluded, 'reason': reason,
                                           'updated_at': utc_now(), 'scope': 'head_bust_query_only'}
            write_json(self.review_path, reviews)
            return reviews[row['content_hash']]

    def get(self, key, content_hash=None, *, allow_excluded=False):
        row = self.rows.get(key)
        if row is None:
            raise ValueError("로컬 검수 목록에 없는 러프입니다.")
        if (content_hash is not None and content_hash != row["content_hash"]) or sha256(row["path"]) != row["content_hash"]:
            raise ValueError("러프 이미지가 변경되었습니다. 새로 불러와 주세요.")
        if not allow_excluded and self.reviews().get(row['content_hash'], {}).get('excluded'):
            raise ValueError('두상·흉상 검수 대상에서 제외한 이미지입니다.')
        return row
