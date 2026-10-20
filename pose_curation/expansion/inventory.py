"""Count explicit action names/tags; unlabelled legacy poses remain uncertain."""
import argparse
import json
from pathlib import Path
import re
import sqlite3

from ..storage import read_json, sha256, utc_now, write_json


def inventory(database, families):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute("SELECT pose_id,action,meta_json FROM poses").fetchall()
    texts = []
    unclassified = 0
    for identity, action, raw in rows:
        metadata = json.loads(raw or "{}")
        unclassified += not bool(metadata.get("category"))
        texts.append(" ".join(str(value or "") for value in
                              (identity, action, metadata.get("style"), metadata.get("category"))))
    counts = [{"family": family["family"], "label": family["label"],
               "named_matches": sum(bool(re.search(family["pattern"], text, re.IGNORECASE)) for text in texts)}
              for family in families]
    return {"created_at": utc_now(), "database_sha256": sha256(database), "poses": len(rows),
            "unclassified_legacy": unclassified, "families": counts,
            "scope": "Explicit names and tags only. A zero is a discovery priority, not proof of a geometric gap."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = inventory(a.database, read_json(a.families)["families"])
    write_json(a.out, result)
    print(json.dumps(result, ensure_ascii=True))
