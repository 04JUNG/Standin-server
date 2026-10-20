"""A two-actor scene must survive candidate indexing as two linked BVHs."""

import sqlite3

from pose_curation.candidates import build_candidates
from src.repo import load_entries
from tests.test_pose_curation import make_motion


def test_fighting_pair_keeps_roles_and_solo_default(tmp_path):
    source = tmp_path / "motion.bvh"
    make_motion(source)
    rows = []
    for pose_id, role in (("attacker", "A"), ("defender", "B"), ("solo", None)):
        target = tmp_path / f"{pose_id}.bvh"
        target.write_bytes(source.read_bytes())
        rows.append({
            "pose_id": pose_id, "bvh": target.name, "movement": "combat",
            "source": "test", "license": "CC0-1.0", "author": "test",
            "source_url": "https://example.org", "source_frame_0based": 0,
            **({"relationship": "fighting", "set_id": "pair", "set_role": role}
               if role else {}),
        })

    assert build_candidates(rows, tmp_path, "batch") == 12
    entries = load_entries(str(tmp_path / "candidates.db"))
    by_id = {}
    for entry in entries:
        by_id.setdefault(entry.pose_id, []).append(entry)
    for pose_id, role in (("attacker", "A"), ("defender", "B")):
        assert len(by_id[pose_id]) == 4
        assert {e.tags["relationship"] for e in by_id[pose_id]} == {"fighting"}
        assert {e.meta["set_id"] for e in by_id[pose_id]} == {"pair"}
        assert {e.meta["set_role"] for e in by_id[pose_id]} == {role}
    assert {e.tags["relationship"] for e in by_id["solo"]} == {"solo"}
    with sqlite3.connect(tmp_path / "candidates.db") as con:
        assert con.execute(
            "SELECT pose_id,set_role FROM poses WHERE set_id='pair' ORDER BY set_role"
        ).fetchall() == [("attacker", "A"), ("defender", "B")]
