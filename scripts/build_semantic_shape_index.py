"""Build critical arm relations and camera observations from unchanged BVHs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.library import load_bvh_pose
from src.semantic_body_index import records_digest
from src.semantic_shapes import SHAPE_VERSION, SHAPE_THRESHOLDS, CAMERAS, relational_geometry, projected_relations


def build_index(db,tags,bvh_root,detail_tags=None):
    started=perf_counter(); raw=json.loads(Path(tags).read_text()); records=raw["poses"] if isinstance(raw,dict) else raw
    by_id={r["pose_id"]:r for r in records}; details={}
    if detail_tags:
        details={r["pose_id"]:[t["value"] for t in r.get("tags",{}).get("body_configuration",[]) if t.get("visual_checked")]
                 for r in json.loads(Path(detail_tags).read_text())["poses"]}
    root=Path(bvh_root).resolve()
    with sqlite3.connect(Path(db).resolve().as_uri()+"?mode=ro",uri=True) as c:
        rows=c.execute("SELECT pose_id,bvh_path FROM poses ORDER BY pose_id").fetchall()
    poses={}; errors=[]
    for pose_id,stored in rows:
        if pose_id not in by_id: continue
        try:
            if Path(stored).name!=pose_id+".bvh": raise ValueError("registry ID mismatch")
            path=(root/Path(stored).name).resolve()
            if path.parent!=root: raise ValueError("outside BVH root")
            p,s=load_bvh_pose(str(path),0); geometry=relational_geometry(p,s)
            r=by_id[pose_id]
            fold_declared=r.get("status")=="reviewed" and any(t.get("field")=="action" and t.get("value")=="fold_arms" for group in r.get("groups",{}).values() for t in group)
            # An arms_crossed archival tag can mean knee-hugging, not chest folding.
            # Only an explicit reviewed action supports the semantic 'fold_arms'.
            folded=geometry["fold_geometry"] if fold_declared else False
            poses[pose_id]={"relations":{"fold_arms":folded,"v_arms":geometry["v_arms"]},
                            "geometry":geometry,"archival_observed_tags":details.get(pose_id,[]),
                            "views":[{"camera":dict(cam),**projected_relations(p,s,cam)} for cam in CAMERAS],
                            "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception as e:
            errors.append({"pose_id":pose_id,"error":type(e).__name__})
    summary={"built":len(poses),"failed":len(errors),"elapsed_ms":round((perf_counter()-started)*1000,2),
             "verified_relations":{key:dict(Counter(str(v["relations"][key]) for v in poses.values())) for key in ("fold_arms","v_arms")}}
    return {"version":SHAPE_VERSION,"thresholds":SHAPE_THRESHOLDS,"cameras":list(CAMERAS),
            "tags_sha256":records_digest(records),"frame":0,"poses":poses,"errors":errors,"summary":summary}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("db","tags","bvh-root","out"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--detail-tags",type=Path)
    args=parser.parse_args(); data=build_index(args.db,args.tags,args.bvh_root,args.detail_tags)
    if not data["poses"]: raise RuntimeError("empty shape index")
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(data,ensure_ascii=False,indent=1))
    print(json.dumps(data["summary"],ensure_ascii=False))
