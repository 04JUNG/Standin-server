"""Build independent ACCAD clip pools; record incompatible clips explicitly."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse

from ..sources.accad import run
from ..storage import read_json, write_json


def _clip(job):
    source, destination, clip, details, count, prefix = job
    try:
        run(source, destination, count, clips={clip: details}, id_prefix=prefix)
        return {"clip": clip, "manifest": str(destination / "manifest.json")}
    except (ValueError, KeyError, OSError, IndexError) as error:
        return {"clip": clip, "error": str(error)}


def build(source, output, config, *, workers=4, count=5):
    if (output / "manifest.json").exists():
        raise ValueError("choose a fresh source pool")
    jobs = [(source, output / "clips" / clip, clip, detail, count, config["id_prefix"])
            for clip, detail in config["clips"].items()]
    poses, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_clip, jobs):
            if "error" in result:
                failures.append(result)
                print("incompatible", result["clip"], result["error"], flush=True)
                continue
            path = Path(result["manifest"])
            rows = read_json(path)["poses"]
            for row in rows:
                row["bvh"] = str((path.parent / row["bvh"]).resolve())
            poses.extend(rows)
            print("exported", result["clip"], len(rows), flush=True)
    write_json(output / "manifest.json", {"poses": poses, "failures": failures,
                                          "automatic_approval": False})
    return {"poses": len(poses), "failed_clips": len(failures)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--count", type=int, default=5)
    a = p.parse_args()
    print(build(a.source, a.output, read_json(a.config), workers=a.workers, count=a.count))
