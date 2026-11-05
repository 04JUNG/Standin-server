"""Expand only around diagnosed blocking constraints, preserving authored anchors."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.local import surface

source, initial, destination = map(Path,sys.argv[1:4])
raw=np.load(source)
change=np.load(initial)
rest,faces=raw["rest"],raw["triangles"]
graph=surface.Neighborhood(faces,rest)
protected=raw["anchors"]|graph.fixed
free=np.zeros(len(rest),bool)
free[change["ids"]]=True
displacement=np.zeros_like(rest)
displacement[change["ids"]]=change["delta"]
destination.mkdir(parents=True,exist_ok=True)
summary=[]
for stage in range(5):
    folder=destination/f"stage-{stage}"
    folder.mkdir(exist_ok=True)
    candidate=folder/"seed.npz"
    np.savez_compressed(candidate,ids=np.flatnonzero(free),delta=displacement[free])
    command=[sys.executable,"-B",str(Path(__file__).with_name("experiment_nonlinear_repair.py")),
             str(source),str(candidate),str(folder),"0","40"]
    if len(sys.argv) > 4:
        command.append(sys.argv[4])
    with (folder/"log.txt").open("w") as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((folder/"report.json").read_text())
    blockers_path=folder/"blockers.json"
    if not blockers_path.exists():
        summary.append(dict(stage=stage,free=int(free.sum()),result=report))
        break
    blockers=json.loads(blockers_path.read_text())
    blocked=np.array(blockers["vertices"],int)
    expand=np.zeros(len(rest),bool)
    expand[blocked]=True
    for _ in range(2):
        previous=expand.copy()
        expand[graph.a[previous[graph.b]]]=True
        expand[graph.b[previous[graph.a]]]=True
    updated=(free|expand)&~protected
    row=dict(stage=stage,free=int(free.sum()),added=int((updated&~free).sum()),
             protectedBlockers=int(protected[blocked].sum()),groups=blockers["groups"],
             targetMm=report["targetMm"])
    summary.append(row)
    print(row,flush=True)
    (destination/"summary.json").write_text(json.dumps(summary,indent=2))
    if np.array_equal(updated,free):
        break
    free=updated
(destination/"summary.json").write_text(json.dumps(summary,indent=2))
