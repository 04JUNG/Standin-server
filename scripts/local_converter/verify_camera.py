"""Real packaged IPC camera regression. Inputs remain local; report contains hashes only."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.verify_runtime import Session, sha


def verify(launcher, sources, pose, output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = {"contract":"personal-camera/1", "models":[]}
    cameras = [
        {"version":"personal-camera/1","sourceBvhSha256":None,"rotation":None},
        {"version":"personal-camera/1","sourceBvhSha256":sha(pose),"rotation":[[0,0,1],[0,1,0],[-1,0,0]]},
        {"version":"personal-camera/1","sourceBvhSha256":sha(pose),"rotation":[[0,-1,0],[1,0,0],[0,0,1]]},
    ]
    with tempfile.TemporaryDirectory(prefix="personal-camera-") as temp:
        storage = Path(temp)/"store"
        for index, source in enumerate(sources):
            source=Path(source).resolve(); model=f"camera-{index}"; digest=sha(source)
            session=Session(launcher,storage)
            try:
                hello=session.request("hello")
                assert hello["cameraContractVersion"]=="personal-camera/1"
                imported=session.request("import",localModelId=model,sourcePath=str(source),sourceHash=digest)
                assert imported["validation"]["passed"]
            finally: session.close()
            session=Session(launcher,storage)
            try:
                inspected=session.request("inspect",localModelId=model)
                assert inspected["sourceHash"]==digest
                records=[]
                for i,camera in enumerate(cameras):
                    fields=dict(localModelId=model,sourceHash=digest,posePath=str(Path(pose).resolve()),poseHash=sha(pose),selectionId=f"camera-{i}",hands={"left":None,"right":None},camera=camera)
                    applied=session.request("apply",**fields)
                    assert applied["snapshot"]["camera"]==camera and applied["validation"]["passed"]
                    destination=output/f"model-{index}-camera-{i}.fbx"
                    session.request("export",localModelId=model,snapshotHash=applied["snapshotHash"],destinationPath=str(destination))
                    assert sha(destination)==applied["artifactHash"]
                    cached=session.request("apply",**fields)
                    assert cached["cacheHit"] and cached["snapshotHash"]==applied["snapshotHash"] and cached["artifactHash"]==applied["artifactHash"]
                    records.append({"camera":camera,"snapshotHash":applied["snapshotHash"],"artifactHash":applied["artifactHash"],"validation":applied["validation"],"retarget":applied["retarget"]})
                assert len(set(r["snapshotHash"] for r in records))==3
                invalid={**fields,"camera":{**cameras[1],"rotation":[[-1,0,0],[0,1,0],[0,0,1]]}}
                session.request("apply",error="CAMERA_UNSUPPORTED",**invalid)
                missing={k:v for k,v in fields.items() if k!="camera"}
                session.request("apply",error="INVALID_REQUEST",**missing)
                session.request("export",error="SNAPSHOT_MISMATCH",localModelId=model,snapshotHash="0"*64,destinationPath=str(output/f"stale-{index}.fbx"))
                session.request("delete",localModelId=model)
                assert all(sha(output/f"model-{index}-camera-{i}.fbx")==r["artifactHash"] for i,r in enumerate(records))
                report["models"].append({"sourceHash":digest,"format":source.suffix[1:],"poseHash":sha(pose),"versions":imported["versions"],"results":records,"restart":True,"cache":True,"invalidCameraRejected":True,"staleRejected":True,"deletePreservedExports":True})
                (output/"verification.json").write_text(json.dumps(report,indent=2))
            finally: session.close()
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--launcher',type=Path,required=True);p.add_argument('--source',type=Path,action='append',required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=verify(a.launcher,a.source,a.pose,a.output)
    print(json.dumps({"models":len(report['models']),"cameraVariants":3,"passed":True}))
