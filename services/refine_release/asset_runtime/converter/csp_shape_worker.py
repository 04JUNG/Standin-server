"""Fresh, approved-source neutral body bake; no auto-executed blend text."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from converter.shape_bake import _load
from converter.csp_vrm import export

def main(job_path):
    job=json.loads(Path(job_path).read_text());recipe=job['recipe'];module=_load(recipe)
    folder=module.export_fixed(str(Path(job_path).parent/'neutral'))
    lineage={k:recipe[k] for k in ('rules','blend_sha256','controller_sha256','shape_sha256','values')}
    qa=export(str(folder/'model.fbx'),job['output'],lineage)
    raw=Path(job['output']).read_bytes()
    report={'schema':'csp-shape.v1','character_id':job['character_id'],'character_sha256':recipe['asset_sha256'],'shape_sha256':recipe['shape_sha256'],'source':{'rules':recipe['rules'],'blendSha256':recipe['blend_sha256'],'controllerSha256':recipe['controller_sha256']},'controls':recipe['values'],'artifact_sha256':hashlib.sha256(raw).hexdigest(),'artifact_bytes':len(raw),'qa':qa}
    Path(job['report']).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1])
