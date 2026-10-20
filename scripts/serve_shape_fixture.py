"""Local-only real converter integration fixture; no credentials or cloud writes."""
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));R=ROOT/'artifacts/shape-staging/release';reg=json.loads((R/'characters.local.json').read_text())
os.environ['CONVERTER_CHARACTER_REGISTRY']=str(R/'characters.local.json');os.environ['CONVERTER_DEFAULT_CHARACTER_ID']=next(iter(reg['characters']));os.environ['CONVERTER_SHAPE_REGISTRY']=str(R/'shape-bakes.local.json');os.environ['CONVERTER_TIMEOUT_SECONDS']='600';os.environ['CONVERTER_CAMERA_TIMEOUT_SECONDS']='600'
for k,v in reg['characters'].items():os.environ[v['artifact_uri_env']]=str(R/'fbx'/(k+'.fbx'))
import uvicorn
from converter_api.app import app
uvicorn.run(app,host='127.0.0.1',port=18031,log_level='warning')
