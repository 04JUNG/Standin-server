"""Fresh candidate FBXs from the exact delivered source and actual recipe defaults."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.character_lod.quality import load
from converter.prepared_body import preset_values
p=argparse.ArgumentParser();p.add_argument('--package',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.package);r=json.loads((root/'shape-bakes.json').read_text());results={}
for cid,raw in r['characters'].items():
 if '-candidate-' not in cid:continue
 source=r['sources'][raw['rules']];c=load(source['blend'],source['controller']);c.set_controls(preset_values(raw),correct=False);export=c.export_fixed(str(root/'export-receipts'));source_fbx=export/'model.fbx';target=root/'fbx'/(cid+'.fbx');shutil.copyfile(source_fbx,target);raw['asset_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();raw['asset']=c.objects()[0]['asset_identity'];results[cid]=json.loads((export/'manifest.json').read_text())
(root/'shape-bakes.json').write_text(json.dumps(r,indent=2)+'\n');(root/'export-report.json').write_text(json.dumps(results,indent=2)+'\n')
