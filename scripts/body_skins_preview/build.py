"""Explicit offline build. Requires Blender, NumPy, Pillow and a local Three.js copy."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--three-root',type=Path,required=True);p.add_argument('--blender',default='blender');p.add_argument('--resolution',type=int,default=4096);a=p.parse_args();out=a.output.resolve();assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
sources=json.loads((HERE/'source_assets.json').read_text())
for row in sources['models']:
 subprocess.run([a.blender,'-b','--python',str(HERE/'prepare_model.py'),'--','--source',str((a.source_root/row['relative_source']).resolve()),'--sha256',row['sha256'],'--out',str(assets),'--id',row['id']],check=True)
 subprocess.run([sys.executable,str(HERE/'generate_textures.py'),'--assets',str(assets),'--id',row['id'],'--resolution',str(a.resolution)],check=True)
 subprocess.run([a.blender,'-b','--python',str(HERE/'export_fbx.py'),'--','--assets',str(assets),'--id',row['id']],check=True)
for intermediate in assets.glob('*.npz'):intermediate.unlink()
for name in ['index.html','style.css','viewer.js']:shutil.copy2(HERE/name,out/name)
vendor=out/'vendor';vendor.mkdir(exist_ok=True)
for name in ['three.module.js','three.core.js']:shutil.copy2(a.three_root/'build'/name,vendor/name)
for name in ['loaders/GLTFLoader.js','controls/OrbitControls.js','utils/BufferGeometryUtils.js','utils/SkeletonUtils.js']:
 dest=vendor/'addons'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.three_root/'examples/jsm'/name,dest)
shutil.copy2(a.three_root/'LICENSE',vendor/'THREE-LICENSE.txt')
subprocess.run([sys.executable,str(HERE/'manifest.py'),'--output',str(out)],check=True)
print(f'Preview: python3 -m http.server 8791 --bind 127.0.0.1 --directory {out}')
