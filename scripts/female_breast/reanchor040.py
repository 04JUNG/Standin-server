"""Restore a copied immutable candidate to a NEW local directory without repinning.

Only local absolute paths in the copied metadata/registries change. Every model,
controller and FBX byte hash is verified and retained. The original directory is
never modified, and no runtime or skin certification is added.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main(a):
 source=a.candidate.resolve();out=a.out.resolve()
 if out.exists():raise ValueError('Use a new destination')
 meta=json.loads((source/'candidate.json').read_text())
 for key in ['blend','controller']:
  path=source/'model'/Path(meta['source'][key]).name
  if sha(path)!=meta['source'][key+'_sha256']:raise ValueError('Source bytes differ: '+key)
 shutil.copytree(source,out,ignore=shutil.ignore_patterns('__pycache__','runtime-full.bin'))
 moves=[]
 for key in ['blend','controller']:
  old=meta['source'][key];meta['source'][key]=str(out/'model'/Path(old).name);moves.append([old,meta['source'][key]])
 write(out/'candidate.json',meta)
 registry=json.loads((out/'release/shape-bakes.local.json').read_text())
 for cid,record in registry['characters'].items():
  if sha(out/'release/fbx'/(cid+'.fbx'))!=record['asset_sha256']:
   raise ValueError('Neutral FBX bytes differ: '+cid)
 for record in registry['sources'].values():
  for key in ['blend','controller']:
   old=record[key];path=out/'release/sources'/Path(old).name
   if sha(path)!=record[key+'_sha256']:raise ValueError('Release source bytes differ')
   record[key]=str(path);moves.append([old,str(path)])
 write(out/'release/shape-bakes.local.json',registry)
 for folder in ['prepared-linux','prepared-macos']:
  path=out/folder/'shape-bakes.local.json'
  if not path.exists():continue
  prepared=json.loads(path.read_text());prepared['sources']=registry['sources']
  for entry in prepared.get('prepared_neutrals',{}).get('entries',{}).values():
   old=entry['path'];entry['path']=str(out/folder/'qa'/Path(old).name);moves.append([old,entry['path']])
   receipt_path=Path(entry['path'])
   if sha(receipt_path)!=entry['sha256']:raise ValueError('Prepared receipt bytes differ')
   receipt=json.loads(receipt_path.read_text())
   if sha(out/folder/'fbx'/('prepared-'+receipt['key']+'.fbx'))!=receipt['fbx_sha256']:
    raise ValueError('Prepared FBX bytes differ')
  write(path,prepared)
 write(out/'local-relocation.json',{'original':str(source),'destination':str(out),'source_hashes_unchanged':True,'paths':moves,'approvals_changed':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
