"""Pinned, content-verified private dataset acquisition and auditable inventory."""
import argparse, concurrent.futures, hashlib, json, pathlib, subprocess, base64, time
COMMIT='61eed393408552e1a0b28645f2e409be217bfcfb'
PREFIX='러프 데이터/오픈베타 러프/'
EXPOSED={'001','018','026','041','074','087'}
def sha(data): return hashlib.sha256(data).hexdigest()
def gitsha(data): return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def api(path):
    p=subprocess.run(['gh','api','repos/04JUNG/Standin-master-docs/'+path],capture_output=True,text=True,check=True)
    return json.loads(p.stdout)
def fetch(root, item):
    path=root/item['path'][len(PREFIX):]
    if path.exists() and gitsha(path.read_bytes())==item['sha']: return
    for attempt in range(3):
        try:
            data=base64.b64decode(api('git/blobs/'+item['sha'])['content'])
            if gitsha(data)!=item['sha']: raise ValueError('blob hash mismatch')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return
        except Exception:
            if attempt==2: raise
            time.sleep(1)
def inventory(tree, root):
    files={x['path'][len(PREFIX):]:x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith(PREFIX)}
    rows=[]
    for path,item in sorted(files.items()):
        if not path.startswith('러프/'): continue
        case=pathlib.Path(path).stem; group=case.split('.')[0]; gt='정답_3d인형/'+case+'.fbx'
        row=dict(case_id=case,original_cut_id=group,image=path,gt=gt,commit=COMMIT,
          source_seed_status='missing',split='development' if group in EXPOSED else 'unassigned',
          split_status='provisional_not_blind',artist_id=None,scene_group=None,
          representation='unknown',visibility_review='pending',camera_status='missing',
          exposure='visual_inspection' if group in EXPOSED else 'prior_exposure_unknown')
        row['files']={}
        for kind,p in [('image',path),('gt',gt)]:
            entry=files.get(p); local=root/p
            row['files'][kind]={'git_blob':entry['sha'] if entry else None,'available':local.exists()}
            if local.exists():
                data=local.read_bytes();row['files'][kind].update(sha256=sha(data),bytes=len(data),verified=entry is not None and gitsha(data)==entry['sha'])
        rows.append(row)
    return rows
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--tree');a.add_argument('--download',action='store_true');args=a.parse_args()
    root=pathlib.Path(args.root);root.mkdir(parents=True,exist_ok=True)
    tree=json.loads(pathlib.Path(args.tree).read_text()) if args.tree else api('git/trees/'+COMMIT+'?recursive=1')
    if tree.get('truncated'): raise ValueError('truncated inventory')
    if args.download:
        items=[x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith(PREFIX)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda x:fetch(root,x),items))
    rows=inventory(tree,root)
    (root/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps({'pairs':len(rows),'groups':len({x['original_cut_id'] for x in rows}),'verified_pairs':sum(all(f.get('verified') for f in x['files'].values()) for x in rows),'seed_missing':sum(x['source_seed_status']=='missing' for x in rows),'blind_split_ready':False}))
if __name__=='__main__': main()
