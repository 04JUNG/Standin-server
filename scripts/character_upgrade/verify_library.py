"""Check current staging BVH inventory against the frozen solver's actual mappings."""
import argparse,json,hashlib,sqlite3,re,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from converter.bone_map import resolve_profile,PROFILES,REQUIRED_BONES
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root);p=r/'live-library'
m=json.loads((p/'library_manifest.json').read_text());assert hashlib.sha256((p/'poses.db').read_bytes()).hexdigest()==m['db_sha256']
rows=sqlite3.connect(p/'poses.db').execute('select pose_id,bvh_path from poses').fetchall();groups=collections.Counter();profiles=collections.Counter();reps={}
for pid,path in rows:
 f=p/'bvh'/Path(path).name;text=f.read_text();names=re.findall(r'(?:ROOT|JOINT)\s+(\S+)',text)
 profile=resolve_profile(names);missing=[b for b in REQUIRED_BONES if PROFILES[profile][b] not in names]
 if missing:raise ValueError((pid,profile,missing))
 groups[tuple(names)]+=1;profiles[profile]+=1;reps.setdefault(profile,pid)
report={'libraryVersion':m['library_version'],'dbSha256':m['db_sha256'],'poseIds':sorted(x[0] for x in rows),'skeletonGroups':len(groups),'profiles':dict(profiles),'representatives':reps,
 'coverage':'All required bone mappings checked; conversion QA samples are not an artistic review of every pose.'}
(r/'pose-compatibility.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='poseIds'},indent=2))
