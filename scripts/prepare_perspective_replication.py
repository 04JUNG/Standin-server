"""Reserve 40 more families without re-tuning any camera/inference parameter."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.experiment_z_axis_search import load_library
from scripts.experiment_z_rendered import save_json,sha256

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    development=ROOT/'out/perspective-coordinates-20261012/protocol.json'
    first=ROOT/'out/perspective-images-20261012/answers.json'
    source=json.loads(development.read_text())
    old=set(source['excluded_families']+source['dev_families'])
    old.update(r['family_id'] for r in json.loads(first.read_text()))
    unique={}
    for pose in load_library(a.library_root):
        if pose.family_id not in old: unique.setdefault(pose.family_id,pose)
    pool=list(unique.values());rng=np.random.default_rng(20261016)
    chosen=[pool[int(i)] for i in rng.choice(len(pool),40,replace=False)]
    save_json(a.out/'protocol.json',{'stage':'independent 40-family image replication selection only',
        'seed':20261016,'dev_families':source['dev_families'],'excluded_families':sorted(old),
        'reserved_future_image_test_pose_ids':[p.pose_id for p in chosen],
        'reserved_future_image_test_families':[p.family_id for p in chosen],
        'fixed_policy':'same 117-view primary versus 4-view baseline, same Human-Art XY and masks; no tuning after first test',
        'stopping_rule':'one additional fixed batch of 40; report separately regardless of outcome, no per-sample peeking',
        'success_gate':'same gate as first batch, no retroactive relaxation; no claim of replicated confidence unless independently passed',
        'hashes':{'original_development':sha256(development),'first_answers':sha256(first),
                  'script':sha256(Path(__file__))}})
    print('Reserved',len(chosen),'families disjoint from',len(old),'previous families')
