"""Migrate the saved synthesis to corrected known source cameras, no re-render."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.refine_v5.learning_data import source_camera

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',required=True);a=p.parse_args();old=Path(a.original);out=Path(a.out);out.mkdir(exist_ok=False)
    (out/'native-data').symlink_to(old/'native-data',target_is_directory=True);d=dict(np.load(old/'dataset/examples.npz'))
    rigs={r['case_id']:r for r in json.loads((old/'native-data/records.json').read_text()) if r['status']=='ready'};seeds={r['case_id']:r for r in json.loads(Path(a.seeds).read_text())}
    for i in np.flatnonzero(d['domain']=='real'):d['geometry'][i,-9:]=source_camera(rigs[str(d['case_id'][i])]['bones'],seeds[str(d['case_id'][i])]['view']).ravel()
    dest=out/'dataset';dest.mkdir();np.savez_compressed(dest/'examples.npz',**d)
    m=json.loads((old/'dataset/manifest.json').read_text());m['source_camera_fix']='selected_library_view_used_not_forced_front'
    for row in m['records']:
        row['real_camera']='fixed_selected_library_view_GT_camera_unavailable';row['selected_view']=seeds[row['case_id']]['view']
    (dest/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
