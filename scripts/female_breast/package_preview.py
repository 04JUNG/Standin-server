"""Package exact evaluated model samples into a portable offline review file."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--candidate',type=Path,required=True)
ap.add_argument('--js',type=Path,required=True)
args=ap.parse_args()
data=np.load(args.candidate/'review-data.npz')
raw=data['faces'].astype('<u2').tobytes()
offsets={}
base_offsets={}
for pose in ['T','ARMS_DOWN','RAISE','BEND']:
    base_offsets[pose]=len(raw)
    raw+=data[f'old_{pose}_4'].astype('<f4').tobytes()
for key in data.files:
    if key=='faces':continue
    offsets[key]=len(raw)
    pose=key[len('old_') if key.startswith('old_') else len('new_'):].rsplit('_',1)[0]
    # Lossless XOR against the neutral size of the same pose: pinned vertices
    # become zeros and compress without quantizing a single coordinate.
    current=data[key].astype('<f4').view('<u4')
    base=data[f'old_{pose}_4'].astype('<f4').view('<u4')
    raw+=np.bitwise_xor(current,base).tobytes()
meta=json.loads((args.candidate/'candidate.json').read_text())
payload={'sizes':[-1,-.75,-.5,-.25,0,.25,.5,.75,1,1.25,1.5,1.75,2],
    'vertexCount':len(data['new_T_0']),'faceCount':len(data['faces']),
    'offsets':offsets,'baseOffsets':base_offsets,
    'source':{'candidate':meta['source'],'parent':meta['parent']},
    'payload':base64.b64encode(gzip.compress(raw,compresslevel=6)).decode()}
template=Path(__file__).with_name('preview.html').read_text()
html=template.replace('__DATA__',json.dumps(payload,separators=(',',':'))).replace('__JS__',args.js.read_text())
dest=args.candidate.parent/'breast-shape-preview.html'
dest.write_text(html)
print(dest,dest.stat().st_size)
