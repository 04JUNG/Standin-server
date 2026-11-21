"""Pin deployed extractor sources, including files outside the Git index."""
import argparse, json
from pathlib import Path
from .service import sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    files=list((a.checkout/'src').rglob('*.py'))+[a.checkout/'experiments/refine_v5/develop_skeleton_seeds.py']
    with a.out.open('x') as f:json.dump({str(p.relative_to(a.checkout)):sha(p) for p in sorted(files)},f,indent=2)
    print(sha(a.out))
