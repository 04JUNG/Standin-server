"""Retry only generated export/audit/report stages; never rerun inference or solve."""
import argparse
from pathlib import Path
from pipeline import read,finish
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--blender',default=__import__('os').environ.get('REFINE_BLENDER','blender'));a=p.parse_args();rows=read(a.out/'status.json')['cases'] if (a.out/'status.json').exists() else read(a.out/'final/results.json');finish(a,rows)
