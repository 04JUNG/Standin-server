"""Only run after the user supplies an actual exported evaluation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from standin_eval.blind_review import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--ratings',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    public=json.loads((a.study/'review-data.json').read_text())
    key=json.loads((a.study/'blinding-key-private.json').read_text())
    result=summarize(public,key,json.loads(a.ratings.read_text()))
    if a.out.exists():raise FileExistsError('preserve existing evaluation; choose a new output file')
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2,ensure_ascii=False))
