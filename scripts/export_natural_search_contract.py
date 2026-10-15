"""Export the standalone OpenAPI contract without credentials/model loading."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api.natural_search_app import app

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('docs/openapi/natural-search-v1.json'))
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(app.openapi(),ensure_ascii=False,indent=2)+'\n')
    print(args.output)
