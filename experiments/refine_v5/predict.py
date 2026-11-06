"""Standalone inference: checkpoint + rough + current source state only."""
import argparse, json
from pathlib import Path
from .model import Corrector
ALLOWED = {'case_id', 'image', 'base_fbx', 'state', 'seed_kind'}

def predict_rows(model, rows):
    result = []
    for r in rows:
        if set(r) - ALLOWED:
            raise ValueError('unexpected inference field')
        result.append(dict(case_id=r['case_id'], base_fbx=r['base_fbx'], prediction=model.predict(r['image'], r['state']).tolist()))
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(predict_rows(Corrector.load(a.checkpoint), json.loads(Path(a.inputs).read_text())), indent=2))
if __name__ == '__main__':
    main()
