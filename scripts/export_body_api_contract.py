#!/usr/bin/env python3
"""Export only the body-selection schema and synthetic /analyze examples; no network."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.body_models import BodyMatchingOut
from tests.test_body_api_contract import BodyAPIContractTests


def main():
    out=Path(__file__).resolve().parents[1]/'docs/contracts/body'
    out.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save('BodyMatchingOut.schema.json',BodyMatchingOut.model_json_schema())
    case=BodyAPIContractTests();case.setUp()
    try:
        for mode in ['auto','shadow','off']:
            # Body-only fixtures avoid snapshotting unrelated inference contracts.
            body=case.response(mode)['body_matching']
            body.pop('elapsed_ms',None)
            save('body.'+mode+'.example.json',body)
    finally:case.doCleanups()
    save('manifest.json',dict(scope='Optional POST /analyze body_matching only',
        evidence='Synthetic contract fixtures, not real assets or accuracy results.',
        generation_command='python scripts/export_body_api_contract.py'))
    print(out)


if __name__=='__main__':main()
