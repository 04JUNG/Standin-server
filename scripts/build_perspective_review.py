"""Refresh presentation without redoing inference or changing rankings."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_z_rendered import sha256,save_json


STANDALONE_STYLE="""
:root{color-scheme:light dark;--background:light-dark(#fff,#171717);--foreground:light-dark(#202124,#eee);--border:light-dark(#ddd,#444);--primary:light-dark(#007bff,#329cff);--primary-foreground:#fff;--font-size-base:14px}
body{margin:16px;background:var(--background);color:var(--foreground);font:var(--font-size-base) system-ui,sans-serif}
*{box-sizing:border-box}[hidden]{display:none!important}
.viz-controls,.viz-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.viz-controls{margin:12px 0}.form-label{display:flex;flex-direction:column;gap:6px;flex:1;min-width:180px}
.form-check{display:flex;align-items:center;gap:8px}.form-control,.form-select{padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--background);color:var(--foreground);font:inherit;max-width:100%}
.btn{padding:9px 14px;border:1px solid var(--border);border-radius:20px;color:var(--foreground);background:var(--background);font:inherit;cursor:pointer}
.btn-primary,.btn[aria-pressed=true]{background:var(--primary);color:var(--primary-foreground)}
.text-small{font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{width:100%}
"""


def build(study,output,standalone=None):
    data=json.loads((study/'review-data-private.json').read_text())
    summary=json.loads((study/'summary.json').read_text())
    assert sha256(study/'rankings.json')==summary['rankings_sha256']
    template=ROOT/'scripts/templates/perspective_rough_review.html'
    fragment=template.read_text().replace('__REVIEW_DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    assert len(fragment.encode())<1_000_000
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(fragment,encoding='utf-8')
    if standalone:
        document='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>러프 검색 비포 애프터 검수</title><style>'+STANDALONE_STYLE+'</style></head><body>'+fragment+'</body></html>'
        standalone.write_text(document,encoding='utf-8')
    save_json(study/'presentation.json',{'template_at_run_sha256':json.loads((study/'protocol.json').read_text())['files']['scripts/templates/perspective_rough_review.html'],
        'template_now_sha256':sha256(template),'rankings_sha256':summary['rankings_sha256'],
        'data_sha256':sha256(study/'review-data-private.json'),'fragment_sha256':sha256(output),
        'changes':'clear raw-template error; default first searchable case; search settings and results unchanged'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--standalone',type=Path)
    a=p.parse_args();build(a.study,a.output,a.standalone)
