"""Regenerate SSR2 from candidate surfaces and all actual published presets.

A new file/source hash is mandatory. Quality is measured; this utility does not
modify any catalog, CSP or skin certification pin. A generated SSR2 is a review
artifact until its independent error report meets the existing preview policy.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/character_upgrade'))
import build_sampled_runtime as build
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--registry',required=True);p.add_argument('--gender',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=Path(a.candidate);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);r=json.loads(Path(a.registry).read_text());rules='standin-body-0.8.0' if a.gender=='male' else 'standin-female-body-0.4.0';presets={v['preset']:(v['preset'],v['defaults']) for cid,v in r['characters'].items() if cid.startswith(a.gender+'-') and v['rules']==rules}
original=build.load
def load(*args):
 c=original(*args);c.PRESETS=presets;return c
build.load=load
build.main(['--blend',str(folder/'model.blend'),'--controller',str(folder/'controller.py'),'--out',str(out/'preview.ssr'),'--report',str(out/'build.json'),'--axis-knots','5','--axis-refine','1','--pair-budget','12','--pair-dense','4','--max-basis','96','--val-few','6','--val-basic','4','--val-all','4'])
