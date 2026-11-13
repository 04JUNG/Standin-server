"""Side-by-side unhighlighted actual-material renders of experimental outputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_mesh_audit import render


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);directory=args.run_dir
    result=json.loads((directory/'surface-only-experiment.json').read_text())
    audit=json.loads((directory/'mesh-audit.json').read_text())['models']
    for p,row in enumerate(result['poses']):
        if 'experimentalFile' not in row:continue
        config={'worstStretchEdges':audit[p]['worstStretchEdges'],'newCrossingFaces':[]}
        offset=(0,1,-.25) if p==0 else (.2,-1,-.2)
        for label,path in [('before',row['sourceFile']),('normals',row['normalOnlyControl']['file']),('after',row['experimentalFile'])]:
            render(directory/path,config,directory/f'surface-only-{p}-{label}.png','hips',offset)
