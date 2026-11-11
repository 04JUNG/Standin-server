"""Server-authored child job only. No arbitrary client paths or executable code."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from converter.interactive_adapter import prepare_body,solve_from_body
from converter.hand_contact import SelectedHandContactError
from converter.solved_state import decode
args=sys.argv[sys.argv.index('--')+1:]
stage,job,identity,base,destination=args
identity=decode(Path(identity).read_bytes())
if stage == 'prepare':
    prepare_body(job,identity,base)
elif stage == 'solve':
    try:
        solve_from_body(base,identity,destination,checkpoint=lambda:None)
    except SelectedHandContactError as exc:
        (Path(destination).parent/'hand-contact-error').write_text(exc.code)
        raise
else:
    raise ValueError('invalid stage')
