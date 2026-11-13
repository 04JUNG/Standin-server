"""Server-authored child job only. No arbitrary client paths or executable code."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from converter.interactive_adapter import prepare_body,solve_from_body,STATE_FILES
from converter.hand_contact import SelectedHandContactError
from converter.solved_state import decode
args=sys.argv[sys.argv.index('--')+1:]
# Optional sixth argument: canonical state encoding chosen by the API process.
stage,job,identity,base,destination=args[:5]
encoding=args[5] if len(args)==6 else 'typed-f64be-json-v1'
if len(args) not in (5,6) or encoding not in STATE_FILES:raise ValueError('invalid worker arguments')
identity=decode(Path(identity).read_bytes())
if stage == 'prepare':
    prepare_body(job,identity,base)
elif stage == 'solve':
    try:
        solve_from_body(base,identity,destination,checkpoint=lambda:None,encoding=encoding)
    except SelectedHandContactError as exc:
        (Path(destination).parent/'hand-contact-error').write_text(exc.code)
        raise
else:
    raise ValueError('invalid stage')
