"""Allow the standalone release contract suite to be collected by repository pytest."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'services/refine_release/runtime/experiments/refine_final_gate'))
