"""Unified terminal stage: paired automatic candidates -> guarded local repair -> FBX.
No VLM calls or search occur here. Never overwrites the input run.
"""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
 cmd=[sys.executable,str(Path(__file__).parent/'experiments/refine_final_gate/run.py'),*sys.argv[1:]]
 for flag in ['--repair','--export']:
  if flag not in cmd:cmd.append(flag)
 raise SystemExit(subprocess.call(cmd))
