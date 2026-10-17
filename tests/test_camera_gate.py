import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.camera_gate import use_expanded


def test_good_existing_match_is_preserved():
    assert not use_expanded(.1,.01,.15,.15)
    assert not use_expanded(0,0,0,0)


def test_expansion_needs_the_frozen_relative_gain():
    assert use_expanded(.4,.2,.3,.3)
    assert not use_expanded(.4,.35,.3,.3)
    assert not use_expanded(.4,.41,.3,.15)


def test_invalid_input_does_not_force_expansion():
    assert not use_expanded(float('nan'),.1,.3,.15)
    assert not use_expanded(float('inf'),.1,.3,.15)
    assert not use_expanded(-1,.1,.3,.15)


if __name__=='__main__':
    for name,f in sorted(list(globals().items())):
        if name.startswith('test_'):f();print('PASS',name)
