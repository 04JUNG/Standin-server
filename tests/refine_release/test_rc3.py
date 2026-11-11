import unittest
import numpy as np
import test_policy
from policy import articulation
class BakedRigTests(unittest.TestCase):
 def test_baked_crouch_is_not_anatomical_neutral(self):
  _,ix,w,p=test_policy.SafetyTests().skeleton();w[ix['RightFoot'],:3,3]=[ -.15,-.5,.5]
  result=articulation(w,w,p,ix,neutral=w)
  self.assertTrue(result['source_pose_baked_into_rest'])
  self.assertFalse(any(x['kind']=='hinge_plane_reversal' for x in result['issues']))
 def test_baked_source_still_blocks_new_reverse_bend(self):
  _,ix,w,p=test_policy.SafetyTests().skeleton();w[ix['RightFoot'],:3,3]=[-.15,-.5,.5];candidate=w.copy();candidate[ix['RightFoot'],2,3]=-.5
  result=articulation(candidate,w,p,ix,neutral=w)
  self.assertTrue(any(x['kind']=='hinge_plane_reversal' and x['bone']=='RightLeg' for x in result['issues']))
 def test_anatomical_neutral_still_uses_calibrated_axes(self):
  _,ix,w,p=test_policy.SafetyTests().skeleton();candidate=w.copy();candidate[ix['RightFoot'],2,3]=.5
  result=articulation(candidate,w,p,ix,neutral=w)
  self.assertFalse(result['source_pose_baked_into_rest']);self.assertTrue(result['calibrated'])

class FailureContracts(unittest.TestCase):
 def test_empty_candidate_pool_writes_withheld_decision(self):
  import tempfile,json,sys
  from pathlib import Path
  from unittest.mock import patch
  import run
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'gate'
   with patch.object(sys,'argv',['gate','--run',str(root/'missing-run'),'--assets',str(root/'assets'),'--out',str(out),'--cases','sample']):run.main()
   decision=json.loads((out/'sample/decision.json').read_text())
   self.assertFalse(decision['release_eligible']);self.assertIsNone(decision['selected']);self.assertEqual(decision['reason'],'no_valid_candidate')
 def test_extractor_abstention_does_not_call_vlm(self):
  import importlib.util,tempfile,json,sys
  from pathlib import Path
  from unittest.mock import patch
  from types import SimpleNamespace
  path=Path(__file__).resolve().parents[2]/'services/refine_release/runtime/run_minute.py';spec=importlib.util.spec_from_file_location('minute_rc3',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  def abstain(args,**kwargs):
   target=Path(args[args.index('--out')+1]);target.mkdir(parents=True);(target/'sample-diagnostic.json').write_text('{}');return SimpleNamespace(returncode=42)
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'result'
   argv=['minute','--assets',d,'--out',str(out),'--case','sample','--extractor-checkout',d,'--model-manifest',d]
   with patch.object(sys,'argv',argv),patch.object(m.subprocess,'run',side_effect=abstain) as call:m.main()
   self.assertEqual(call.call_count,1);decision=json.loads((out/'gate/sample/decision.json').read_text());self.assertEqual(decision['new_vlm_calls'],0);self.assertFalse(decision['release_eligible'])
