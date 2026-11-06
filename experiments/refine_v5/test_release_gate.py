import tempfile,unittest
from pathlib import Path
from experiments.refine_v5.release_gate import finalize,sha
from experiments.refine_v5.target_safety import skin_nonregression
class ReleaseTests(unittest.TestCase):
    def test_forged_pass_without_registered_profile_restores_exact_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);src=p/'base';candidate=p/'candidate';dest=p/'out';src.write_bytes(b'original FBX bytes');candidate.write_bytes(b'bad changed bytes')
            audit=dict(source_sha256=sha(src),candidate_sha256=sha(candidate),native_validated=True,skin=dict(complete_collision_certificate=True))
            result=finalize(src,candidate,dest,audit,'a'*64)
            self.assertEqual(result['status'],'fallback_exact_B0');self.assertEqual(dest.read_bytes(),src.read_bytes())
            with self.assertRaises(ValueError):finalize(src,candidate,src,audit,'a'*64)
    def test_new_penetration_is_not_hidden_by_fewer_existing_collisions(self):
        a=dict(collisions=[dict(regions=['hand','head'],max_depth_mm=4,penetrating_vertices=10)],region_closure_failures={})
        b=dict(collisions=[dict(regions=['hand','torso'],max_depth_mm=3,penetrating_vertices=5)],region_closure_failures={})
        self.assertTrue(skin_nonregression(a,b));self.assertEqual(skin_nonregression(a,a),[])
if __name__=='__main__':unittest.main()
