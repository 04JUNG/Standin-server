"""Fail-closed provenance/correspondence and honest paired-timing boundaries."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from scripts.character_lod.contract import require_hash,require_subset,require_paired_runs

class CharacterLODContract(unittest.TestCase):
    def test_projection_is_fixed_normalized_and_protected_points_are_exact(self):
        import numpy as np
        from scripts.character_lod.correspondence import Correspondence
        mapping=Correspondence([[0,0,0],[0,1,2]],[[1,0,0],[.25,.25,.5]])
        values=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        np.testing.assert_array_equal(mapping.apply(values),[[0,0,0],[.5,1,0]])
        self.assertEqual(mapping.exact_mask([0]).tolist(),[True,False])
        for bad in [[[1,0,0],[.5,.5,.5]],[[1,0,0],[-.1,.1,1]],[[1,0,0],[float('nan'),0,1]]]:
            with self.assertRaises(ValueError):Correspondence([[0,0,0],[0,1,2]],bad)

    def test_frame_offset_tracks_translation_and_scale_without_changing_weights(self):
        import numpy as np
        from scripts.character_lod.correspondence import Correspondence
        mapping=Correspondence([[0,1,2]],[[.5,.25,.25]],[[0,0,.1]])
        triangle=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        np.testing.assert_allclose(mapping.apply(triangle),[[.25,.25,.1]])
        shift=np.array([2.,3.,4.]);np.testing.assert_allclose(mapping.apply(triangle*2+shift),[[.5,.5,.2]]+shift)
        self.assertFalse(mapping.exact_mask([0,1,2])[0])
        np.testing.assert_allclose(mapping.linear(np.ones((3,5))),np.ones((1,5)))

    def test_final_screen_cannot_hide_hand_or_body_error_in_global_percentile(self):
        from scripts.character_lod.contract import final_geometry_pass
        good=dict(retained_world_p99_mm=0.,protected_max_mm=0.,hand_max_mm=0.,body_surface_p99_mm=.4)
        self.assertTrue(final_geometry_pass(good))
        for key,value in [('hand_max_mm',.06),('body_surface_p99_mm',.51),('protected_max_mm',float('nan'))]:
            self.assertFalse(final_geometry_pass({**good,key:value}))
        self.assertFalse(final_geometry_pass({}))

    def test_hand_graph_mask_includes_blended_wrist_and_regions_partition(self):
        import numpy as np
        from scripts.character_lod.region_policy import hand_graph_mask,region_labels,region_counts
        names=['Spine','LeftForeArm','LeftHand','LeftHandIndex1','LeftFoot','LeftUpLeg']
        weights=np.array([[1,0,0,0,0,0],[0,.98,.02,0,0,0],[0,0,.5,.5,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.assertEqual(hand_graph_mask(names,weights).tolist(),[False,True,True,False,False])
        counts=region_counts(region_labels(names,weights))
        self.assertEqual(counts,dict(hands=1,feet=1,head_neck=0,arms=1,legs=1,torso=1))
        self.assertEqual(sum(counts.values()),len(weights))

    def test_neutral_and_body_views_never_curl_fingers(self):
        from scripts.character_lod.pose_controls import finger_test_rotations
        for pose in ('neutral','raised','elbow','deep','sit','cross'):
            self.assertEqual(finger_test_rotations(pose), ())
        self.assertTrue(any(name.startswith('LeftHand') and angle==75
            for name,axis,angle in finger_test_rotations('hands')))

    def test_unapproved_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.blend';p.write_bytes(b'approved')
            digest=hashlib.sha256(p.read_bytes()).hexdigest();require_hash(p,digest)
            p.write_bytes(b'substituted')
            with self.assertRaises(ValueError):require_hash(p,digest)

    def test_protected_connectivity_and_identity(self):
        require_subset([0,1,2,4],[0],[[0,1,2],[1,3,4]],[[0,1,2],[1,2,3]])
        for ids,faces in [([0,1,1],[[0,1,2]]),([1,2,3],[[0,1,2]]),([0,1,2],[[0,1,1]]),([0,1,2],[[0,2,1]])]:
            with self.assertRaises(ValueError):require_subset(ids,[0],[[0,1,2]],faces)

    def test_incomplete_and_mixed_measurements_cannot_support_claim(self):
        rows=[dict(gender='male',case='base',skin='default',variant=v,trial=i,ok=True,controls={'muscle':.5},bvh_sha256='a'*64,report={'shape_bake':{'neutral_preparation':{'status':'prepared'}}}) for v in ('original','candidate') for i in range(3)]
        require_paired_runs(rows)
        with self.assertRaises(ValueError):require_paired_runs(rows[:-1])
        bad=copy.deepcopy(rows);bad[-1]['controls']['muscle']=.8
        with self.assertRaises(ValueError):require_paired_runs(bad)
        bad=copy.deepcopy(rows);bad[-1]['report']['shape_bake']['neutral_preparation']['status']='generated'
        with self.assertRaises(ValueError):require_paired_runs(bad)
        bad=copy.deepcopy(rows);bad[-1]['bvh_sha256']='b'*64
        with self.assertRaises(ValueError):require_paired_runs(bad)

if __name__=='__main__':unittest.main()
