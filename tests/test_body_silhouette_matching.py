import unittest
import numpy as np
from src.experimental.body_matching.silhouette_matching import extract_profile,rank_profiles

class SilhouetteTests(unittest.TestCase):
    def mask(self):
        m=np.zeros((220,120),bool);m[10:210,40:80]=True;return m
    def profile(self,m=None,**kw):
        args=dict(full_body_visible=True,upright=True,foreshortening=False,ownership_verified=True);args.update(kw)
        return extract_profile(self.mask() if m is None else m,**args)
    def test_uniform_scale_and_translation_preserve_dimensionless_shape(self):
        a=self.profile();m=np.pad(np.repeat(np.repeat(self.mask(),2,0),2,1),((7,2),(5,3)))
        b=self.profile(m)
        self.assertLess(np.max(abs(np.array(a['widths'])-b['widths'])),.001)
    def test_crops_ownership_and_perspective_cannot_bypass_gate(self):
        for kw in (dict(full_body_visible=False),dict(upright=False),dict(foreshortening=True),dict(ownership_verified=False)):
            with self.assertRaises(ValueError):self.profile(**kw)
        with self.assertRaises(ValueError):self.profile(self.mask()[10:210])
    def test_multiple_views_do_not_duplicate_candidate_and_ties_stay_tied(self):
        p=self.profile();refs=[dict(body_id=b,view=v,profile=p) for b,v in [('a','front'),('a','side'),('b','front')]]
        r=rank_profiles(p,refs);self.assertEqual(len(r['candidates']),2);self.assertEqual(r['tied_body_ids'],['a','b'])
        self.assertIsNone(r['auto_body_id']);self.assertFalse(r['application_policy']['allowed'])
    def test_duplicate_reference_and_nonfinite_profile_rejected(self):
        p=self.profile();r=dict(body_id='a',view='front',profile=p)
        with self.assertRaises(ValueError):rank_profiles(p,[r,r])
        p['widths'][0]=float('nan')
        with self.assertRaises(ValueError):rank_profiles(p,[])
