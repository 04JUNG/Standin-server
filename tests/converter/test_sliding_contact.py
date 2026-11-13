import unittest
import numpy as np
from scripts.local_converter.experimental.sliding_contact import certificate
from scripts.local_converter.experimental.contact import contact_guard
from scripts.local_converter.experimental.contact_features import crossing_edges, measures
from scripts.local_converter.experimental.contact_path_limits import motion_limits


class SlidingTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[-2.,-1.,0.],[2.,-1.,0.],[0.,2.,0.]])
        self.b=np.array([[-.25,0.,-1.],[1.,0.,1.],[-1.,0.,1.]])

    def test_actual_slide_without_growth(self):
        self.assertTrue(certificate(self.a,self.a,self.b,self.b+[.125,0,0])[0])
        self.assertLessEqual(self.coefficients(self.a,self.b+[.125,0,0]).max(),1e-12)

    def coefficients(self,a,b):
        features=np.array([crossing_edges(self.a,self.b,4.)])
        return motion_limits(np.vstack((self.a,self.b)),np.vstack((a,b)),
                             np.arange(6).reshape(1,2,3),features,1.)

    def test_only_opt_in_accepts_persistent_sliding_pair(self):
        start=np.vstack((self.a,self.b))
        end=np.vstack((self.a,self.b+[.125,0,0]))
        faces=np.arange(6).reshape(2,3)
        self.assertFalse(contact_guard(start,end,faces)["passCheck"])
        result=contact_guard(start,end,faces,sliding=True)
        self.assertTrue(result["passCheck"])
        self.assertEqual(result["certifiedExistingPairs"],1)
        features=np.array([crossing_edges(self.a,self.b,4.)])
        initial=measures(start,faces[None],features)
        final=measures(end,faces[None],features)
        np.testing.assert_allclose(initial,final,atol=1e-12)

    def test_longer_intersection_with_equal_depths_is_rejected(self):
        end=self.b.copy()
        end[:,0]*=1.25
        self.assertEqual(certificate(self.a,self.a,self.b,end),(False,"length"))
        self.assertGreater(self.coefficients(self.a,end)[:14].max(),0)

    def test_growth_rejected(self):
        self.assertFalse(certificate(self.a,self.a*1.1,self.b,self.b*1.1)[0])

    def test_depth_can_worsen_between_two_bounded_endpoints(self):
        a=np.array([[-.9375,-.625,.0625],[.5625,-.0625,.0625],[-.375,.75,-.4375]])
        b=np.array([[-.375,.25,-.375],[.625,.3125,.9375],[-.75,-.25,.0625]])
        def depths(x,y):
            n=np.cross(y[1]-y[0],y[2]-y[0])
            return np.abs((x-y[0])@n)/np.linalg.norm(n)
        start=np.r_[depths(self.a,self.b),depths(self.b,self.a)]
        end=np.r_[depths(a,b),depths(b,a)]
        mid_a,mid_b=(self.a+a)/2,(self.b+b)/2
        self.assertTrue(np.all(end<start))
        self.assertTrue(np.any(np.r_[depths(mid_a,mid_b),depths(mid_b,mid_a)]>start))
        self.assertEqual(certificate(self.a,a,self.b,b),(False,"depth"))
        self.assertGreater(self.coefficients(a,b)[14:].max(),0)

    def test_ambiguous_initial_contact_fails_closed(self):
        self.assertFalse(certificate(self.a,self.a,self.a,self.a)[0])
