import unittest
import numpy as np
from src.experimental.body_matching.morphology import measure_neutral

class MorphologyTests(unittest.TestCase):
    def fixture(self):
        # Symmetric ellipsoid sections at known authored levels; not a person label.
        rows=[]
        for z in np.linspace(0,2,1001):
            width=.18 if z<1 else .20
            for angle in np.linspace(0,2*np.pi,16,endpoint=False):rows.append((width*np.cos(angle),.1*np.sin(angle),z))
        lm=dict(chin=1.75,hip=1.,shoulder=1.5,knee=.55,ankle=.1)
        joints=dict(LeftArm=[.15,0,1.5],RightArm=[-.15,0,1.5],LeftForeArm=[.2,0,1.5],LeftHand=[.3,0,1.5],LeftUpLeg=[.1,0,1],LeftLeg=[.1,0,.55],LeftFoot=[.1,0,.1])
        return np.array(rows),lm,joints
    def test_head_units_are_measured_and_translation_scale_invariant(self):
        co,lm,j=self.fixture();a=measure_neutral(co,lm,j)
        self.assertEqual(a['head_units'],8.)
        scale=100.;offset=7.
        q=co*scale;q[:,2]+=offset
        b=measure_neutral(q,{k:v*scale+offset for k,v in lm.items()},
            {k:[p[0]*scale,p[1]*scale,p[2]*scale+offset] for k,p in j.items()})
        self.assertEqual(a,b)
    def test_invalid_or_degenerate_inputs_fail_instead_of_creating_defaults(self):
        co,lm,j=self.fixture()
        for points,marks in ((co*float('nan'),lm),(co,{**lm,'chin':3}),(co[:4],lm)):
            with self.assertRaises(ValueError):measure_neutral(points,marks,j)
