import unittest
import numpy as np
from converter.local.surface import Neighborhood
from scripts.local_converter.experiment_local_motion_fit import fit

class LocalMotionFitTests(unittest.TestCase):
 def test_rigid_motion_preserves_authored_relief(self):
  n=9;r=np.array([(i,j,.2*np.sin(i/2)*np.cos(j/3)) for i in range(n) for j in range(n)],float);f=[]
  for i in range(n-1):
   for j in range(n-1):
    a=i*n+j;f.extend([(a,a+1,a+n),(a+1,a+n+1,a+n)])
  g=Neighborhood(np.array(f),r);angle=.7;rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]);p=r@rot.T+[20,-3,8]
  np.testing.assert_allclose(fit(g,r,p,1.),p,atol=1e-7)

if __name__=='__main__':unittest.main()
