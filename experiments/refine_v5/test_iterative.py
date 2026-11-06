import unittest
import numpy as np
from experiments.refine_v5.iterative import solve,hierarchy_digest
from experiments.refine_v5.goal_probe_v2 import prepare
from experiments.refine_v5.learning_data import axes,frame,POINTS

def fixture():
    positions=[('Hips',None,[0,0,0]),('Spine','Hips',[0,0,.15]),('Spine1','Spine',[0,0,.3]),('Spine2','Spine1',[0,0,.45]),('Head','Spine2',[0,0,.7])]
    for side,sign in [('Left',1),('Right',-1)]:
        positions += [(side+'Shoulder','Spine2',[sign*.15,0,.45]),(side+'Arm',side+'Shoulder',[sign*.25,0,.45]),(side+'ForeArm',side+'Arm',[sign*.5,0,.4]),(side+'Hand',side+'ForeArm',[sign*.7,0,.3]),(side+'UpLeg','Hips',[sign*.1,0,0]),(side+'Leg',side+'UpLeg',[sign*.1,0,-.35]),(side+'Foot',side+'Leg',[sign*.1,0,-.7])]
    bones=[]
    for name,parent,pos in positions:
        m=np.eye(4);m[:3,3]=pos;bones.append(dict(name=name,parent=parent,matrix=m.tolist()))
    rig=dict(bones=bones,source_sha256='a'*64)
    by,c,s,b=frame(bones);cam=np.array([[1,0,0],[0,0,-1],[0,1,0]])@b
    p=(np.array([np.array(by[n]['matrix'])[:3,3] for n in POINTS])-c)@b@cam.T;p=p[:,:2];q=(p-p[[0,3]].mean(0))/np.linalg.norm(p[0]-p[3]);q[2,1]-=.1
    evidence=dict(points=q.tolist(),point_mask=[True]*6+[False]*6,part_mask=[True]*4+[False]*4+[True]*5,camera=cam.tolist(),ownership_verified=True,support_verified=False)
    prediction=dict(axes=axes(bones).tolist(),uncertainty_rad=[.5]*13)
    profile=dict(name='conservative12',body_asset_sha256='b'*64,hierarchy_sha256=hierarchy_digest(bones),offline_only=True)
    return rig,evidence,prediction,profile

class IterationTests(unittest.TestCase):
    def test_unknown_evidence_rejects_gt(self):
        r,e,p,f=fixture();e['gt']='x'
        with self.assertRaises(ValueError):solve(r,e,p,profile=f)
    def test_ownership_and_timeout_are_exact_B0(self):
        r,e,p,f=fixture();e['ownership_verified']=False
        out=solve(r,e,p,profile=f)
        self.assertEqual(out['status'],'fallback_ownership')
        e['ownership_verified']=True;out=solve(r,e,p,profile=f,seconds=0)
        self.assertTrue(np.array_equal(np.array(out['delta_rotvec']),np.zeros((4,3))))
        for b in r['bones']:self.assertEqual(out['planned_heads'][b['name']],np.array(b['matrix'])[:3,3].tolist())
    def test_cumulative_bounds_and_source_target_separation(self):
        r,e,p,f=fixture();out=solve(r,e,p,profile=f,rounds=3,max_nfev=8)
        self.assertFalse(out['target_applied']);self.assertFalse(set(out['active_bones'])&{'Head','LeftHand','RightHand','LeftFoot','RightFoot'})
        self.assertLessEqual(np.linalg.norm(out['delta_rotvec'],axis=1).max(),np.radians(12)+1e-6)
        for n in ['LeftLeg','RightLeg','LeftFoot','RightFoot']:
            original=next(b['matrix'] for b in r['bones'] if b['name']==n)
            self.assertEqual(out['planned_heads'][n],np.array(original)[:3,3].tolist())
    def test_stale_selected_profile_is_rejected(self):
        r,e,p,f=fixture();context=dict(kind='offline_fixture_selection',source_sha256='c'*64,body_asset_sha256='b'*64)
        with self.assertRaises(ValueError):solve(r,e,p,profile=f,selection=context)
    def test_rediagnosis_failure_keeps_last_valid(self):
        r,e,p,f=fixture()
        def broken(_):raise RuntimeError('model unavailable')
        out=solve(r,e,p,profile=f,redetect=broken)
        self.assertEqual(out['status'],'fallback_exact_B0');self.assertEqual(out['trace'][0]['status'],'rediagnosis_failure_keep_best')
    def test_iteration_cap(self):
        r,e,p,f=fixture()
        with self.assertRaises(ValueError):solve(r,e,p,profile=f,rounds=4)


class SourceIsolationTests(unittest.TestCase):
    def test_target_detail_proposals_do_not_influence_source_solve(self):
        r,e,p,f=fixture();first=solve(r,e,p,profile=f,max_nfev=10)
        target=np.array(p['axes']);target[8:]*=-1;p['axes']=target.tolist()
        second=solve(r,e,p,profile=f,max_nfev=10)
        self.assertEqual(first['planned_heads'],second['planned_heads'])
    def test_joint_proposals_keep_lengths_and_rotation_identity(self):
        from experiments.refine_v5.proposals import derive
        r,e,p,f=fixture();out=derive(r['bones'],p['axes']);points=np.array(out['body_relative_joint_targets'])
        _,ix,_,_,base=prepare(r);_,_,scale,_=frame(r['bones'])
        for a,b in [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]:
            length=np.linalg.norm(base[ix[POINTS[b]],:3,3]-base[ix[POINTS[a]],:3,3])/scale
            self.assertAlmostEqual(np.linalg.norm(points[a]-points[b]),length,places=6)
        self.assertLess(max(np.linalg.norm(v) for v in out['local_SO3_residual_proposals'].values()),1e-6)

if __name__=='__main__':unittest.main()
