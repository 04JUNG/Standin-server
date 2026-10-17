import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from standin_eval.z_diagnostics import fit_distances, foreground_bbox, undo_quarter_turn, infer_pose


def test_fit_removes_translation_scale_without_mirroring():
    rng=np.random.default_rng(71)
    pose=rng.normal(size=(17,2)).astype(np.float32)
    mirror=pose.copy()
    mirror[:,0]*=-1
    library=np.stack((pose,mirror))
    values=fit_distances(library,pose*240+[520,610],np.ones(17,bool))
    assert values[0]<1e-5
    assert values[1]>.5


def test_rotation_inverse_in_nonsquare_image():
    width,height=80,50
    original=np.array([[13,21],[70,4]],dtype=float)
    for k in range(4):
        im=np.zeros((height,width),np.uint8)
        im[int(original[0,1]),int(original[0,0])]=1
        rotated=np.rot90(im,k)
        y,x=np.where(rotated==1)
        got=undo_quarter_turn(np.column_stack((x,y)),k,width,height)
        assert np.array_equal(got[0],original[0])


def test_crop_uses_pixels_and_falls_back_on_blank_input():
    im=np.full((100,80,3),255,np.uint8)
    assert foreground_bbox(im)==[0,0,80,100]
    im[20:70,30:50]=80
    assert foreground_bbox(im)==[30,20,50,70]


def test_rotation_selection_uses_confidence_and_restores_pixel_frame():
    class FakePose:
        calls=0
        def __call__(self,image,**kwargs):
            turn=self.calls
            self.calls+=1
            # The second hypothesis wins solely by body confidence.
            point=[21,66] if turn==1 else [0,0]
            return np.tile(point,(1,17,1)),np.full((1,17),.8 if turn==1 else .4)
    got=infer_pose(np.full((50,80,3),255,np.uint8),FakePose(),
                   turns=(0,1,2,3),is_3d=False)
    assert got['turn']==1
    assert got['keypoints2d'][0]==[13.,21.]
    assert got['z_meters']==[0.]*17


if __name__=="__main__":
    test_fit_removes_translation_scale_without_mirroring()
    test_rotation_inverse_in_nonsquare_image()
    test_crop_uses_pixels_and_falls_back_on_blank_input()
    test_rotation_selection_uses_confidence_and_restores_pixel_frame()
    print("PASS 4 diagnostic input/search contracts")
