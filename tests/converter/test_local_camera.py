import copy
import unittest
from converter.local.camera import CAMERA_VERSION, SOURCE_CAMERA, validate_camera
from converter.local.contract import LocalError, digest, snapshot, versions, validate_request, PROTOCOL
class PersonalCameraTests(unittest.TestCase):
    def test_rotation_changes_snapshot_without_changing_pose_or_model(self):
        model = {"localModelId":"model", "sourceHash":"a"*64, "versions":versions()}
        source = snapshot(model,"b"*64,{"left":None,"right":None},SOURCE_CAMERA)
        rotated = {"version":CAMERA_VERSION,"sourceBvhSha256":"c"*64,"rotation":[[0,0,1],[0,1,0],[-1,0,0]]}
        self.assertNotEqual(digest(source),digest(snapshot(model,"b"*64,{"left":None,"right":None},rotated)))
        captured = validate_camera(rotated)
        rotated['rotation'][0][0] = 4
        self.assertEqual(captured['rotation'][0][0],0)
    def test_invalid_camera_is_never_silently_source_orientation(self):
        proper = {"version":CAMERA_VERSION,"sourceBvhSha256":"a"*64,"rotation":[[1,0,0],[0,1,0],[0,0,1]]}
        for change in ({"version":"unknown"},{"sourceBvhSha256":None},{"rotation":[[-1,0,0],[0,1,0],[0,0,1]]},{"rotation":[[2,0,0],[0,1,0],[0,0,1]]},{"rotation":[[float('nan'),0,0],[0,1,0],[0,0,1]]}):
            with self.subTest(change=change), self.assertRaises(LocalError): validate_camera({**proper,**change})
        with self.assertRaises(LocalError): validate_camera(None)
    def test_apply_requires_explicit_camera(self):
        request={"protocolVersion":PROTOCOL,"requestId":"apply","action":"apply","localModelId":"model","sourceHash":"a"*64,"posePath":"/pose.bvh","poseHash":"b"*64,"selectionId":"s","hands":{"left":None,"right":None}}
        with self.assertRaises(LocalError): validate_request(request)
        validate_request({**request,"camera":copy.deepcopy(SOURCE_CAMERA)})
if __name__=='__main__': unittest.main()
