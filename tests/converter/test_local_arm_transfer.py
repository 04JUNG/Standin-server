"""Small geometric cases for the local-only arm correction boundary."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class LocalArmTransferTests(unittest.TestCase):
    def setUp(self):
        try:
            from converter.local import arm_transfer
            from mathutils import Matrix
        except ImportError:
            self.skipTest("Requires Blender")
        self.transfer, self.Matrix = arm_transfer, Matrix

    def frame(self, points):
        table = {name: name for name in points}
        with patch.object(self.transfer.rt, "_rest_world",
                          side_effect=lambda arm, name: self.Matrix.Translation(points[name])):
            return self.transfer.torso_frame(None, table, True)

    def test_coincident_clavicle_roots_use_actual_upperarm_span(self):
        points = {"spine": (0, 0, 0), "neck": (0, 0, 1),
                  "shoulder.L": (0, 0, .8), "shoulder.R": (0, 0, .8),
                  "upperarm.L": (.3, 0, .8), "upperarm.R": (-.3, 0, .8)}
        fallback = self.frame(points)
        points["shoulder.L"], points["shoulder.R"] = (.3, 0, .8), (-.3, 0, .8)
        ordinary = self.frame(points)
        self.assertIsNotNone(fallback)
        self.assertLess(self.transfer.rt._rotation_error_deg(fallback, ordinary), .001)
        self.assertAlmostEqual(fallback.to_3x3().determinant(), 1., places=5)

    def test_no_observable_lateral_span_keeps_ambiguous_frame(self):
        points = {"spine": (0, 0, 0), "neck": (0, 0, 1),
                  "shoulder.L": (0, 0, .8), "shoulder.R": (0, 0, .8),
                  "upperarm.L": (0, 0, .8), "upperarm.R": (0, 0, .8)}
        self.assertIsNone(self.frame(points))

    def test_direction_candidate_rolls_back_when_surface_repair_is_lost(self):
        import numpy as np
        from converter.local import surface

        bone = SimpleNamespace(name="arbitrary", matrix_basis=self.Matrix.Identity(4))
        arm = SimpleNamespace(pose=SimpleNamespace(bones=[bone]))
        mesh = SimpleNamespace(name="unnamed", data=None)
        points = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0]])
        reference = SimpleNamespace(triangles=np.array([[0, 1, 2]]), anchors=np.zeros(3, bool))

        def transfer(src, dst, meshes, neutral, report, side, allow_direction_only):
            if allow_direction_only:
                bone.matrix_basis = self.Matrix.Translation((1, 0, 0))
                return {"applied": True, "palmFrameValidated": False,
                        "handFramePolicy": "preserve_world_orientation"}
            return {"applied": False, "reason": "palm_basis_unavailable"}

        with patch.object(self.transfer, "within_budget", return_value=True), \
                patch.object(self.transfer, "transfer_side", side_effect=transfer), \
                patch.object(self.transfer, "evaluated_points", return_value=points), \
                patch.object(surface, "prepare_reference", return_value=reference), \
                patch.object(surface, "correct", return_value=(points, {"parts": [
                    {"applied": False, "reason": "metrics_regressed"}]})):
            result = self.transfer.apply(None, arm, [mesh], {mesh.name: points}, None)
        self.assertEqual(tuple(bone.matrix_basis.translation), (0., 0., 0.))
        for side in ("L", "R"):
            self.assertFalse(result[side]["applied"])
            self.assertEqual(result[side]["reason"], "surface_candidate_rejected")
            self.assertFalse(result[side]["rejectedCandidate"]["palmFrameValidated"])

    def test_new_direction_path_rejects_a_new_crease_without_extreme_stretch(self):
        import numpy as np

        rest = np.array([(x, y, 0.) for x in range(3) for y in range(3)])
        faces = np.array([(a, a+1, a+3) for a in (0, 1, 3, 4)] +
                         [(a+1, a+4, a+3) for a in (0, 1, 3, 4)])
        candidate = rest.copy()
        candidate[4, 2] = .6
        samples = [(None, faces, rest, rest)]
        with patch.object(self.transfer, "evaluated_points", return_value=candidate):
            # Existing landmark-based behavior remains unchanged; the new
            # unobserved-palm path adds a shape gate on the same data.
            self.assertTrue(self.transfer.skin_safe(samples))
            self.assertFalse(self.transfer.skin_safe(samples, strict=True))


if __name__ == "__main__":
    unittest.main()
