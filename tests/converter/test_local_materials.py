import copy
import unittest

from converter.local.contract import LocalError
from converter.local.materials import specular_ior_level, compare_materials
from converter.local.rig_profiles import skin_probe_names, register
from converter import bone_map


class SolidSpecularTests(unittest.TestCase):
    def test_skin_probes_use_known_joint_roles_for_actual_mixamo_and_legacy_cs3c(self):
        register()
        self.assertEqual(skin_probe_names(bone_map.MIXAMO.values()),
                         ("mixamorig:LeftForeArm", "mixamorig:RightForeArm", "mixamorig:LeftLeg", "mixamorig:RightLeg", "mixamorig:Spine"))
        legacy = bone_map.PROFILES['cs3c_humanoid_v2']
        self.assertEqual(skin_probe_names(legacy.values()),
                         ("LeftLowerArm", "RightLowerArm", "LeftLowerLeg", "RightLowerLeg", "Spine"))
        with self.assertRaises(LocalError): skin_probe_names(["unknown"])

    def test_actual_standin_reflectance_and_matte_are_representable(self):
        self.assertAlmostEqual(specular_ior_level(
            (.47999998927116394, .5299999713897705, .5899999737739563),
            (.11999999731779099, .13249999284744263, .14749999344348907)), .5)
        self.assertEqual(specular_ior_level((0, 0, 0), (0, 0, 0)), 0)
        self.assertEqual(specular_ior_level((.4, .5, .6), (0, 0, 0)), 0)

    def test_unrepresentable_or_invalid_colors_are_not_averaged_or_clamped(self):
        for diffuse, specular in [((0, 0, 0), (.1, .1, .1)),
                                  ((.4, .5, .6), (.2, .1, .3)),
                                  ((.4, .5, .6), (.4, .5, .6)),
                                  ((.4, .5, .6), (float('nan'), 0, 0)),
                                  ((.4, .5, .6), (-.1, 0, 0))]:
            with self.subTest(diffuse=diffuse, specular=specular), self.assertRaises(LocalError):
                specular_ior_level(diffuse, specular)

    def test_export_check_detects_dropped_reflection_and_nonfinite_parameters(self):
        expected = {'material': {'diffuse': [.48, .53, .59], 'specular': [.12, .1325, .1475],
                                 'shininess': 10.24, 'opacity': 1}}
        self.assertTrue(compare_materials(expected, copy.deepcopy(expected))['passed'])
        for value in ([0, 0, 0], [float('nan'), .1325, .1475]):
            actual = copy.deepcopy(expected)
            actual['material']['specular'] = value
            with self.assertRaises(LocalError) as error:
                compare_materials(expected, actual)
            self.assertEqual(error.exception.code, 'ROUNDTRIP_FAILED')


if __name__ == '__main__': unittest.main()
