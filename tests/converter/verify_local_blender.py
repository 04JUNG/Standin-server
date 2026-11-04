"""Run in Blender: bounded decompression and actual authored-rig regressions."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.local.contract import LIMITS, LocalError
from converter.local.inspection import bounded_fbx_parser, import_fbx, inspect, roundtrip
from scripts.local_converter.make_fixture import build
from converter.local.materials import specular_ior_level


class BlenderLocalTests(unittest.TestCase):
    def specular_fixture(self, root, lose_specular=False):
        build(root / "fixture.fbx")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm, meshes = import_fbx(root / "fixture.fbx")
        for mat in {s.material for m in meshes for s in m.material_slots}:
            mat.use_nodes = True
            shader = mat.node_tree.nodes.get("Principled BSDF")
            shader.inputs["Base Color"].default_value = (.48, .53, .59, 1)
            shader.inputs["Roughness"].default_value = .68
            shader.inputs["Specular IOR Level"].default_value = 0 if lose_specular else specular_ior_level((.48,.53,.59),(.12,.1325,.1475))
        character, _ = inspect(arm, meshes)
        expected = {name: {**mat, "fbx": {"diffuse":[.48,.53,.59],"specular":[.12,.1325,.1475],"shininess":10.24,"opacity":1}}
                    for name,mat in character["materials"].items()}
        return arm, meshes, character, expected

    def test_nonzero_source_specular_survives_real_fbx_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm, meshes, character, expected = self.specular_fixture(root)
            report = roundtrip(arm, meshes, root/"reflective.fbx", character["mapping"], expected)
            self.assertTrue(report["materialPreservation"]["passed"])
            self.assertLess(report["materialPreservation"]["maxSpecularError"], 1e-6)

    def test_real_export_of_dropped_specular_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm, meshes, character, expected = self.specular_fixture(root, lose_specular=True)
            with self.assertRaises(LocalError) as error:
                roundtrip(arm, meshes, root/"lost-specular.fbx", character["mapping"], expected)
            self.assertEqual(error.exception.code, "ROUNDTRIP_FAILED")

    def test_declared_array_limit_before_allocation(self):
        from io import BytesIO
        with bounded_fbx_parser() as parser:
            with self.assertRaises(LocalError) as error:
                parser.unpack_array(BytesIO(struct.pack("<III", LIMITS["expandedBytes"], 1, 8)).read, "d", 8, False)
        self.assertEqual(error.exception.code, "LIMIT_EXCEEDED")

    def test_actual_zlib_output_is_bounded_by_declaration(self):
        with bounded_fbx_parser() as parser:
            with self.assertRaises(LocalError) as error:
                parser._decompress_and_insert_array([None], 0, (zlib.compress(b"x" * 1024**2), 1, "i", 4, False))
        self.assertEqual(error.exception.code, "LIMIT_EXCEEDED")

    def test_multimesh_transforms_helpers_skin_and_asymmetric_fingers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(root / "fixture.fbx")
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm, meshes = import_fbx(root / "fixture.fbx")
            character, caps = inspect(arm, meshes)
            self.assertEqual(len(meshes), 2)
            self.assertIn("ForearmHelper.L", arm.data.bones)
            self.assertTrue(any(abs(m.matrix_world.translation.y) > .01 for m in meshes))
            self.assertEqual(caps["hands"]["left"]["fingerBones"], 15)
            self.assertEqual(caps["hands"]["left"]["reason"], "UNVALIDATED_FINGER_RIG")
            self.assertEqual(caps["hands"]["right"]["reason"], "NO_FINGER_RIG")
            report = roundtrip(arm, meshes, root / "roundtrip.fbx", character["mapping"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["meshCount"], 2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderLocalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Local Blender regressions failed")
