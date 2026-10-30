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


class BlenderLocalTests(unittest.TestCase):
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
