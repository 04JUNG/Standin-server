"""CSP final-name compatibility must never truncate skin or rewrite other data."""
import struct
import unittest

from converter.local.contract import LocalError, VERSIONS, snapshot, digest
from converter.local.csp_compat import MAGIC, patch_bytes, plan_aliases


def prop(name):
    value = name + b'\x00\x01Model'
    return b'S' + struct.pack('<I', len(value)) + value


class CspCompatibilityTests(unittest.TestCase):
    def test_exact_single_property_only_and_reversible(self):
        models = [(7, b'root_group_\x00\x01Model', b'LimbNode')]
        plan = plan_aliases(models)
        before = MAGIC + b'\xe8\x1c\0\0' + b'untouched mesh/weights/matrices' + prop(b'root_group_') + b'footer'
        after = patch_bytes(before, plan)
        self.assertEqual(len(before), len(after))
        self.assertEqual(after.replace(prop(b'StandinBase'), prop(b'root_group_')), before)
        self.assertEqual(plan, [{'objectId':7,'original':'root_group_','exported':'StandinBase'}])

    def test_female_and_other_names_are_byte_identical(self):
        for models in ([], [(1,b'root_group_\x00\x01Model',b'Mesh')],
                       [(1,b'mixamorig:Hips\x00\x01Model',b'LimbNode')]):
            with self.subTest(models=models):
                plan = plan_aliases(models)
                before = MAGIC+b'\xe8\x1c\0\0'+b'no edit'
                self.assertEqual(plan, [])
                self.assertEqual(patch_bytes(before, plan), before)

    def test_alias_collision_uses_unique_deterministic_equal_length_name(self):
        models = [(1,b'root_group_\x00\x01Model',b'LimbNode'),
                  (2,b'StandinBase\x00\x01Model',b'Mesh'),
                  (3,b'Standin0000\x00\x01Model',b'LimbNode')]
        self.assertEqual(plan_aliases(models)[0]['exported'], 'Standin0001')
        self.assertEqual(plan_aliases(models), plan_aliases(models))

    def test_ambiguous_and_tampered_data_fail_closed(self):
        models = [(1,b'root_group_\x00\x01Model',b'LimbNode')]
        with self.assertRaises(LocalError): plan_aliases(models+models)
        for data in (b'not FBX', MAGIC+b'\0'*20, MAGIC+prop(b'root_group_')*2):
            with self.subTest(data=data), self.assertRaises(LocalError):
                patch_bytes(data,plan_aliases(models))
        with self.assertRaises(LocalError):
            patch_bytes(MAGIC+prop(b'root_group_'),[{'original':'root_group_','exported':'short'}])

    def test_v10_registered_model_is_rejected_by_v11_host(self):
        import json
        import tempfile
        from pathlib import Path
        from converter.local.contract import versions, sha256
        from converter.local.host import Host
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = Host(root, root/"runtime", lambda _: None)
            try:
                folder = root/"models/old"
                folder.mkdir()
                fbx = folder/"character.fbx"
                fbx.write_bytes(b"old normalized file")
                manifest = {"versions": {**versions(), "converterVersion": "personal-converter/10"},
                            "normalizedFormat": "fbx", "normalizedHash": sha256(fbx)}
                (folder/"manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(LocalError) as error:
                    host.load_model("old")
                self.assertEqual(error.exception.code, "RUNTIME_MISMATCH")
                self.assertEqual(fbx.read_bytes(), b"old normalized file")
            finally:
                host.close()

    def test_v10_snapshots_cannot_reuse_v11_crashing_artifacts(self):
        old={'localModelId':'model','sourceHash':'a'*64,'versions':{**VERSIONS,'converterVersion':'personal-converter/10'}}
        new={**old,'versions':VERSIONS}
        self.assertNotEqual(digest(snapshot(old,'b'*64,{'left':None,'right':None})),
                            digest(snapshot(new,'b'*64,{'left':None,'right':None})))
        self.assertEqual(VERSIONS['converterVersion'],'personal-converter/11')
