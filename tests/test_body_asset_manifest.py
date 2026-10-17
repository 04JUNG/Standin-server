"""Draft inventory tests use explicit fake Blender/FBX bytes and literal controllers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimental.body_matching.asset_manifest import (
    build_manifest, canonical, consumer_catalogs, sha256, validate_manifest, write_draft_bundle)
from src.experimental.body_matching.catalog import load_catalog


class BodyAssetManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.delivery_path = self.root / 'DELIVERY.json'
        self.index_path = self.root / 'delivery/FBX_INDEX.json'
        self.index_path.parent.mkdir()
        self.delivery = dict(models={}, named_fbx_index='delivery/FBX_INDEX.json',
                             production_promoted=False, s3_uploaded=False)
        self.index = []
        for family in ('male', 'female'):
            folder = self.root / family
            folder.mkdir()
            model = folder / 'model.blend'
            model.write_bytes(b'FAKE BLENDER FIXTURE ' + family.encode())
            controller = folder / 'controller.py'
            controller.write_text("raise RuntimeError('controller must never execute')\n"
                "VERSION='standin-body-1.2.3'\nRIG_VERSION='fixture-rig'\n"
                "PRESETS={'BASE':('base',{}),'TOON64':('6.4 heads',{'heads':6.4})}\n")
            self.delivery['models'][family] = dict(version='1.2.3', model=str(model),
                model_sha256=sha256(model), controller=str(controller), controller_sha256=sha256(controller), presets=2)
            for preset in ('BASE', 'TOON64'):
                fbx = folder / (preset + '.fbx')
                fbx.write_bytes(b'FAKE FBX FIXTURE ' + family.encode() + preset.encode())
                self.index.append(dict(sex=family, preset=preset, pose='T', fbx=fbx.relative_to(self.root).as_posix(), sha256=sha256(fbx)))
                (folder / (preset + '--T.json')).write_text(json.dumps(dict(
                    sex=family, preset=preset, pose='T', **{'pass': True},
                    input_hashes={str(model):sha256(model), str(controller):sha256(controller)})))
        self.save_inputs()

    def save_inputs(self):
        self.delivery_path.write_text(json.dumps(self.delivery))
        self.index_path.write_text(json.dumps(self.index))

    def build(self):
        return build_manifest(self.delivery_path, self.root)

    def test_deterministic_complete_draft_without_controller_execution(self):
        manifest = self.build()
        self.assertEqual(canonical(manifest), canonical(self.build()))
        self.assertEqual(len(manifest['assets']), 4)
        self.assertEqual(len({a['body_id'] for a in manifest['assets']}), 4)
        for asset in manifest['assets']:
            self.assertEqual(asset['availability'], 'draft')
            self.assertIsNone(asset['measurements'])
            self.assertEqual(asset['supported_pose_ids'], [])
            self.assertIsNone(asset['authoring']['resolved_parameters'])
            self.assertEqual(asset['metadata']['rendering_style'], 'stylized')
            self.assertNotEqual(asset['metadata']['presentation_style'], 'stylized')
        toon = next(a for a in manifest['assets'] if a['preset']=='TOON64')
        self.assertEqual(toon['authoring']['overrides']['heads'], 6.4)
        self.assertIsNone(toon['metadata']['age_design'])

    def test_consumer_catalogs_share_fingerprint_but_do_not_advertise_pending_assets(self):
        outputs = consumer_catalogs(self.build(), self.root)
        catalogs = [v for k,v in outputs.items() if k!='manifest.json']
        self.assertEqual(len({c['source_manifest_sha256'] for c in catalogs}), 1)
        self.assertEqual(outputs['body-models.json']['assets'], [])
        self.assertEqual(outputs['characters.json']['characters'], {})
        self.assertEqual(len(outputs['body-catalog.json']['assets']), 4)
        output = self.root / 'bundle'
        write_draft_bundle(self.build(), output, self.root)
        catalog = load_catalog(output/'body-catalog.json')
        self.assertEqual(catalog.eligible(['some-real-pose']), [])
        self.assertEqual(catalog.issues, ())

    def test_all_references_are_relative_to_repository(self):
        manifest = self.build()
        refs = [manifest['delivery'], manifest['neutral_fbx_index']]
        refs += [s[k] for s in manifest['sources'].values() for k in ('model','controller')]
        refs += [a[k] for a in manifest['assets'] for k in ('neutral_fbx','authoring_qa')]
        self.assertTrue(all(not Path(r['path']).is_absolute() for r in refs))
        self.assertNotIn(str(self.root), canonical(manifest).decode())

    def test_changed_source_fbx_or_qa_is_rejected(self):
        manifest = self.build()
        for ref in (manifest['sources']['male']['controller'], manifest['sources']['female']['model'],
                    manifest['assets'][0]['neutral_fbx'], manifest['assets'][0]['authoring_qa']):
            path = self.root/ref['path']
            before = path.read_bytes()
            path.write_bytes(before+b'changed')
            with self.assertRaises(ValueError): validate_manifest(manifest, self.root)
            path.write_bytes(before)

    def test_duplicate_or_missing_neutral_fbx_rejected(self):
        self.index.append(copy.deepcopy(self.index[0]))
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'duplicate_fbx_identity'): self.build()
        self.index = self.index[1:-1]
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'neutral_fbx_missing'): self.build()

    def test_fbx_path_escape_rejected_even_for_unused_posed_fixture(self):
        self.index.append(dict(sex='male',preset='BASE',pose='DOWN85',fbx='../outside.fbx',sha256='a'*64))
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'outside_repository'): self.build()

    def test_wrong_model_version_and_preset_count_rejected(self):
        for key,value in (('version','9.9.9'),('presets',3)):
            before = self.delivery['models']['male'][key]
            self.delivery['models']['male'][key] = value
            self.save_inputs()
            with self.assertRaises(ValueError): self.build()
            self.delivery['models']['male'][key] = before
        self.save_inputs()

    def test_qa_from_another_preset_or_source_rejected(self):
        qa_path = self.root/'male/BASE--T.json'
        raw = json.loads(qa_path.read_text())
        for update in ({'preset':'TOON64'}, {'input_hashes':{}}, {'pass':False}):
            qa_path.write_text(json.dumps({**raw, **update}))
            with self.assertRaisesRegex(ValueError,'neutral_qa_source_mismatch'): self.build()
        qa_path.write_text(json.dumps(raw))

    def test_pending_inventory_cannot_be_edited_into_approved_catalog(self):
        manifest = self.build()
        for key,value in (('availability','eligible'),('supported_pose_ids',['pose']),('measurements',{})):
            changed = copy.deepcopy(manifest)
            changed['assets'][0][key] = value
            with self.assertRaisesRegex(ValueError,'cannot_grant_product_eligibility'):
                consumer_catalogs(changed,self.root)

    def test_valid_hash_from_another_preset_cannot_be_substituted(self):
        manifest = self.build()
        manifest['assets'][0]['neutral_fbx'] = manifest['assets'][1]['neutral_fbx']
        with self.assertRaisesRegex(ValueError,'neutral_preset_mismatch'): validate_manifest(manifest,self.root)

    def test_missing_manifest_preset_and_invented_overrides_rejected(self):
        manifest = self.build()
        changed = copy.deepcopy(manifest)
        changed['assets'].pop()
        with self.assertRaisesRegex(ValueError,'preset_coverage_mismatch'): validate_manifest(changed,self.root)
        changed = copy.deepcopy(manifest)
        changed['assets'][0]['authoring']['overrides']['heads'] = 8
        with self.assertRaisesRegex(ValueError,'source_definition_mismatch'): validate_manifest(changed,self.root)

    def test_bundle_is_idempotent_and_does_not_replace_other_output(self):
        output = self.root/'bundle'
        write_draft_bundle(self.build(), output, self.root)
        first = {p.name:p.read_bytes() for p in output.iterdir()}
        write_draft_bundle(self.build(), output, self.root)
        self.assertEqual(first,{p.name:p.read_bytes() for p in output.iterdir()})
        (output/'unrelated.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'output_exists'): write_draft_bundle(self.build(),output,self.root)
        self.assertEqual((output/'unrelated.txt').read_text(),'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
