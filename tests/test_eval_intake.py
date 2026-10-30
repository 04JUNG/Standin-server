"""Offline preparation contracts; no authored source asset is committed."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.inventory import prepare_inventory, write_csv, INTAKE_FIELDS, PERSON_FIELDS
from standin_eval.intake import import_intake
from standin_eval.dataset import load_dataset, validate_dataset
from standin_eval.util import read_jsonl


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        from PIL import Image
        Image.new('RGB', (20, 30)).save(self.images / '=source.png')
        (self.images / 'duplicate.png').write_bytes((self.images / '=source.png').read_bytes())
        self.output = self.root / 'inventory'
        self.summary = prepare_inventory(self.images, self.output)

    def complete(self):
        cut = read_jsonl(self.output / 'inventory.jsonl')[0]['cut_id']
        write_csv(self.output / 'completed-intake.csv', INTAKE_FIELDS, [{
            '컷ID': cut, '작가그룹ID': 'artist-1', '작품ID': 'work-1',
            '원본페이지장면그룹ID': 'scene-1', '평가세트': 'D1', '대표셋_어려운셋': '대표',
            '인물수': '1', '평가사용허용': '허용', '팀내공유허용': '불허',
            '외부VLM전송허용': '확인 필요', 'expected_route': 'core'}])
        person = {'컷ID': cut, 'person_id': 'permanent-A', 'bbox_xyxy': '[1,2,19,29]',
                  'bbox_source': 'human-review', 'bbox_confirmed': '예',
                  '3D보조대상': '예', 'out_of_scope': '아니오', 'scale_class': 'near',
                  '가린부위': '판단 불가', '작가가아는숨은자세': ''}
        write_csv(self.output / 'completed-person.csv', PERSON_FIELDS, [person])
        return person

    def run_import(self, completed=True):
        return import_intake(self.output,
            self.output / ('completed-intake.csv' if completed else 'rough-intake.csv'),
            self.output / ('completed-person.csv' if completed else 'person-intent.csv'),
            self.root / 'dataset', dataset_id='offline-v1')

    def test_duplicate_inventory_preserves_unknowns_and_source_bytes(self):
        self.assertEqual((self.summary['files'], self.summary['unique_image_contents'], self.summary['duplicate_files']), (2, 1, 1))
        rows = read_jsonl(self.output / 'inventory.jsonl')
        self.assertIsNone(rows[0]['person_count'])
        self.assertIsNone(rows[0]['scene_group_id'])
        self.assertEqual(rows[0]['external_vlm_permission'], 'unknown')
        self.assertEqual(self.summary['independent_groups'], None)
        with (self.output / 'rough-intake.csv').open(encoding='utf-8-sig') as f:
            self.assertTrue(next(csv.DictReader(f))['파일명'].startswith("'="))
        self.assertEqual((self.images / '=source.png').read_bytes(), (self.images / 'duplicate.png').read_bytes())
        with self.assertRaises(FileExistsError):
            prepare_inventory(self.images, self.output)

    def test_blank_human_rows_do_not_become_dataset(self):
        result = self.run_import(completed=False)
        self.assertFalse(result['exported'])
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertFalse((self.root / 'dataset').exists())

    def test_completed_labels_reuse_legacy_seal_and_keep_consent_unknown(self):
        self.complete()
        self.assertTrue(self.run_import()['exported'])
        ds = load_dataset(self.root / 'dataset')
        self.assertEqual(len(ds.target_persons), 1)
        self.assertEqual(validate_dataset(ds), [])
        self.assertEqual(ds.cuts[0]['external_vlm_permission'], '확인 필요')
        self.assertEqual(ds.persons[0]['person_id'], 'permanent-A')
        self.assertEqual(ds.persons[0]['human_intent']['가린부위'], '판단 불가')

    def test_changed_inventory_is_rejected(self):
        path = self.output / 'inventory.jsonl'
        path.write_text(path.read_text().replace('unknown', 'allow'))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.run_import(completed=False)

    def test_changed_image_is_rejected(self):
        self.complete()
        (self.images / '=source.png').write_bytes(b'changed')
        self.assertFalse(self.run_import()['exported'])

    def test_output_cannot_write_into_source(self):
        with self.assertRaisesRegex(ValueError, 'read-only'):
            prepare_inventory(self.images, self.images / 'output')

    def test_nonfinite_bbox_and_missing_cut_do_not_pass(self):
        row = self.complete()
        row['bbox_xyxy'] = '[NaN,2,19,29]'
        write_csv(self.output / 'completed-person.csv', PERSON_FIELDS, [row])
        self.assertFalse(self.run_import()['exported'])


if __name__ == '__main__':
    unittest.main()
