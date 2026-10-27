import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from standin_eval.final_fbx_evaluator import seal_export_policy
from standin_eval.pipeline_report import build_pipeline_report
from standin_eval.util import hash_json, read_json, sha256_file, write_json, write_jsonl


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.unit = dict(unit_id='unit-A', cut_id='cut', gt_person_id='person',
            selection_revision='s1', body_revision='b1', hand_revision='h1',
            output_scope='full', camera={'azimuth': 0}, view='front')
        self.row = dict(unit_id='unit-A', arm='F25', result_bvh_sha256='a'*64, state='pass')
        write_jsonl(self.root / 'frozen_units.private.jsonl', [self.unit])
        write_json(self.root / 'experiment.frozen.json', {'units_sha256': hash_json([self.unit])})
        write_jsonl(self.root / 'experiment_arms.jsonl', [self.row])
        write_json(self.root / 'experiment_report.json', {'decision': 'INCONCLUSIVE'})
        write_json(self.root / 'search_evidence.json', {})
        write_json(self.root / 'manifest.json', dict(mode='experiment', run_id='fixture',
            frozen_sha256=sha256_file(self.root / 'experiment.frozen.json'),
            result_seal={n:sha256_file(self.root / n) for n in
                ('experiment_arms.jsonl', 'experiment_report.json', 'search_evidence.json')}))
        self.sealed = seal_export_policy([{'task_id':'save-A', 'selection_keys':[{'unit_id':'unit-A','arm':'F25'}]}],
            {'policy_id':'offline', 'required_dimensions':['native_save'], 'minimum_tasks':1})
        self.selection = dict(self.unit, arm='F25', scope='full', source_sha256='a'*64,
            camera_sha256=hash_json({'camera':self.unit['camera'],'view':'front'}))
        self.selection.pop('output_scope')

    def report(self, selection=None):
        return build_pipeline_report(self.root,self.sealed,[{'task_id':'save-A',
            'execution_kind':'fixture','selections':[selection or self.selection]}])

    def test_exact_link_does_not_certify_fixture_product_quality(self):
        r=self.report()
        self.assertEqual(r['lineage_links']['pass'],1)
        self.assertEqual(r['decision'],'INCONCLUSIVE')

    def test_wrong_source_or_revision_fails(self):
        for field, value in [('source_sha256','b'*64),('hand_revision','h0'),('camera_sha256','c'*64)]:
            with self.subTest(field=field):
                r=self.report({**self.selection,field:value})
                self.assertEqual(r['decision'],'FAIL')
                self.assertEqual(r['lineage_links']['fail'],1)

    def test_missing_observation_retains_denominator(self):
        r=build_pipeline_report(self.root,self.sealed,[])
        self.assertEqual(r['lineage_links']['denominator'],1)
        self.assertEqual(r['lineage_links']['unknown'],1)

    def test_tampered_report_or_units_rejected(self):
        write_json(self.root / 'experiment_report.json', {'decision':'PASS'})
        with self.assertRaisesRegex(ValueError,'seal'):
            self.report()


if __name__ == '__main__':
    unittest.main()
