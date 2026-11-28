import subprocess
import sys
import unittest
from pathlib import Path
from src.experimental.cached_limb_inventory import match_cached_people


class InventoryTests(unittest.TestCase):
    def case(self):
        return {'expected_person_ids':['cut-P1','cut-P2'],'source_person_boxes':[
            {'reference_person_id':'cut-P1','box':[0,0,10,20]},
            {'reference_person_id':'cut-P2','box':[30,0,40,20]}]}

    def test_permuted_cache_indices_follow_boxes(self):
        mapped=match_cached_people(self.case(),[{'index':0,'box':[30,0,40,20]},{'index':1,'box':[0,0,10,20]}])
        self.assertEqual(mapped['cut-P1']['cache_person_index'],1)
        self.assertEqual(mapped['cut-P2']['cache_person_index'],0)

    def test_missing_person_not_replaced_with_first_cache_person(self):
        mapped=match_cached_people(self.case(),[{'box':[0,0,10,20]}])
        self.assertEqual(mapped['cut-P1']['status'],'matched')
        self.assertEqual(mapped['cut-P2']['status'],'held')

    def test_duplicate_boxes_abstain(self):
        mapped=match_cached_people(self.case(),[{'box':[0,0,10,20]},{'box':[0,0,10,20]}])
        self.assertEqual(mapped['cut-P1']['status'],'held')

    def test_same_cached_person_cannot_fill_two_frozen_people(self):
        case=self.case();case['source_person_boxes'][1]['box']=[0,0,10,20]
        mapped=match_cached_people(case,[{'box':[0,0,10,20]}])
        self.assertTrue(all(r['status']=='held' for r in mapped.values()))

    def test_single_person_crop_change_does_not_invent_second_identity(self):
        case={'expected_person_ids':['one-P1'],'source_person_boxes':[{'reference_person_id':'one-P1','box':[0,0,10,20]}]}
        self.assertEqual(match_cached_people(case,[{'box':[0,0,9,18]}])['one-P1']['status'],'matched')

    def test_default_native_execution_is_disabled_before_blender_or_asset_import(self):
        repo=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(repo/'scripts/run_cached_limb_pilot_native.py'),
            str(repo),'/missing-workspace','/missing-run'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('experimental diagnosis is disabled',result.stderr)
        self.assertNotIn('bpy',result.stderr)


if __name__=='__main__':unittest.main()
