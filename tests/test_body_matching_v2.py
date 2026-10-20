"""Coverage regressions; no live API required."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_body_matching import person_payload
from src.experimental.body_matching.schema import parse_person


def observed():
    p=person_payload('p')
    p.update(full_body_visible=True,coverage=dict(head='visible',torso='visible',arms='visible',legs='visible'))
    return p


class CoverageTests(unittest.TestCase):
    def test_cropped_person_cannot_supply_head_count(self):
        p=observed();p['full_body_visible']=False;p['attributes']['head_ratio_class']['value']='h3'
        q=parse_person(p,'p');self.assertIsNone(q['attributes']['head_ratio_class']['value'])
        self.assertIsNotNone(q['attributes']['frame_width']['value'])
        self.assertEqual(parse_person(q,'p'),q)

    def test_covered_torso_keeps_exposed_arm_muscularity(self):
        p=observed();p['clothing_occlusion']=True;p['coverage']['torso']='unknown'
        q=parse_person(p,'p')['attributes']
        self.assertIsNotNone(q['muscularity_visual']['value'])
        for k in ['frame_width','soft_volume','volume_distribution']:self.assertIsNone(q[k]['value'])

    def test_foreshortening_and_missing_feet_exclude_head_count(self):
        for change in ['foreshortening','coverage']:
            p=observed()
            if change=='foreshortening':p[change]=True
            else:p['coverage']['legs']='unknown'
            self.assertIsNone(parse_person(p,'p')['attributes']['head_ratio_class']['value'])

    def test_ownership_unknown_masks_everything(self):
        p=observed();p['ownership_ambiguous']=True
        self.assertTrue(all(x['value'] is None for x in parse_person(p,'p')['attributes'].values()))

    def test_invalid_or_partial_coverage_rejected(self):
        for change in ['partial','invalid','flag']:
            p=observed()
            if change=='partial':del p['coverage']['arms']
            elif change=='invalid':p['coverage']['arms']='clear'
            else:p['full_body_visible']='yes'
            with self.assertRaises(ValueError):parse_person(p,'p')



if __name__ == "__main__":
    unittest.main()
