import copy
import json
import unittest
from pathlib import Path
from .curation import curate, validate_review, review_mask

class CurationTests(unittest.TestCase):
    def setUp(self):
        p = Path(__file__).resolve().parents[2] / 'evaluation/refine_v5/visual_review_v2.json'
        self.review = json.loads(p.read_text())
        self.inventory = [dict(case_id=r['case_id'], original_cut_id=r['original_cut_id'], files={'image': {'sha256': r['image_sha256']}}) for r in self.review['records']]
    def test_whole_corpus_group_integrity(self):
        result = curate(self.inventory, self.review)
        seen = {}
        for r in result['records']:
            group = r['source_risk_group']
            self.assertEqual(seen.setdefault(group, r['split']), r['split'])
        for split, expected in [('development',60),('validation',20),('test',20)]:
            self.assertEqual(len({r['original_cut_id'] for r in result['records'] if r['split']==split}),expected)
        self.assertFalse(result['artist_disjoint_verified'])
        for row in result['records']:
            if row.get('prior_model_evaluation_sources'): self.assertEqual(row['split'],'development')
    def test_no_implicit_human_authority(self):
        review = copy.deepcopy(self.review); review['records'][0]['human_verified'] = True
        with self.assertRaises(ValueError): validate_review(self.inventory, review)
    def test_hash_and_duplicate_rejection(self):
        review = copy.deepcopy(self.review); review['records'][0]['image_sha256']='wrong'
        with self.assertRaises(ValueError): validate_review(self.inventory, review)
        review = copy.deepcopy(self.review); review['records'].append(review['records'][0])
        with self.assertRaises(ValueError): validate_review(self.inventory, review)
    def test_face_and_unknown_not_match(self):
        for row in self.review['records']:
            if row['scope']=='face' or row['target_identity']=='ambiguous': self.assertFalse(any(review_mask(row)))
if __name__=='__main__': unittest.main()
