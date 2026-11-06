import tempfile, unittest
from pathlib import Path
from .dataset import inventory, gitsha, PREFIX

class DatasetTests(unittest.TestCase):

    def test_group_and_hash_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / '러프').mkdir()
            (root / '러프/087.4.png').write_bytes(b'altered')
            tree = {'tree': [{'path': PREFIX + '러프/087.4.png', 'type': 'blob', 'sha': gitsha(b'original')}]}
            r = inventory(tree, root)[0]
            self.assertFalse(r['files']['image']['verified'])
            self.assertFalse(r['files']['gt']['available'])
            self.assertEqual(r['original_cut_id'], '087')
            self.assertEqual(r['split'], 'development')

    def test_unknown_is_not_holdout(self):
        r = inventory({'tree': [{'path': PREFIX + '러프/002.png', 'type': 'blob', 'sha': 'x'}]}, Path('/nonexistent'))[0]
        self.assertEqual(r['split'], 'unassigned')
        self.assertEqual(r['source_seed_status'], 'missing')
if __name__ == '__main__':
    unittest.main()
