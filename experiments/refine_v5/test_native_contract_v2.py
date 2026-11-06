import unittest
from .native_contract_v2 import compare_native

class NativeContractTests(unittest.TestCase):

    def setUp(self):
        self.heads = {'Root': [0, 0, 0], 'Head_end': [0, 1, 0]}
        self.parents = {'Root': None, 'Head_end': 'Root'}
        self.lengths = {'Root': 1.0, 'Head_end': 0.1}

    def call(self, heads=None, parents=None, lengths=None):
        return compare_native(self.heads, heads if heads is not None else self.heads, self.parents, parents if parents is not None else self.parents, self.lengths, lengths if lengths is not None else self.lengths)

    def test_finite_roundtrip(self):
        self.assertEqual(self.call()['max_head_error_m'], 0.0)

    def test_added_and_missing_terminal_rejected(self):
        for heads in [self.heads | {'Head_end_end': [0, 2, 0]}, {'Root': [0, 0, 0]}]:
            with self.assertRaises(ValueError):
                self.call(heads=heads)

    def test_parent_change_rejected(self):
        with self.assertRaises(ValueError):
            self.call(parents={'Root': None, 'Head_end': None})

    def test_nan_and_length_regression_rejected(self):
        with self.assertRaises(ValueError):
            self.call(heads={'Root': [float('nan'), 0, 0], 'Head_end': [0, 1, 0]})
        with self.assertRaises(ValueError):
            self.call(lengths=self.lengths | {'Head_end': 0.2})
if __name__ == '__main__':
    unittest.main()
