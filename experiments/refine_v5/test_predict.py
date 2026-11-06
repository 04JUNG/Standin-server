import unittest
from .predict import predict_rows

class InferenceTests(unittest.TestCase):

    def test_gt_field_rejected_before_model(self):
        with self.assertRaises(ValueError):
            predict_rows(None, [{'case_id': 'x', 'gt': 'secret.fbx'}])

    def test_empty_input_does_not_access_gt(self):
        self.assertEqual(predict_rows(None, []), [])
if __name__ == '__main__':
    unittest.main()
