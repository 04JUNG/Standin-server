import tempfile, unittest
from pathlib import Path
import numpy as np
from .model import Corrector

class ModelTests(unittest.TestCase):

    def test_checkpoint_reload_and_wrong_version(self):
        x = np.zeros((2, 589))
        x[:, 0] = 1
        x[1, 1] = 1
        y = np.ones((2, 12))
        m = Corrector.fit(x, y)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.npz'
            m.save(path)
            np.testing.assert_allclose(Corrector.load(path).weights, m.weights)
            np.savez(path, weights=m.weights, version='wrong')
            with self.assertRaises(ValueError):
                Corrector.load(path)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            Corrector.fit([[float('nan')]], [[1.0]])

    def test_ridge_fits_tiny_numeric_batch(self):
        x = np.eye(3)
        y = np.arange(36).reshape(3, 12)
        m = Corrector.fit(x, y)
        np.testing.assert_allclose(x @ m.weights, y, atol=4e-05)
if __name__ == '__main__':
    unittest.main()
