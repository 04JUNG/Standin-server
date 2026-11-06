"""Run all v5 experiment boundary tests without a pytest dependency."""
import unittest
from experiments.refine_v5 import test_learning

def main():
    suite=unittest.defaultTestLoader.discover('experiments/refine_v5',pattern='test_*.py',top_level_dir='.')
    for name in dir(test_learning):
        if name.startswith('test_'):suite.addTest(unittest.FunctionTestCase(getattr(test_learning,name)))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
