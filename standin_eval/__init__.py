"""Standin evaluation harness.

The package is intentionally independent from the production API contract.  It
stores every run as versioned JSON/JSONL artifacts under ``out/eval``.
"""

from .dataset import EvalDataset, load_dataset

__all__ = ["EvalDataset", "load_dataset"]
__version__ = "0.1.0"
