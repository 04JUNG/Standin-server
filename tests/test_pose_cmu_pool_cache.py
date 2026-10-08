"""Old frame counts and changed capture sources must not be silently reused."""

from copy import deepcopy

import pytest

from pose_curation.expansion.cmu_pool import _validate_cached
from pose_curation.storage import sha256


def test_pool_cache_binds_sampling_source_and_output(tmp_path):
    bvh = tmp_path / "pose.bvh"
    bvh.write_text("reviewable pose", encoding="utf-8")
    provenance = {"capture.amc": {"sha256": "original", "url": "official"}}
    sampling = {"count": 14, "implementation_sha256": {"adapter": "v1"}}
    cached = {
        "config_sha256": "config",
        "source_provenance": provenance,
        "sampling": sampling,
        "poses": [{"bvh": str(bvh), "bvh_sha256": sha256(bvh)}],
    }
    _validate_cached(cached, "config", provenance, sampling)
    for field, replacement in (
        ("config_sha256", "changed"),
        ("source_provenance", {}),
        ("sampling", {**sampling, "count": 18}),
        ("sampling", {**sampling, "implementation_sha256": {"adapter": "v2"}}),
    ):
        changed = deepcopy(cached)
        changed[field] = replacement
        with pytest.raises(ValueError, match="fresh output"):
            _validate_cached(changed, "config", provenance, sampling)
    bvh.write_text("changed output", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output"):
        _validate_cached(cached, "config", provenance, sampling)
