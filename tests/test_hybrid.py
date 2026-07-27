from __future__ import annotations

from guji.retrieve.hybrid import rrf_fuse


def test_rrf_rewards_agreement():
    # 'b' is top-3 in both rankings; should win over items ranked high in only one
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = rrf_fuse([dense, sparse], k=10, rrf_k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_scores_use_reciprocal_rank():
    fused = dict(rrf_fuse([["x", "y"]], k=10, rrf_k=60))
    assert fused["x"] > fused["y"]
    assert abs(fused["x"] - 1 / 60) < 1e-9
    assert abs(fused["y"] - 1 / 61) < 1e-9


def test_rrf_truncates_to_k():
    dense = [f"d{i}" for i in range(20)]
    assert len(rrf_fuse([dense], k=5, rrf_k=60)) == 5
