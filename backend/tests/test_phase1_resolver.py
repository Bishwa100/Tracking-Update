"""
Phase 1 decision-boundary tests — pin the identity-resolution logic so the
multi-angle / duplicate-visitor fixes can't silently regress.

Covers the pure (DB-free) functions in identity_resolver plus the bounded
FaceEmbeddingCache. Run with pytest, or directly:

    venv/Scripts/python.exe tests/test_phase1_resolver.py
"""

import os
import sys
from uuid import uuid4

import numpy as np

# Make the backend package importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services import identity_resolver as ir
from app.ml_models import FaceEmbeddingCache


# ── _best_per_visitor ────────────────────────────────────────────────────────

def test_collapse_keeps_best_per_visitor_and_picks_different_runner_up():
    a, b = uuid4(), uuid4()
    collapsed = ir._best_per_visitor([(a, 0.58), (a, 0.54), (b, 0.50), (b, 0.57)])
    assert collapsed[0] == (a, 0.58)        # best score for A
    assert collapsed[1] == (b, 0.57)        # runner-up is a DIFFERENT visitor


# ── _decide_from_face ────────────────────────────────────────────────────────

def test_same_visitor_top_rows_are_returning_not_ambiguous():
    a = uuid4()
    res = ir._decide_from_face([(a, 0.58), (a, 0.54)])
    assert res.match_source == "face" and res.visitor_id == a
    assert not res.is_ambiguous


def test_two_close_different_visitors_are_ambiguous():
    a, b = uuid4(), uuid4()
    res = ir._decide_from_face([(a, 0.58), (b, 0.555)])
    assert res.is_ambiguous


def test_clear_returning_with_distant_runner_up():
    a, b = uuid4(), uuid4()
    res = ir._decide_from_face([(a, 0.70), (b, 0.40)])
    assert res.match_source == "face" and res.visitor_id == a


def test_grey_zone_is_held_not_new():
    a, b = uuid4(), uuid4()
    res = ir._decide_from_face([(a, 0.50), (b, 0.20)])
    assert res.match_source == "grey_zone"
    assert not res.is_new and res.visitor_id is None


def test_confident_stranger_below_reject_is_new():
    a = uuid4()
    res = ir._decide_from_face([(a, settings.REJECT_SIMILARITY - 0.05)])
    assert res.is_new and res.match_source == "new"


def test_empty_gallery_is_new():
    res = ir._decide_from_face([])
    assert res.is_new and res.match_source == "new"


def test_masked_offset_loosens_returning_threshold():
    a = uuid4()
    # 0.51 is grey-zone at the 0.55 threshold, but RETURNING with a -0.05 offset.
    held = ir._decide_from_face([(a, 0.51)], threshold_offset=0.0)
    matched = ir._decide_from_face([(a, 0.51)], threshold_offset=-0.05)
    assert held.match_source == "grey_zone"
    assert matched.match_source == "face"


# ── FaceEmbeddingCache (bounded) ─────────────────────────────────────────────

def test_cache_lru_eviction_caps_size():
    c = FaceEmbeddingCache(max_entries=3, ttl_seconds=0)
    for i in range(5):
        c.put(i, np.zeros(4, dtype=np.float32))
    assert c.size == 3
    assert c.get(0) is None and c.get(1) is None     # oldest evicted
    assert c.get(4) is not None
    assert c.evictions == 2


def test_cache_lru_is_recency_aware():
    c = FaceEmbeddingCache(max_entries=2, ttl_seconds=0)
    c.put(1, np.zeros(4, dtype=np.float32))
    c.put(2, np.zeros(4, dtype=np.float32))
    c.get(1)                                          # touch 1 -> 2 is now LRU
    c.put(3, np.zeros(4, dtype=np.float32))
    assert c.get(1) is not None and c.get(2) is None


def test_cache_ttl_expiry():
    import time

    c = FaceEmbeddingCache(max_entries=10, ttl_seconds=1)
    c.put(7, np.ones(4, dtype=np.float32))
    assert c.get(7) is not None                        # fresh
    # Monkey-patch the stored insertion time into the past to avoid sleeping.
    emb, _ = c._store[7]
    c._store[7] = (emb, time.monotonic() - 5)
    assert c.get(7) is None                            # expired -> miss


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            passed += 1
    print(f"\n{passed} tests passed")
