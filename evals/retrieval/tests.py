"""Unit tests for retrieval evaluation metrics.

This module contains unit tests for the metric functions used in retrieval evaluation.
Tests cover recall@k, reciprocal rank, and gold rank calculations with various scenarios
including edge cases like empty results and items not found.

Run with:
    python evals/retrieval/tests.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.evaluate import recall_at_k, reciprocal_rank, gold_rank


# ─────────────────────────────────────────────────────────────
# Tests for recall_at_k
# ─────────────────────────────────────────────────────────────


def test_recall_at_k_found_at_position_1():
    """Test recall@k when gold item is at position 1."""
    assert recall_at_k("v1", ["v1", "v2", "v3"], k=1) == 1
    assert recall_at_k("v1", ["v1", "v2", "v3"], k=3) == 1


def test_recall_at_k_found_within_k():
    """Test recall@k when gold item is within top-k but not first."""
    assert recall_at_k("v1", ["v2", "v1", "v3"], k=3) == 1
    assert recall_at_k("v1", ["v2", "v3", "v1"], k=3) == 1
    assert recall_at_k("v1", ["v2", "v1"], k=2) == 1


def test_recall_at_k_found_outside_k():
    """Test recall@k when gold item exists but is outside top-k."""
    assert recall_at_k("v1", ["v2", "v1", "v3"], k=1) == 0
    assert recall_at_k("v1", ["v2", "v3", "v1"], k=2) == 0
    assert recall_at_k("v1", ["v2", "v3", "v4", "v1"], k=3) == 0


def test_recall_at_k_not_found():
    """Test recall@k when gold item is not in results at all."""
    assert recall_at_k("v1", ["v2", "v3"], k=3) == 0
    assert recall_at_k("v1", ["v2", "v3", "v4"], k=10) == 0


def test_recall_at_k_empty_results():
    """Test recall@k with empty result list."""
    assert recall_at_k("v1", [], k=1) == 0
    assert recall_at_k("v1", [], k=10) == 0


def test_recall_at_k_k_larger_than_results():
    """Test recall@k when k is larger than the result list."""
    assert recall_at_k("v1", ["v1"], k=10) == 1
    assert recall_at_k("v1", ["v2", "v1"], k=100) == 1
    assert recall_at_k("v1", ["v2"], k=100) == 0


# ─────────────────────────────────────────────────────────────
# Tests for reciprocal_rank
# ─────────────────────────────────────────────────────────────


def test_reciprocal_rank_at_position_1():
    """Test reciprocal rank when gold item is at position 1 (RR=1.0)."""
    assert reciprocal_rank("v1", ["v1", "v2", "v3"]) == 1.0
    assert reciprocal_rank("v1", ["v1"]) == 1.0


def test_reciprocal_rank_at_position_2():
    """Test reciprocal rank when gold item is at position 2 (RR=0.5)."""
    assert reciprocal_rank("v1", ["v2", "v1", "v3"]) == 0.5
    assert reciprocal_rank("v1", ["v2", "v1"]) == 0.5


def test_reciprocal_rank_at_position_3():
    """Test reciprocal rank when gold item is at position 3 (RR≈0.333)."""
    result = reciprocal_rank("v1", ["v2", "v3", "v1"])
    assert abs(result - 0.333) < 0.01


def test_reciprocal_rank_at_various_positions():
    """Test reciprocal rank at various positions."""
    assert reciprocal_rank("v1", ["v2", "v3", "v4", "v1"]) == 0.25  # Position 4: 1/4
    assert reciprocal_rank("v1", ["v2", "v3", "v4", "v5", "v1"]) == 0.2  # Position 5: 1/5
    assert reciprocal_rank("v1", ["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v1"]) == 0.1  # Position 10: 1/10


def test_reciprocal_rank_not_found():
    """Test reciprocal rank when gold item is not found (RR=0.0)."""
    assert reciprocal_rank("v1", ["v2", "v3"]) == 0.0
    assert reciprocal_rank("v1", ["v2", "v3", "v4", "v5"]) == 0.0


def test_reciprocal_rank_empty_results():
    """Test reciprocal rank with empty result list (RR=0.0)."""
    assert reciprocal_rank("v1", []) == 0.0


# ─────────────────────────────────────────────────────────────
# Tests for gold_rank
# ─────────────────────────────────────────────────────────────


def test_gold_rank_at_position_1():
    """Test gold_rank when item is at position 1."""
    assert gold_rank("v1", ["v1", "v2", "v3"]) == 1
    assert gold_rank("v1", ["v1"]) == 1


def test_gold_rank_at_position_2():
    """Test gold_rank when item is at position 2."""
    assert gold_rank("v1", ["v2", "v1", "v3"]) == 2
    assert gold_rank("v1", ["v2", "v1"]) == 2


def test_gold_rank_at_position_3():
    """Test gold_rank when item is at position 3."""
    assert gold_rank("v1", ["v2", "v3", "v1"]) == 3
    assert gold_rank("v1", ["v2", "v3", "v1", "v4"]) == 3


def test_gold_rank_at_various_positions():
    """Test gold_rank at various positions."""
    assert gold_rank("v1", ["v2", "v3", "v4", "v1"]) == 4
    assert gold_rank("v1", ["v2", "v3", "v4", "v5", "v1"]) == 5
    assert gold_rank("v1", ["v2"] * 9 + ["v1"]) == 10  # Position 10


def test_gold_rank_not_found():
    """Test gold_rank when item is not found (returns -1)."""
    assert gold_rank("v1", ["v2", "v3"]) == -1
    assert gold_rank("v1", ["v2", "v3", "v4", "v5"]) == -1


def test_gold_rank_empty_results():
    """Test gold_rank with empty result list (returns -1)."""
    assert gold_rank("v1", []) == -1


# ─────────────────────────────────────────────────────────────
# Edge case and integration tests
# ─────────────────────────────────────────────────────────────


def test_all_metrics_with_same_input():
    """Test that all metrics work correctly on the same input."""
    retrieved_ids = ["v2", "v1", "v3", "v4"]
    gold_id = "v1"

    # Gold is at position 2
    assert gold_rank(gold_id, retrieved_ids) == 2
    assert reciprocal_rank(gold_id, retrieved_ids) == 0.5
    assert recall_at_k(gold_id, retrieved_ids, k=1) == 0  # Not in top-1
    assert recall_at_k(gold_id, retrieved_ids, k=2) == 1  # In top-2
    assert recall_at_k(gold_id, retrieved_ids, k=3) == 1  # In top-3


def test_all_metrics_when_not_found():
    """Test that all metrics handle 'not found' consistently."""
    retrieved_ids = ["v2", "v3", "v4"]
    gold_id = "v1"

    assert gold_rank(gold_id, retrieved_ids) == -1
    assert reciprocal_rank(gold_id, retrieved_ids) == 0.0
    assert recall_at_k(gold_id, retrieved_ids, k=1) == 0
    assert recall_at_k(gold_id, retrieved_ids, k=10) == 0


def test_all_metrics_when_found_first():
    """Test that all metrics handle first position correctly."""
    retrieved_ids = ["v1", "v2", "v3"]
    gold_id = "v1"

    assert gold_rank(gold_id, retrieved_ids) == 1
    assert reciprocal_rank(gold_id, retrieved_ids) == 1.0
    assert recall_at_k(gold_id, retrieved_ids, k=1) == 1
    assert recall_at_k(gold_id, retrieved_ids, k=3) == 1


def test_metrics_with_duplicate_ids():
    """Test metrics when retrieved_ids contains duplicates (first occurrence should count)."""
    # In practice, duplicates shouldn't happen, but test behavior anyway
    retrieved_ids = ["v2", "v1", "v3", "v1"]  # v1 appears twice
    gold_id = "v1"

    # Should use first occurrence (position 2)
    assert gold_rank(gold_id, retrieved_ids) == 2
    assert reciprocal_rank(gold_id, retrieved_ids) == 0.5


def test_metrics_with_special_characters():
    """Test metrics with video IDs containing special characters."""
    retrieved_ids = ["vid-123", "vid_456", "vid.789"]

    assert gold_rank("vid-123", retrieved_ids) == 1
    assert gold_rank("vid_456", retrieved_ids) == 2
    assert gold_rank("vid.789", retrieved_ids) == 3
    assert gold_rank("vid/000", retrieved_ids) == -1


def test_metrics_with_long_result_lists():
    """Test metrics with long result lists (100+ items)."""
    retrieved_ids = [f"v{i}" for i in range(1, 101)]  # v1 through v100

    # Test first position
    assert gold_rank("v1", retrieved_ids) == 1
    assert reciprocal_rank("v1", retrieved_ids) == 1.0

    # Test middle position
    assert gold_rank("v50", retrieved_ids) == 50
    assert reciprocal_rank("v50", retrieved_ids) == 0.02

    # Test last position
    assert gold_rank("v100", retrieved_ids) == 100
    assert reciprocal_rank("v100", retrieved_ids) == 0.01

    # Test not found
    assert gold_rank("v101", retrieved_ids) == -1
    assert reciprocal_rank("v101", retrieved_ids) == 0.0


# ─────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Collect all test functions
    test_functions = [
        test_recall_at_k_found_at_position_1,
        test_recall_at_k_found_within_k,
        test_recall_at_k_found_outside_k,
        test_recall_at_k_not_found,
        test_recall_at_k_empty_results,
        test_recall_at_k_k_larger_than_results,
        test_reciprocal_rank_at_position_1,
        test_reciprocal_rank_at_position_2,
        test_reciprocal_rank_at_position_3,
        test_reciprocal_rank_at_various_positions,
        test_reciprocal_rank_not_found,
        test_reciprocal_rank_empty_results,
        test_gold_rank_at_position_1,
        test_gold_rank_at_position_2,
        test_gold_rank_at_position_3,
        test_gold_rank_at_various_positions,
        test_gold_rank_not_found,
        test_gold_rank_empty_results,
        test_all_metrics_with_same_input,
        test_all_metrics_when_not_found,
        test_all_metrics_when_found_first,
        test_metrics_with_duplicate_ids,
        test_metrics_with_special_characters,
        test_metrics_with_long_result_lists,
    ]

    print(f"Running {len(test_functions)} tests...\n")

    failed = 0
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1

    passed = len(test_functions) - failed
    print(f"\n{passed}/{len(test_functions)} tests passed")

    if failed > 0:
        print(f"{failed} test(s) failed")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
