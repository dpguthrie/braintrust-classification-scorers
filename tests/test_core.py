"""Unit tests for the core scoring logic (no Braintrust dependency).

Validates precision_recall_f1 and aggregate_scores across many edge cases
including single values, empty inputs, None, duplicates, mixed types, etc.

Also validates *aggregate* scores across full datasets, which is what users
actually see in the Braintrust UI.
"""

import pytest
from braintrust_classification_scorers._core import (
    _to_set,
    aggregate_scores,
    precision_recall_f1,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def s(output, expected):
    """Shorthand -- returns {"precision": ..., "recall": ..., "f1": ...}."""
    r = precision_recall_f1(output, expected)
    return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}


def agg(dataset):
    """Shorthand that returns only the three aggregate scores."""
    r = aggregate_scores(dataset)
    return {"precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}


# ═════════════════════════════════════════════════════════════════════════
#  _to_set normalization
# ═════════════════════════════════════════════════════════════════════════

class TestToSet:
    def test_list(self):
        assert _to_set(["a", "b"]) == {"a", "b"}

    def test_single_string(self):
        assert _to_set("paris") == {"paris"}

    def test_empty_list(self):
        assert _to_set([]) == set()

    def test_none(self):
        assert _to_set(None) == set()

    def test_set_passthrough(self):
        assert _to_set({"x", "y"}) == {"x", "y"}

    def test_tuple(self):
        assert _to_set(("a", "b")) == {"a", "b"}

    def test_single_int(self):
        assert _to_set(42) == {42}

    def test_single_bool(self):
        assert _to_set(True) == {True}

    def test_empty_string(self):
        """An empty string is still a value, not 'nothing'."""
        assert _to_set("") == {""}

    def test_empty_tuple(self):
        assert _to_set(()) == set()


# ═════════════════════════════════════════════════════════════════════════
#  Perfect matches
# ═════════════════════════════════════════════════════════════════════════

class TestPerfectMatch:
    def test_lists_identical(self):
        assert s(["a", "b", "c"], ["a", "b", "c"]) == {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }

    def test_single_string_both(self):
        assert s("paris", "paris") == {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }

    def test_single_element_lists(self):
        assert s(["paris"], ["paris"]) == {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }

    def test_order_irrelevant(self):
        assert s(["c", "a", "b"], ["b", "c", "a"]) == {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
        }

    def test_single_int(self):
        assert s(42, 42) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


# ═════════════════════════════════════════════════════════════════════════
#  No overlap
# ═════════════════════════════════════════════════════════════════════════

class TestNoOverlap:
    def test_completely_different_lists(self):
        r = s(["x", "y"], ["a", "b"])
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_single_string_mismatch(self):
        r = s("paris", "london")
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_disjoint_large(self):
        r = s(["a", "b", "c", "d"], ["w", "x", "y", "z"])
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ═════════════════════════════════════════════════════════════════════════
#  Empty / None edge cases
# ═════════════════════════════════════════════════════════════════════════

class TestEmptyAndNone:
    def test_empty_output_nonempty_expected(self):
        """No predictions -> precision undefined, recall = 0."""
        r = s([], ["a", "b"])
        assert r["precision"] is None
        assert r["recall"] == 0.0
        assert r["f1"] is None

    def test_nonempty_output_empty_expected(self):
        """Nothing expected -> all FP, precision = 0, recall undefined."""
        r = s(["a", "b"], [])
        assert r["precision"] == 0.0
        assert r["recall"] is None
        assert r["f1"] is None

    def test_both_empty(self):
        """Nothing predicted, nothing expected -> all undefined."""
        r = s([], [])
        assert r["precision"] is None
        assert r["recall"] is None
        assert r["f1"] is None

    def test_none_output(self):
        r = s(None, ["a"])
        assert r["precision"] is None
        assert r["recall"] == 0.0
        assert r["f1"] is None

    def test_none_expected(self):
        r = s(["a"], None)
        assert r["precision"] == 0.0
        assert r["recall"] is None
        assert r["f1"] is None

    def test_both_none(self):
        r = s(None, None)
        assert r["precision"] is None
        assert r["recall"] is None
        assert r["f1"] is None

    def test_single_string_vs_empty(self):
        r = s("paris", [])
        assert r["precision"] == 0.0
        assert r["recall"] is None

    def test_empty_vs_single_string(self):
        r = s([], "paris")
        assert r["precision"] is None
        assert r["recall"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
#  Partial overlap
# ═════════════════════════════════════════════════════════════════════════

class TestPartialOverlap:
    def test_output_strict_subset(self):
        """Predicted 1 of 3 correctly -> P=1.0, R=1/3, F1=0.5."""
        r = s(["a"], ["a", "b", "c"])
        assert r["precision"] == 1.0
        assert r["recall"] == pytest.approx(1 / 3)
        assert r["f1"] == pytest.approx(0.5)

    def test_output_strict_superset(self):
        """Found all expected + extras -> P=3/5, R=1.0, F1=0.75."""
        r = s(["a", "b", "c", "d", "e"], ["a", "b", "c"])
        assert r["precision"] == pytest.approx(3 / 5)
        assert r["recall"] == 1.0
        assert r["f1"] == pytest.approx(0.75)

    def test_symmetric_partial(self):
        """2 of 3 on each side -> P=2/3, R=2/3, F1=2/3."""
        r = s(["a", "b", "x"], ["a", "b", "c"])
        assert r["precision"] == pytest.approx(2 / 3)
        assert r["recall"] == pytest.approx(2 / 3)
        assert r["f1"] == pytest.approx(2 / 3)

    def test_one_correct_one_wrong(self):
        """P=1/2, R=1/2, F1=1/2."""
        r = s(["a", "x"], ["a", "b"])
        assert r == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_one_tp_many_fp(self):
        """P=1/4, R=1.0, F1=2/5."""
        r = s(["a", "x", "y", "z"], ["a"])
        assert r["precision"] == pytest.approx(1 / 4)
        assert r["recall"] == 1.0
        assert r["f1"] == pytest.approx(2 * (1 / 4) * 1.0 / (1 / 4 + 1.0))

    def test_one_tp_many_fn(self):
        """P=1.0, R=1/4, F1=2/5."""
        r = s(["a"], ["a", "b", "c", "d"])
        assert r["precision"] == 1.0
        assert r["recall"] == pytest.approx(1 / 4)
        assert r["f1"] == pytest.approx(2 * 1.0 * (1 / 4) / (1.0 + 1 / 4))


# ═════════════════════════════════════════════════════════════════════════
#  Single value vs list mixing
# ═════════════════════════════════════════════════════════════════════════

class TestMixedTypes:
    def test_single_string_output_list_expected_match(self):
        r = s("paris", ["paris", "france"])
        assert r["precision"] == 1.0
        assert r["recall"] == 0.5
        assert r["f1"] == pytest.approx(2 / 3)

    def test_single_string_output_list_expected_no_match(self):
        r = s("london", ["paris", "france"])
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_list_output_single_string_expected_match(self):
        r = s(["paris", "france"], "paris")
        assert r["precision"] == 0.5
        assert r["recall"] == 1.0
        assert r["f1"] == pytest.approx(2 / 3)

    def test_list_output_single_string_expected_no_match(self):
        r = s(["paris", "france"], "london")
        assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_int_output_list_expected(self):
        r = s(1, [1, 2, 3])
        assert r["precision"] == 1.0
        assert r["recall"] == pytest.approx(1 / 3)


# ═════════════════════════════════════════════════════════════════════════
#  Duplicates (set-based, so dupes collapse)
# ═════════════════════════════════════════════════════════════════════════

class TestDuplicates:
    def test_duplicates_in_output(self):
        """Dupes in output shouldn't inflate FP count."""
        r = s(["a", "a", "a"], ["a"])
        assert r == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_duplicates_in_expected(self):
        r = s(["a"], ["a", "a", "a"])
        assert r == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_duplicates_both_sides_partial(self):
        # set-based: output={a,b}, expected={a,c} -> TP=1, FP=1, FN=1
        r = s(["a", "a", "b"], ["a", "a", "c"])
        assert r == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_all_same_element(self):
        r = s(["a", "a"], ["a", "a"])
        assert r == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


# ═════════════════════════════════════════════════════════════════════════
#  Score ranges -- every non-None score must be in [0, 1]
# ═════════════════════════════════════════════════════════════════════════

class TestScoreRanges:
    CASES = [
        (["a"], ["a"]),
        (["a", "b"], ["c"]),
        ([], ["a"]),
        (["a"], []),
        ("x", "x"),
        ("x", "y"),
        (["a", "b", "c", "d"], ["a"]),
        (["a"], ["a", "b", "c", "d"]),
        (None, None),
        (None, ["a"]),
        (["a"], None),
    ]

    @pytest.mark.parametrize("output,expected", CASES)
    def test_in_valid_range(self, output, expected):
        r = s(output, expected)
        for name, val in r.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{name}={val} out of range"


# ═════════════════════════════════════════════════════════════════════════
#  F1 is always the harmonic mean of P and R
# ═════════════════════════════════════════════════════════════════════════

class TestF1IsHarmonicMean:
    CASES = [
        (["a", "b", "x"], ["a", "b", "c"]),
        (["a"], ["a", "b", "c"]),
        (["a", "b", "c", "d", "e"], ["a", "b", "c"]),
        (["a", "x"], ["a", "b"]),
        ("paris", ["paris", "france"]),
        (["a"], ["a"]),
        (["x"], ["y"]),
    ]

    @pytest.mark.parametrize("output,expected", CASES)
    def test_f1_equals_harmonic_mean(self, output, expected):
        r = precision_recall_f1(output, expected)
        p, rc, f = r["precision"], r["recall"], r["f1"]
        if p is not None and rc is not None and (p + rc) > 0:
            expected_f1 = 2 * p * rc / (p + rc)
            assert f == pytest.approx(expected_f1)
        elif p is None or rc is None:
            assert f is None
        else:
            # P and R both 0.0 -> F1 is 0.0 (defined but worst possible)
            assert f == 0.0


# ═════════════════════════════════════════════════════════════════════════
#  Metadata correctness
# ═════════════════════════════════════════════════════════════════════════

class TestMetadata:
    def test_tp_fp_fn_counts(self):
        r = precision_recall_f1(["a", "b", "x"], ["a", "b", "c"])
        assert r["metadata"] == {"tp": 2, "fp": 1, "fn": 1}

    def test_all_tp(self):
        r = precision_recall_f1(["a", "b"], ["a", "b"])
        assert r["metadata"] == {"tp": 2, "fp": 0, "fn": 0}

    def test_all_fp(self):
        r = precision_recall_f1(["x", "y"], ["a", "b"])
        assert r["metadata"] == {"tp": 0, "fp": 2, "fn": 2}

    def test_all_fn(self):
        r = precision_recall_f1([], ["a", "b"])
        assert r["metadata"] == {"tp": 0, "fp": 0, "fn": 2}

    def test_empty_both(self):
        r = precision_recall_f1([], [])
        assert r["metadata"] == {"tp": 0, "fp": 0, "fn": 0}

    def test_single_string(self):
        r = precision_recall_f1("a", "a")
        assert r["metadata"] == {"tp": 1, "fp": 0, "fn": 0}

    def test_tp_plus_fp_equals_output_size(self):
        """TP + FP should always equal |output_set|."""
        r = precision_recall_f1(["a", "b", "c", "x", "y"], ["a", "b", "c", "d"])
        m = r["metadata"]
        assert m["tp"] + m["fp"] == 5  # len of output set
        assert m["tp"] + m["fn"] == 4  # len of expected set


# ═════════════════════════════════════════════════════════════════════════
#  Exact expected values from the example eval dataset
# ═════════════════════════════════════════════════════════════════════════

class TestEvalDatasetRows:
    """Validate the exact numbers the example eval should produce."""

    def test_perfect_match(self):
        r = s(["eiffel tower", "paris", "france"], ["eiffel tower", "paris", "france"])
        assert r == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_partial_overlap(self):
        r = s(["apple", "iphone", "samsung"], ["apple", "iphone", "macbook"])
        assert r["precision"] == pytest.approx(2 / 3)
        assert r["recall"] == pytest.approx(2 / 3)
        assert r["f1"] == pytest.approx(2 / 3)

    def test_no_predictions(self):
        r = s([], ["concert"])
        assert r["precision"] is None
        assert r["recall"] == 0.0
        assert r["f1"] is None

    def test_high_precision_low_recall(self):
        r = s(["python"], ["python", "javascript", "web development"])
        assert r["precision"] == 1.0
        assert r["recall"] == pytest.approx(1 / 3)
        assert r["f1"] == pytest.approx(0.5)

    def test_low_precision_perfect_recall(self):
        r = s(
            ["nasa", "mars", "perseverance", "rover", "space"],
            ["nasa", "mars", "perseverance"],
        )
        assert r["precision"] == pytest.approx(3 / 5)
        assert r["recall"] == 1.0
        assert r["f1"] == pytest.approx(0.75)


# ═════════════════════════════════════════════════════════════════════════
#  AGGREGATION TESTS
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateUniform:
    """Datasets where every row has the same characteristics."""

    def test_all_perfect(self):
        dataset = [
            (["a", "b"], ["a", "b"]),
            (["x", "y", "z"], ["x", "y", "z"]),
            ("single", "single"),
        ]
        assert agg(dataset) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_all_complete_mismatch(self):
        dataset = [
            (["a"], ["z"]),
            (["x", "y"], ["p", "q"]),
        ]
        assert agg(dataset) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_all_empty_outputs(self):
        """Every row predicted nothing -> P=None everywhere, R=0 everywhere."""
        dataset = [
            ([], ["a", "b"]),
            ([], ["x"]),
            ([], ["p", "q", "r"]),
        ]
        r = agg(dataset)
        assert r["precision"] is None
        assert r["recall"] == 0.0
        assert r["f1"] is None

    def test_all_empty_expected(self):
        """Every row had nothing to find -> R=None everywhere, P=0 everywhere."""
        dataset = [
            (["a", "b"], []),
            (["x"], []),
        ]
        r = agg(dataset)
        assert r["precision"] == 0.0
        assert r["recall"] is None
        assert r["f1"] is None

    def test_all_empty_both(self):
        """Nothing predicted, nothing expected anywhere."""
        dataset = [([], []), ([], []), (None, None)]
        r = agg(dataset)
        assert r["precision"] is None
        assert r["recall"] is None
        assert r["f1"] is None


class TestAggregateNoneExclusion:
    """Verify that None rows are correctly excluded from the average."""

    def test_one_none_precision_excluded(self):
        dataset = [
            (["a"], ["a"]),          # P=1.0
            ([], ["b"]),             # P=None  (excluded)
            (["c", "x"], ["c"]),     # P=1/2
        ]
        r = aggregate_scores(dataset)
        assert r["counts"]["precision"] == 2
        assert r["precision"] == pytest.approx((1.0 + 0.5) / 2)

    def test_one_none_recall_excluded(self):
        dataset = [
            (["a"], ["a"]),          # R=1.0
            (["b"], []),             # R=None  (excluded)
            (["c"], ["c", "d"]),     # R=1/2
        ]
        r = aggregate_scores(dataset)
        assert r["counts"]["recall"] == 2
        assert r["recall"] == pytest.approx((1.0 + 0.5) / 2)

    def test_multiple_none_rows(self):
        dataset = [
            ([], ["a"]),       # P=None
            ([], ["b"]),       # P=None
            ([], ["c"]),       # P=None
            (["d"], ["d"]),    # P=1.0
        ]
        r = aggregate_scores(dataset)
        assert r["counts"]["precision"] == 1
        assert r["precision"] == 1.0

    def test_f1_none_when_precision_all_none(self):
        dataset = [
            ([], ["a"]),
            ([], ["b"]),
        ]
        r = agg(dataset)
        assert r["precision"] is None
        assert r["recall"] == 0.0
        assert r["f1"] is None


class TestAggregateSpecificValues:
    """Check exact aggregate numbers for carefully constructed datasets."""

    def test_half_perfect_half_zero(self):
        dataset = [
            (["a"], ["a"]),
            (["x"], ["y"]),
        ]
        assert agg(dataset) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_three_rows_simple(self):
        dataset = [
            (["a"], ["a"]),
            (["a", "x"], ["a"]),
            (["x", "y"], ["a", "b"]),
        ]
        r = agg(dataset)
        assert r["precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert r["recall"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert r["f1"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)

    def test_high_precision_low_recall_dataset(self):
        dataset = [
            (["a"], ["a", "b", "c"]),
            (["x"], ["x", "y"]),
            (["p"], ["p", "q", "r", "s"]),
        ]
        r = agg(dataset)
        assert r["precision"] == 1.0
        assert r["recall"] == pytest.approx((1 / 3 + 1 / 2 + 1 / 4) / 3)
        assert r["f1"] == pytest.approx((0.5 + 2 / 3 + 2 / 5) / 3)

    def test_low_precision_perfect_recall_dataset(self):
        dataset = [
            (["a", "x"], ["a"]),
            (["a", "b", "x", "y"], ["a", "b"]),
        ]
        r = agg(dataset)
        assert r["precision"] == 0.5
        assert r["recall"] == 1.0
        assert r["f1"] == pytest.approx(2 / 3)

    def test_asymmetric_none_exclusion(self):
        dataset = [
            (["a"], ["a"]),
            ([], ["b"]),
            (["c"], []),
        ]
        r = aggregate_scores(dataset)
        assert r["counts"]["precision"] == 2
        assert r["precision"] == 0.5
        assert r["counts"]["recall"] == 2
        assert r["recall"] == 0.5
        assert r["counts"]["f1"] == 1
        assert r["f1"] == 1.0


class TestAggregateEvalDataset:
    """The exact dataset from the example eval."""

    EVAL_DATASET = [
        (["eiffel tower", "paris", "france"], ["eiffel tower", "paris", "france"]),
        (["apple", "iphone", "samsung"], ["apple", "iphone", "macbook"]),
        ([], ["concert"]),
        (["python"], ["python", "javascript", "web development"]),
        (["nasa", "mars", "perseverance", "rover", "space"], ["nasa", "mars", "perseverance"]),
    ]

    def test_aggregate_precision(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["counts"]["precision"] == 4
        expected = (1.0 + 2 / 3 + 1.0 + 3 / 5) / 4
        assert r["precision"] == pytest.approx(expected)

    def test_aggregate_recall(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["counts"]["recall"] == 5
        expected = (1.0 + 2 / 3 + 0.0 + 1 / 3 + 1.0) / 5
        assert r["recall"] == pytest.approx(expected)

    def test_aggregate_f1(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["counts"]["f1"] == 4
        expected = (1.0 + 2 / 3 + 0.5 + 0.75) / 4
        assert r["f1"] == pytest.approx(expected)

    def test_aggregate_precision_approx_value(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["precision"] == pytest.approx(0.8167, abs=1e-3)

    def test_aggregate_recall_approx_value(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["recall"] == pytest.approx(0.6, abs=1e-3)

    def test_aggregate_f1_approx_value(self):
        r = aggregate_scores(self.EVAL_DATASET)
        assert r["f1"] == pytest.approx(0.7292, abs=1e-3)


class TestAggregateSingleRow:
    """Aggregate of a one-row dataset should equal the row score."""

    CASES = [
        (["a", "b"], ["a", "b"]),
        (["a", "x"], ["a", "b"]),
        ([], ["a"]),
        ("paris", "paris"),
        (["a", "b", "c"], ["x", "y", "z"]),
    ]

    @pytest.mark.parametrize("output,expected", CASES)
    def test_single_row_equals_row_score(self, output, expected):
        row = precision_recall_f1(output, expected)
        a = agg([(output, expected)])
        assert a["precision"] == row["precision"]
        assert a["recall"] == row["recall"]
        assert a["f1"] == row["f1"]


class TestAggregateLargeDatasets:
    """Stress tests with larger datasets."""

    def test_100_perfect_rows(self):
        dataset = [(["a", "b", "c"], ["a", "b", "c"])] * 100
        assert agg(dataset) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_100_mixed_rows(self):
        perfect = [(["a"], ["a"])] * 50
        zero = [(["x"], ["y"])] * 50
        dataset = perfect + zero
        assert agg(dataset) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_99_none_precision_one_real(self):
        dataset = [([], ["a"])] * 99 + [(["a"], ["a"])]
        r = aggregate_scores(dataset)
        assert r["counts"]["precision"] == 1
        assert r["precision"] == 1.0
        assert r["counts"]["recall"] == 100
        assert r["recall"] == pytest.approx(1 / 100)

    def test_mixed_types_aggregate(self):
        dataset = [
            ("paris", "paris"),
            ("london", ["london", "uk"]),
            (["nyc", "us"], "nyc"),
        ]
        r = agg(dataset)
        assert r["precision"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
        assert r["recall"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
        assert r["f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)


class TestAggregateCountsAlwaysCorrect:
    """Parametrized: verify counts match the number of non-None row scores."""

    DATASETS = {
        "all_defined": [
            (["a"], ["a"]),
            (["x"], ["y"]),
        ],
        "one_none_precision": [
            (["a"], ["a"]),
            ([], ["b"]),
        ],
        "one_none_recall": [
            (["a"], ["a"]),
            (["b"], []),
        ],
        "mixed_nones": [
            (["a"], ["a"]),
            ([], ["b"]),
            (["c"], []),
            ([], []),
        ],
    }

    @pytest.mark.parametrize("name,dataset", DATASETS.items())
    def test_counts_match_non_none_rows(self, name, dataset):
        r = aggregate_scores(dataset)
        for metric in ("precision", "recall", "f1"):
            non_none = sum(
                1 for row in r["row_scores"] if row[metric] is not None
            )
            assert r["counts"][metric] == non_none, (
                f"{name}: {metric} count {r['counts'][metric]} != {non_none} non-None rows"
            )
