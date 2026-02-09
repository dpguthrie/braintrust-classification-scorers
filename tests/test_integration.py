"""Integration tests that run real Braintrust evals (locally, no API key needed).

These tests call EvalAsync(..., no_send_logs=True) to exercise the full
Braintrust framework -- scorer invocation, per-row score logging, None
handling, and aggregate summarization -- then assert on the actual summary
scores that a user would see in the Braintrust UI.
"""

import pytest
from braintrust import EvalAsync, Score

from scorer import precision_recall_f1, precision_recall_f1_scorer


# ── Helpers ──────────────────────────────────────────────────────────────

async def run_eval(name, dataset, task=None):
    """Run a local eval and return the EvalResultWithSummary."""
    task = task or (lambda input: input.get("output", []))
    return await EvalAsync(
        name,
        data=lambda: dataset,
        task=task,
        scores=[precision_recall_f1_scorer],
        no_send_logs=True,
    )


def summary_scores(result):
    """Extract {metric: value} from the summary."""
    return {
        name: ss.score
        for name, ss in result.summary.scores.items()
    }


# ═════════════════════════════════════════════════════════════════════════
#  Per-row score verification (through the real framework)
# ═════════════════════════════════════════════════════════════════════════

class TestPerRowThroughFramework:

    @pytest.mark.asyncio
    async def test_perfect_match_row(self):
        dataset = [
            {"input": {"output": ["a", "b", "c"]}, "expected": ["a", "b", "c"]},
        ]
        result = await run_eval("test-perfect-row", dataset)
        row = result.results[0]
        assert row.scores["precision"] == 1.0
        assert row.scores["recall"] == 1.0
        assert row.scores["f1"] == 1.0

    @pytest.mark.asyncio
    async def test_partial_match_row(self):
        dataset = [
            {"input": {"output": ["a", "b", "x"]}, "expected": ["a", "b", "c"]},
        ]
        result = await run_eval("test-partial-row", dataset)
        row = result.results[0]
        assert row.scores["precision"] == pytest.approx(2 / 3)
        assert row.scores["recall"] == pytest.approx(2 / 3)
        assert row.scores["f1"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_empty_output_row(self):
        dataset = [
            {"input": {"output": []}, "expected": ["a"]},
        ]
        result = await run_eval("test-empty-output-row", dataset)
        row = result.results[0]
        assert row.scores.get("precision") is None
        assert row.scores["recall"] == 0.0
        assert row.scores.get("f1") is None

    @pytest.mark.asyncio
    async def test_single_string_output_row(self):
        dataset = [
            {"input": {"output": "paris"}, "expected": ["paris", "france"]},
        ]
        result = await run_eval(
            "test-single-string-row",
            dataset,
            task=lambda input: input["output"],
        )
        row = result.results[0]
        assert row.scores["precision"] == 1.0
        assert row.scores["recall"] == 0.5


# ═════════════════════════════════════════════════════════════════════════
#  Aggregate: uniform datasets
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateUniform:
    @pytest.mark.asyncio
    async def test_all_perfect(self):
        dataset = [
            {"input": {"output": ["a", "b"]}, "expected": ["a", "b"]},
            {"input": {"output": ["x", "y", "z"]}, "expected": ["x", "y", "z"]},
        ]
        result = await run_eval("test-all-perfect", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    @pytest.mark.asyncio
    async def test_all_mismatch(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["z"]},
            {"input": {"output": ["x", "y"]}, "expected": ["p", "q"]},
        ]
        result = await run_eval("test-all-mismatch", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
#  Aggregate: None exclusion
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateNoneExclusion:
    @pytest.mark.asyncio
    async def test_empty_output_excluded_from_precision(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a"]},
            {"input": {"output": []}, "expected": ["b"]},
            {"input": {"output": ["c", "x"]}, "expected": ["c"]},
        ]
        result = await run_eval("test-none-precision", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == pytest.approx(0.75)
        assert scores["recall"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_empty_expected_excluded_from_recall(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a"]},
            {"input": {"output": ["b"]}, "expected": []},
        ]
        result = await run_eval("test-none-recall", dataset)
        scores = summary_scores(result)
        assert scores["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_all_empty_outputs_precision_missing(self):
        dataset = [
            {"input": {"output": []}, "expected": ["a"]},
            {"input": {"output": []}, "expected": ["b"]},
        ]
        result = await run_eval("test-all-none-precision", dataset)
        assert "precision" not in result.summary.scores
        assert result.summary.scores["recall"].score == 0.0

    @pytest.mark.asyncio
    async def test_f1_absent_when_precision_all_none(self):
        dataset = [
            {"input": {"output": []}, "expected": ["a"]},
            {"input": {"output": []}, "expected": ["b"]},
        ]
        result = await run_eval("test-f1-absent", dataset)
        assert "f1" not in result.summary.scores


# ═════════════════════════════════════════════════════════════════════════
#  Aggregate: exact values for specific datasets
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateExactValues:
    @pytest.mark.asyncio
    async def test_half_perfect_half_zero(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a"]},
            {"input": {"output": ["x"]}, "expected": ["y"]},
        ]
        result = await run_eval("test-half-half", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 0.5
        assert scores["recall"] == 0.5
        assert scores["f1"] == 0.5

    @pytest.mark.asyncio
    async def test_high_precision_low_recall(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a", "b", "c"]},
            {"input": {"output": ["x"]}, "expected": ["x", "y"]},
            {"input": {"output": ["p"]}, "expected": ["p", "q", "r", "s"]},
        ]
        result = await run_eval("test-high-p-low-r", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 1.0
        assert scores["recall"] == pytest.approx((1 / 3 + 1 / 2 + 1 / 4) / 3)

    @pytest.mark.asyncio
    async def test_asymmetric_none_exclusion(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a"]},
            {"input": {"output": []}, "expected": ["b"]},
            {"input": {"output": ["c"]}, "expected": []},
        ]
        result = await run_eval("test-asymmetric", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 0.5
        assert scores["recall"] == 0.5
        assert scores["f1"] == 1.0

    @pytest.mark.asyncio
    async def test_three_row_mixed(self):
        dataset = [
            {"input": {"output": ["a"]}, "expected": ["a"]},
            {"input": {"output": ["a", "x"]}, "expected": ["a"]},
            {"input": {"output": ["x", "y"]}, "expected": ["a", "b"]},
        ]
        result = await run_eval("test-three-mixed", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert scores["recall"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert scores["f1"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)


# ═════════════════════════════════════════════════════════════════════════
#  Aggregate: the actual eval dataset from the example
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateEvalDataset:

    EXTRACTIONS = {
        "The Eiffel Tower is in Paris, France": ["eiffel tower", "paris", "france"],
        "Apple released the new iPhone and MacBook": ["apple", "iphone", "samsung"],
        "The concert was amazing last night": [],
        "Python and JavaScript are used in web development": ["python"],
        "NASA launched the Mars rover Perseverance": ["nasa", "mars", "perseverance", "rover", "space"],
    }

    DATASET = [
        {"input": "The Eiffel Tower is in Paris, France", "expected": ["eiffel tower", "paris", "france"]},
        {"input": "Apple released the new iPhone and MacBook", "expected": ["apple", "iphone", "macbook"]},
        {"input": "The concert was amazing last night", "expected": ["concert"]},
        {"input": "Python and JavaScript are used in web development", "expected": ["python", "javascript", "web development"]},
        {"input": "NASA launched the Mars rover Perseverance", "expected": ["nasa", "mars", "perseverance"]},
    ]

    @pytest.mark.asyncio
    async def test_aggregate_precision(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        expected_p = (1.0 + 2 / 3 + 1.0 + 3 / 5) / 4
        assert result.summary.scores["precision"].score == pytest.approx(expected_p)

    @pytest.mark.asyncio
    async def test_aggregate_recall(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        expected_r = (1.0 + 2 / 3 + 0.0 + 1 / 3 + 1.0) / 5
        assert result.summary.scores["recall"].score == pytest.approx(expected_r)

    @pytest.mark.asyncio
    async def test_aggregate_f1(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        expected_f1 = (1.0 + 2 / 3 + 0.5 + 0.75) / 4
        assert result.summary.scores["f1"].score == pytest.approx(expected_f1)

    @pytest.mark.asyncio
    async def test_aggregate_approx_values(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        assert result.summary.scores["precision"].score == pytest.approx(0.8167, abs=1e-3)
        assert result.summary.scores["recall"].score == pytest.approx(0.6, abs=1e-3)
        assert result.summary.scores["f1"].score == pytest.approx(0.7292, abs=1e-3)

    @pytest.mark.asyncio
    async def test_precision_absent_from_concert_row(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        concert_row = next(r for r in result.results if r.input == "The concert was amazing last night")
        assert concert_row.scores.get("precision") is None
        assert concert_row.scores["recall"] == 0.0
        assert concert_row.scores.get("f1") is None

    @pytest.mark.asyncio
    async def test_all_five_rows_present(self):
        result = await EvalAsync(
            "Integration Test", data=lambda: self.DATASET,
            task=lambda input: self.EXTRACTIONS[input],
            scores=[precision_recall_f1_scorer], no_send_logs=True,
        )
        assert len(result.results) == 5


# ═════════════════════════════════════════════════════════════════════════
#  Aggregate: larger dataset stress test
# ═════════════════════════════════════════════════════════════════════════

class TestAggregateLargeDataset:
    @pytest.mark.asyncio
    async def test_50_perfect_50_zero(self):
        perfect = [{"input": {"output": ["a"]}, "expected": ["a"]}] * 50
        zero = [{"input": {"output": ["x"]}, "expected": ["y"]}] * 50
        dataset = perfect + zero
        result = await run_eval("test-large-50-50", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == pytest.approx(0.5)
        assert scores["recall"] == pytest.approx(0.5)
        assert scores["f1"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_99_none_precision_one_real(self):
        dataset = (
            [{"input": {"output": []}, "expected": ["a"]}] * 99
            + [{"input": {"output": ["a"]}, "expected": ["a"]}]
        )
        result = await run_eval("test-large-99-none", dataset)
        scores = summary_scores(result)
        assert scores["precision"] == 1.0
        assert scores["recall"] == pytest.approx(1 / 100)
