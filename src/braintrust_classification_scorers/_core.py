"""Core scoring logic -- no braintrust dependency.

This module can be imported and tested without the Braintrust SDK installed.

Public API:
  - precision_recall_f1()   : compute P, R, F1 for a single row
  - aggregate_scores()      : simulate Braintrust's default average aggregation
"""

from __future__ import annotations

from typing import Any


def _to_set(value: Any) -> set:
    """Normalize a value into a set for comparison.

    Handles: list, set, tuple, single string, single non-string scalar, None.
    A bare string like ``"paris"`` becomes ``{"paris"}``, **not**
    ``{"p","a","r","i","s"}``.
    """
    if value is None:
        return set()
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return set(value)
    # Single scalar (string, int, etc.) -- wrap it
    return {value}


def precision_recall_f1(
    output: Any,
    expected: Any,
) -> dict[str, Any]:
    """Compute set-based precision, recall, and F1 for a single eval row.

    Both *output* and *expected* are normalised to sets before comparison.
    Accepts lists, tuples, single scalars (including bare strings), or
    ``None``.

    Args:
        output:   Predicted items -- list, single value, or ``None``.
        expected: Ground-truth items -- list, single value, or ``None``.

    Returns:
        A dict with keys ``"precision"``, ``"recall"``, ``"f1"`` (each
        ``float | None``) and a ``"metadata"`` sub-dict containing the raw
        ``tp``, ``fp``, ``fn`` counts.

        ``None`` means the metric is *undefined* for this row:

        * **precision** is ``None`` when there are no predictions
          (``TP + FP == 0``).
        * **recall** is ``None`` when there are no expected items
          (``TP + FN == 0``).
        * **f1** is ``None`` when either precision or recall is ``None``.

        When a Braintrust scorer returns ``Score(score=None)``, the row is
        excluded from the default average aggregation, which is exactly the
        behaviour we want.
    """
    output_set = _to_set(output)
    expected_set = _to_set(expected)

    tp = len(output_set & expected_set)
    fp = len(output_set - expected_set)
    fn = len(expected_set - output_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None:
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        f1 = None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metadata": {"tp": tp, "fp": fp, "fn": fn},
    }


def aggregate_scores(
    dataset: list[tuple[Any, Any]],
) -> dict[str, Any]:
    """Simulate Braintrust's default average aggregation over a dataset.

    For each metric, collects all non-``None`` row scores and returns their
    mean.  If every row is ``None`` for a metric, the aggregate is ``None``.

    This is useful for writing offline tests that verify the numbers a user
    would see in the Braintrust UI without actually calling the API.

    Args:
        dataset: List of ``(output, expected)`` tuples, one per eval row.

    Returns:
        A dict with keys ``"precision"``, ``"recall"``, ``"f1"``
        (each ``float | None``), plus ``"counts"`` (how many non-``None``
        values contributed) and ``"row_scores"`` (the per-row dicts).
    """
    metrics = ("precision", "recall", "f1")
    collectors: dict[str, list[float]] = {m: [] for m in metrics}
    row_scores: list[dict[str, Any]] = []

    for output, expected in dataset:
        result = precision_recall_f1(output, expected)
        row_scores.append(result)
        for m in metrics:
            if result[m] is not None:
                collectors[m].append(result[m])

    aggregated: dict[str, Any] = {
        m: sum(vals) / len(vals) if vals else None
        for m, vals in collectors.items()
    }
    aggregated["counts"] = {m: len(vals) for m, vals in collectors.items()}
    aggregated["row_scores"] = row_scores
    return aggregated
