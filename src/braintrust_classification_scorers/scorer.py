"""Braintrust-compatible scorer that returns Score objects.

Requires the ``braintrust`` package (install with
``pip install braintrust-classification-scorers[braintrust]``).
"""

from __future__ import annotations

from typing import Any

from braintrust import Score, current_span

from ._core import precision_recall_f1


def precision_recall_f1_scorer(
    input: Any,  # noqa: A002 – matches Braintrust scorer signature
    output: Any,
    expected: Any,
    **kwargs: Any,
) -> list[Score]:
    """Braintrust scorer returning precision, recall, and F1.

    Drop this directly into the ``scores`` list of a Braintrust ``Eval`` call::

        from braintrust import Eval
        from braintrust_classification_scorers import precision_recall_f1_scorer

        Eval(
            "My Project",
            data=lambda: [...],
            task=my_task,
            scores=[precision_recall_f1_scorer],
        )

    The scorer treats *output* and *expected* as sets and computes:

    * **precision** -- ``|output ∩ expected| / |output|``
    * **recall** -- ``|output ∩ expected| / |expected|``
    * **f1** -- harmonic mean of precision and recall

    Metrics that are undefined for a row (e.g. precision when the output is
    empty) are returned as ``Score(score=None)``, which Braintrust excludes
    from the default average aggregation.

    Each precision and recall ``Score`` includes ``metadata`` with the raw
    ``tp``, ``fp``, ``fn`` counts for debugging.
    """
    result = precision_recall_f1(output, expected)
    metadata = result.pop("metadata", {})
    current_span.log({"metrics": metadata})
    return [
        Score(name="precision", score=result["precision"], metadata=metadata),
        Score(name="recall", score=result["recall"], metadata=metadata),
        Score(name="f1", score=result["f1"]),
    ]
