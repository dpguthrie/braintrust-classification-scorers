"""Set-based precision, recall, and F1 scorers for Braintrust evals.

Quick start::

    from braintrust import Eval
    from braintrust_classification_scorers import precision_recall_f1_scorer

    Eval(
        "My Project",
        data=lambda: [{"input": ..., "expected": [...]}],
        task=my_task,
        scores=[precision_recall_f1_scorer],
    )

The core math (``precision_recall_f1``, ``aggregate_scores``) has **no**
dependency on the Braintrust SDK and can be imported independently for
testing or offline analysis.
"""

from ._core import aggregate_scores, precision_recall_f1

__all__ = [
    "aggregate_scores",
    "precision_recall_f1",
    "precision_recall_f1_scorer",
]


def __getattr__(name: str):
    # Lazy import so the package works without braintrust installed
    # unless you actually use the scorer.
    if name == "precision_recall_f1_scorer":
        from .scorer import precision_recall_f1_scorer

        return precision_recall_f1_scorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
