"""Example: keyword extraction eval with precision/recall/F1 scoring.

Run with:
    braintrust eval examples/keyword_extraction.py

Or, to run locally without sending logs to Braintrust:
    python examples/keyword_extraction.py
"""

from braintrust import Eval

from braintrust_classification_scorers import precision_recall_f1_scorer

# ---------------------------------------------------------------------------
# Fake keyword extractor -- hardcoded per input to exercise different
# precision/recall scenarios without calling an LLM.
# ---------------------------------------------------------------------------
FAKE_EXTRACTIONS = {
    "The Eiffel Tower is in Paris, France": ["eiffel tower", "paris", "france"],
    "Apple released the new iPhone and MacBook": ["apple", "iphone", "samsung"],
    "The concert was amazing last night": [],  # system finds nothing
    "Python and JavaScript are used in web development": ["python"],
    "NASA launched the Mars rover Perseverance": [
        "nasa",
        "mars",
        "perseverance",
        "rover",
        "space",
    ],
}


def fake_keyword_extractor(input):
    return FAKE_EXTRACTIONS.get(input, [])


# ---------------------------------------------------------------------------
# Dataset -- each row exercises a different precision/recall scenario.
#
#   Row 0: Perfect match          P=1.0   R=1.0   F1=1.0
#   Row 1: Partial overlap        P=2/3   R=2/3   F1=2/3
#   Row 2: No predictions         P=None  R=0.0   F1=None
#   Row 3: High precision/low R   P=1.0   R=1/3   F1=0.5
#   Row 4: Low precision/full R   P=3/5   R=1.0   F1=0.75
#
# Expected aggregates (Braintrust UI):
#   precision ≈ 0.817  (4 rows, row 2 excluded)
#   recall    = 0.600  (5 rows)
#   f1        ≈ 0.729  (4 rows, row 2 excluded)
# ---------------------------------------------------------------------------
DATASET = [
    {
        "input": "The Eiffel Tower is in Paris, France",
        "expected": ["eiffel tower", "paris", "france"],
    },
    {
        "input": "Apple released the new iPhone and MacBook",
        "expected": ["apple", "iphone", "macbook"],
    },
    {
        "input": "The concert was amazing last night",
        "expected": ["concert"],
    },
    {
        "input": "Python and JavaScript are used in web development",
        "expected": ["python", "javascript", "web development"],
    },
    {
        "input": "NASA launched the Mars rover Perseverance",
        "expected": ["nasa", "mars", "perseverance"],
    },
]

Eval(
    "Keyword Extraction - Precision/Recall",
    data=lambda: DATASET,
    task=fake_keyword_extractor,
    scores=[precision_recall_f1_scorer],
)
