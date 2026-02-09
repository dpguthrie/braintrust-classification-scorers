# braintrust-classification-scorers

Set-based **precision**, **recall**, and **F1** scorers for [Braintrust](https://www.braintrust.dev/) evals.

Use these when your eval output and expected values are **lists of items** (tags, entities, categories, keywords, etc.) and you want to measure how well the output matches.

## Quick start

```bash
pip install braintrust
```

Everything lives in `scorer.py`. Drop the scorer into any Braintrust `Eval` call:

```python
from braintrust import Eval
from scorer import precision_recall_f1_scorer

Eval(
    "My Project",
    data=lambda: [
        {
            "input": "Extract keywords from this text...",
            "expected": ["python", "javascript", "web development"],
        },
    ],
    task=my_keyword_extractor,
    scores=[precision_recall_f1_scorer],
)
```

Each row will produce three scores: `precision`, `recall`, and `f1`. Braintrust aggregates them automatically.

## How the math works

Both `output` and `expected` are converted to **sets**, then compared:

| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | `\|output ∩ expected\| / \|output\|` | Of what was predicted, how much was correct? |
| **Recall** | `\|output ∩ expected\| / \|expected\|` | Of what was correct, how much was found? |
| **F1** | `2 * P * R / (P + R)` | Harmonic mean of precision and recall |

### Confusion matrix breakdown

For each row, items are classified as:

- **True Positives (TP)** -- in both `output` and `expected`
- **False Positives (FP)** -- in `output` but not `expected`
- **False Negatives (FN)** -- in `expected` but not `output`

Then: `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`.

## None handling and aggregation

This is the most important design decision. Braintrust's default aggregation is **average of non-None scores**. When a metric is undefined for a row, the scorer returns `Score(score=None)`, and that row is **excluded from the average**.

| Scenario | Precision | Recall | F1 |
|---|---|---|---|
| Normal row (TP + FP > 0, TP + FN > 0) | `TP/(TP+FP)` | `TP/(TP+FN)` | harmonic mean |
| Empty output (no predictions) | **None** | `0.0` | **None** |
| Empty expected (nothing to find) | `0.0` | **None** | **None** |
| Both empty | **None** | **None** | **None** |
| Complete mismatch (P=0, R=0) | `0.0` | `0.0` | `0.0` |

Why this matters: if a system returns no predictions for a row, precision is meaningless (0/0), so it should not drag down the aggregate. But recall is well-defined (0 out of N expected) and should count.

## Worked example

The [examples/keyword_extraction.py](examples/keyword_extraction.py) eval produces these per-row scores:

| Row | Output | Expected | P | R | F1 |
|---|---|---|---|---|---|
| Eiffel Tower | {eiffel tower, paris, france} | {eiffel tower, paris, france} | 1.0 | 1.0 | 1.0 |
| Apple/iPhone | {apple, iphone, samsung} | {apple, iphone, macbook} | 0.667 | 0.667 | 0.667 |
| Concert | {} | {concert} | **None** | 0.0 | **None** |
| Python/JS | {python} | {python, javascript, web development} | 1.0 | 0.333 | 0.5 |
| NASA/Mars | {nasa, mars, perseverance, rover, space} | {nasa, mars, perseverance} | 0.6 | 1.0 | 0.75 |

**Aggregates (what you see in the Braintrust UI):**

- **Precision**: (1.0 + 0.667 + 1.0 + 0.6) / **4** = **0.817** -- row 2 excluded
- **Recall**: (1.0 + 0.667 + 0.0 + 0.333 + 1.0) / **5** = **0.600**
- **F1**: (1.0 + 0.667 + 0.5 + 0.75) / **4** = **0.729** -- row 2 excluded

## Input normalization

The scorer accepts flexible input types. All of these work:

```python
# Lists (standard)
output=["paris", "france"], expected=["paris", "london"]

# Single strings (auto-wrapped)
output="paris", expected=["paris", "france"]

# Mixed
output=["paris"], expected="paris"

# Tuples, sets
output=("a", "b"), expected={"a", "c"}

# None (treated as empty)
output=None, expected=["a"]
```

Duplicates are collapsed (set-based comparison), and order is irrelevant.

## API reference

### `precision_recall_f1_scorer(input, output, expected)`

The Braintrust-compatible scorer. Returns a list of three `Score` objects (`precision`, `recall`, `f1`). Each precision/recall `Score` includes `metadata` with the raw `tp`, `fp`, `fn` counts.

```python
from scorer import precision_recall_f1_scorer

Eval("Project", data=..., task=..., scores=[precision_recall_f1_scorer])
```

### `precision_recall_f1(output, expected)`

The underlying scoring function. Returns a plain dict -- useful for testing or debugging outside of an eval run.

```python
from scorer import precision_recall_f1

result = precision_recall_f1(["a", "b", "x"], ["a", "b", "c"])
# {
#     "precision": 0.6667,
#     "recall": 0.6667,
#     "f1": 0.6667,
#     "metadata": {"tp": 2, "fp": 1, "fn": 1},
# }
```

### `aggregate_scores(dataset)`

Simulates Braintrust's default average aggregation locally. Useful for offline testing.

```python
from scorer import aggregate_scores

result = aggregate_scores([
    (["a", "b"], ["a", "b", "c"]),   # P=1.0, R=2/3
    ([], ["x"]),                      # P=None, R=0.0
])
# result["precision"] == 1.0       (only 1 contributing row)
# result["recall"] == 0.333        (average of 2/3 and 0.0)
# result["counts"]["precision"] == 1
# result["counts"]["recall"] == 2
```

## Development

```bash
git clone https://github.com/braintrustdata/braintrust-classification-scorers.git
cd braintrust-classification-scorers
pip install braintrust pytest pytest-asyncio
```

### Running tests

```bash
pytest -v
```

## License

MIT
