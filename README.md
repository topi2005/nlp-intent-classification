# Banking Customer-Query Intent Classification

How much better is a transformer than a classical TF-IDF baseline at recognising what a banking customer is asking, and where does each approach fail?

This is a small, rigorous NLP project: public data, a documented evaluation protocol, hypotheses written **before** any experiment was run, and an honest error analysis.

## Status

| Step | What | Status |
|------|------|--------|
| 1 | Data loading, data-quality and leakage checks | Done |
| 2 | TF-IDF + linear SVM baseline | Done |
| 3 | Sentence embeddings + linear classifier | Planned |
| 4 | Fine-tuned DistilBERT (PyTorch, Hugging Face) | Planned |
| 5 | Error analysis, learning curve, robustness to typos | Planned |
| 6 | Final results and findings | Planned |

The results section below is filled in only as each experiment is actually run.

## Dataset

[Banking77](https://huggingface.co/datasets/PolyAI/banking77): customer-service queries from the banking domain, each labelled with one of 77 intents (for example `lost_or_stolen_card` or `card_arrival`). It was introduced in Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*.

Observed with `explore.py`:

| Property | Value |
|----------|-------|
| Train / test queries | 10,003 / 3,080 |
| Intents | 77 |
| Examples per class (train) | min 35, median 127, max 187 (about 5x imbalance) |
| Words per query | mean 11.9, median 10, max 79 |
| Duplicate queries inside train | 4 |
| Queries present in both train and test | 6 |
| Identical queries with conflicting labels | 0 |

### Data handling decisions

- **Duplicates in train are dropped.** Only 4, but there is no reason to keep them.
- **Test queries that also appear in train are removed from the test set,** so the test set is genuinely unseen. This affects 6 distinct queries (about 0.2% of the test set). Scores are therefore not exactly comparable with published Banking77 numbers.
- **Macro-F1 is the primary metric,** because the training classes are imbalanced and per-class behaviour matters. The test set is balanced (40 queries per intent, 3,074 after cleaning), so macro-F1 and accuracy are nearly identical on it; both are reported.
- Near-duplicate queries (not just exact matches) were **not** checked; this is a known limitation.

## Evaluation protocol

- Hyperparameters are chosen by **5-fold stratified cross-validation on the training set only**.
- The test set is used **once per final model**, never for tuning.
- Fixed random seed (42) throughout.
- Every experiment saves its metrics, per-class scores and predictions to `results/`, so the error analysis uses the actual predictions rather than re-running models.

## Hypotheses (written before running any experiment)

These were written down after the data checks and before running `baseline.py` or any model. They will be confirmed or refuted in the results; they are not edited afterwards.

1. **Confusion concentrates in card-related intents.** The lowest per-class F1 scores will be among intents about cards (lost or stolen card, card swallowed, card not working, contactless or virtual card not working, card acceptance), because their wording overlaps heavily.
2. **Small classes do worse.** Intents with fewer than about 60 training examples (such as `contactless_not_working`, `virtual_card_not_working` and `card_acceptance`) will have lower recall than the median class.
3. **Baseline accuracy.** The TF-IDF + linear SVM baseline will reach a test accuracy between 0.85 and 0.92, with macro-F1 slightly below accuracy.
4. **Transformer gain.** The fine-tuned DistilBERT will improve test macro-F1 over the baseline by roughly 3 to 7 points. The gain will come mostly from queries that share few words with training examples, and the confusable card intents will remain the hardest for both models.
5. **Data efficiency.** With only 10% of the training data, the sentence-embedding model will beat the TF-IDF baseline by a wider margin than it does with the full data.
6. **Robustness.** Adding random typos to the test queries will hurt the TF-IDF baseline more than the transformer, because word-level features break on misspellings while subword tokenisation degrades more gracefully.

## Methods

### Baseline (Step 2)

- Features: TF-IDF over word unigrams and bigrams with sublinear term frequency.
- Classifier: linear SVM (`LinearSVC`). Logistic regression was tried first but was too slow for a cross-validation grid; a linear SVM is the standard fast baseline for text and trains in about a second.
- Regularisation `C` is chosen from {0.1, 0.3, 1, 3} by macro-F1 under cross-validation on the training set.

### Planned

- **Sentence embeddings + linear classifier:** a pretrained sentence encoder turns each query into a vector, with a simple classifier on top.
- **Fine-tuned DistilBERT:** PyTorch and Hugging Face Transformers, trained on a free Colab GPU, with a short maximum sequence length because queries are short.
- **Error analysis:** per-class scores, the most frequently confused intent pairs, and a manual look at misclassified queries.
- **Learning curve:** performance at 10%, 25%, 50% and 100% of the training data.
- **Robustness:** performance on test queries with random character-level typos.

## Results

| Model | Test accuracy | Test macro-F1 | Notes |
|-------|---------------|---------------|-------|
| TF-IDF + linear SVM | 0.894 | 0.894 | C=3; 5-fold CV macro-F1 0.879 |
| Sentence embeddings + linear classifier | | | |
| Fine-tuned DistilBERT | | | |

### Hypothesis outcomes

_To be completed after the experiments: each hypothesis marked confirmed, partly confirmed or refuted, with the evidence._

**H3 (baseline accuracy): partly confirmed**. Test accuracy was 0.894, inside the predicted 0.85 to 0.92 range. The claim that macro-F1 would be slightly below accuracy was wrong: they are essentially equal (0.8937 vs 0.8936). The test set is balanced, so the two metrics converge; the hypothesis overlooked that the imbalance exists only in the training data.

## Project structure

```
.
├── data/
│   └── raw/               # Banking77 CSVs (gitignored)
├── results/               # metrics, per-class scores, predictions
├── data.py                # loads Banking77 (local CSVs, else Hugging Face Hub)
├── explore.py             # dataset statistics and leakage / quality checks
├── baseline.py            # TF-IDF + linear SVM, cross-validated, evaluated once on test
├── requirements.txt
└── README.md
```

## Setup and usage

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
source .venv/Scripts/activate      # Windows, Git Bash
pip install -r requirements.txt
```

### Getting the data

`data.py` first looks for local CSVs in `data/raw/` and otherwise downloads the dataset from the Hugging Face Hub. If you are offline or the Hub is unreachable, download the original files:

```bash
mkdir -p data/raw
curl -L -o data/raw/train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
curl -L -o data/raw/test.csv  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
```

### Running

```bash
python explore.py      # dataset statistics and data-quality checks
python baseline.py     # baseline: cross-validation, then one test evaluation
```

Outputs are written to `results/`: `baseline_metrics.json`, `baseline_per_class.csv` and `baseline_predictions.csv`.

## Limitations so far

- Only exact duplicates were checked; near-duplicates between train and test may remain.
- Banking77 queries are short, single-intent and relatively clean, so results may not carry over to long or noisy text such as emails or complaint narratives.
- Removing 6 overlapping queries from the test set makes scores slightly incomparable with published results.
- Results will come from a single fixed seed unless stated otherwise, so small differences between models should not be over-interpreted.
