import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load_banking77

SEED = 42
RESULTS = Path("results")
C_GRID = [0.1, 0.3, 1, 3]


def _key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.strip()


def clean(train: pd.DataFrame, test: pd.DataFrame):
    """Drop duplicate train queries, and test queries that also appear in train."""
    train = train.assign(_k=_key(train["text"])).drop_duplicates("_k")
    test = test.assign(_k=_key(test["text"]))
    n_before = len(test)
    test = test[~test["_k"].isin(set(train["_k"]))]
    print(f"Cleaning: train -> {len(train)} rows, test {n_before} -> {len(test)} rows")
    return (train.drop(columns="_k").reset_index(drop=True),
            test.drop(columns="_k").reset_index(drop=True))


def make_model(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LinearSVC(C=c)),
    ])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    train, test, names = load_banking77()
    train, test = clean(train, test)

    # 1) choose C using cross-validation on TRAIN only: the test set is not touched here
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for c in C_GRID:
        t0 = time.time()
        scores = cross_val_score(make_model(c), train["text"], train["label"],
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_scores[c] = float(scores.mean())
        print(f"C={c:<4} CV macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})  [{time.time()-t0:.0f}s]")
    best_c = max(cv_scores, key=cv_scores.get)
    print(f"Best C by CV: {best_c}")

    # 2) fit once on all training data, evaluate once on the held-out test set
    model = make_model(best_c).fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    acc = accuracy_score(test["label"], pred)
    macro_f1 = f1_score(test["label"], pred, average="macro")
    print(f"\nTEST  accuracy = {acc:.4f}   macro-F1 = {macro_f1:.4f}")

    # 3) save everything needed for later steps (error analysis, comparison table)
    json.dump({"model": "tfidf_linearsvc", "best_C": best_c, "cv_macro_f1": cv_scores,
               "test_accuracy": acc, "test_macro_f1": macro_f1,
               "n_train": len(train), "n_test": len(test)},
              open(RESULTS / "baseline_metrics.json", "w"), indent=2)
    report = classification_report(test["label"], pred, labels=range(len(names)),
                                   target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS / "baseline_per_class.csv")
    test.assign(pred=pred, pred_name=[names[i] for i in pred]).to_csv(
        RESULTS / "baseline_predictions.csv", index=False)


if __name__ == "__main__":
    main()