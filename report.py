import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

RESULTS = Path("results")
SEED = 42


def train_val_split(train: pd.DataFrame, val_size: float = 0.1):
    """Stratified train/validation split, identical for every model that uses it."""
    tr, val = train_test_split(train, test_size=val_size, stratify=train["label"],
                               random_state=SEED)
    return tr.reset_index(drop=True), val.reset_index(drop=True)


def save_results(prefix: str, test: pd.DataFrame, pred, names: list[str], extra: dict | None = None):
    """Write metrics, per-class scores and predictions in the same format as the baseline,
    so analyse.py works unchanged for every model."""
    RESULTS.mkdir(exist_ok=True)
    acc = accuracy_score(test["label"], pred)
    macro_f1 = f1_score(test["label"], pred, average="macro")
    metrics = {"model": prefix, "test_accuracy": acc, "test_macro_f1": macro_f1,
               "n_test": len(test), **(extra or {})}
    with open(RESULTS / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(test["label"], pred, labels=range(len(names)),
                                   target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS / f"{prefix}_per_class.csv")
    test.assign(pred=pred, pred_name=[names[int(i)] for i in pred]).to_csv(
        RESULTS / f"{prefix}_predictions.csv", index=False)
    print(f"[{prefix}] TEST accuracy = {acc:.4f}   macro-F1 = {macro_f1:.4f}")
    return metrics
