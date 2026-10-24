"""Step 5 experiments (run on Colab with a GPU):
  H5  learning curve: all three models trained on 10%, 25%, 50% and 100% of the training split
  H6  typo robustness: the 100% models evaluated on test queries with increasing typo rates
Every model is evaluated on exactly the same test queries (clean and perturbed)."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import clean
from data import load_banking77
from embeddings import ENCODER
from finetune import BATCH, MODEL, fit, make_loader, predict, set_seed
from perturb import perturb_texts
from report import RESULTS, SEED, train_val_split

FRACTIONS = [0.1, 0.25, 0.5, 1.0]
TYPO_RATES = [0.0, 0.1, 0.25, 0.5]
TARGET_STEPS = 1100      # same optimiser-step budget at every fraction (the 100% run is 4 epochs)
BASELINE_C, EMBEDDING_C = 3, 100   # fixed from the earlier experiments, not re-tuned per fraction


def subsample_idx(labels: pd.Series, frac: float) -> np.ndarray:
    positions = np.arange(len(labels))
    if frac >= 1.0:
        return positions
    idx, _ = train_test_split(positions, train_size=frac, stratify=labels, random_state=SEED)
    return np.sort(idx)


def score(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


def main() -> None:
    set_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train, test, names = load_banking77()
    train, test = clean(train, test)
    tr, val = train_val_split(train)
    y_test = test["label"].to_numpy()

    # identical clean and typo-perturbed test sets for every model
    test_texts = {r: (list(test["text"]) if r == 0 else perturb_texts(test["text"], r, SEED))
                  for r in TYPO_RATES}
    print("example query:", test_texts[0.0][0], "->", test_texts[0.25][0], "(typo rate 0.25)")

    encoder = SentenceTransformer(ENCODER)

    def embed(texts):
        return encoder.encode(list(texts), batch_size=128, normalize_embeddings=True,
                              show_progress_bar=False)

    X_tr_all = embed(tr["text"])
    X_test = {r: embed(t) for r, t in test_texts.items()}
    tok = AutoTokenizer.from_pretrained(MODEL)
    val_dl = make_loader(tok, val["text"], val["label"], shuffle=False)

    curve, typo = [], []

    def record(model_name, frac, n_train, extra, preds_by_rate):
        for rate, pred in preds_by_rate.items():
            s = score(y_test, pred)
            if rate == 0:
                curve.append({"model": model_name, "fraction": frac, "n_train": n_train, **extra, **s})
            if frac == 1.0:
                typo.append({"model": model_name, "typo_rate": rate, **s})

    for frac in FRACTIONS:
        idx = subsample_idx(tr["label"], frac)
        sub = tr.iloc[idx].reset_index(drop=True)
        rates = TYPO_RATES if frac == 1.0 else [0.0]
        print(f"\n===== fraction {frac}: {len(sub)} training queries =====")

        # 1) TF-IDF + linear SVM
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                         ("clf", LinearSVC(C=BASELINE_C))]).fit(sub["text"], sub["label"])
        record("baseline", frac, len(sub), {}, {r: pipe.predict(test_texts[r]) for r in rates})

        # 2) frozen sentence embeddings + logistic regression
        clf = LogisticRegression(C=EMBEDDING_C, max_iter=2000).fit(X_tr_all[idx], sub["label"])
        record("embeddings", frac, len(sub), {}, {r: clf.predict(X_test[r]) for r in rates})

        # 3) fine-tuned DistilBERT with a fixed step budget
        steps_per_epoch = math.ceil(len(sub) / BATCH)
        epochs = math.ceil(TARGET_STEPS / steps_per_epoch)
        set_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=len(names)).to(device)
        train_dl = make_loader(tok, sub["text"], sub["label"], shuffle=True)
        best_epoch, best_f1 = fit(model, train_dl, val_dl, val["label"], device, epochs)
        preds = {r: predict(model, make_loader(tok, test_texts[r], y_test, shuffle=False), device)
                 for r in rates}
        record("distilbert", frac, len(sub), {"epochs": epochs, "best_epoch": best_epoch}, preds)
        del model
        torch.cuda.empty_cache()

        pd.DataFrame(curve).to_csv(RESULTS / "learning_curve.csv", index=False)   # save as we go
        pd.DataFrame(typo).to_csv(RESULTS / "typo_robustness.csv", index=False)

    curve_df, typo_df = pd.DataFrame(curve), pd.DataFrame(typo)

    # ---------- H5 ----------
    acc = curve_df.pivot(index="fraction", columns="model", values="accuracy")
    print("\n== Learning curve: test accuracy ==")
    print(acc.round(4).to_string())
    gap = acc["embeddings"] - acc["baseline"]
    print("\nEmbeddings minus baseline (accuracy):")
    print(gap.round(4).to_string())

    # ---------- H6 ----------
    tacc = typo_df.pivot(index="typo_rate", columns="model", values="accuracy")
    print("\n== Typo robustness: test accuracy ==")
    print(tacc.round(4).to_string())
    print("\nAccuracy lost relative to clean queries:")
    print((tacc.loc[0.0] - tacc).round(4).to_string())

    # ---------- plots ----------
    for df, x, xlabel, fname, logx in [
        (curve_df, "n_train", "training queries", "learning_curve.png", True),
        (typo_df, "typo_rate", "share of words with a typo", "typo_robustness.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, g in df.groupby("model"):
            ax.plot(g[x], g["accuracy"], marker="o", label=m)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=150)
        plt.close(fig)
    print("\nSaved results/learning_curve.csv, typo_robustness.csv and both plots.")


if __name__ == "__main__":
    main()
