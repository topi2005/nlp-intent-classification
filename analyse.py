import sys

import pandas as pd

from data import load_banking77


def main(prefix: str = "baseline", k: int = 10) -> None:
    train, _, names = load_banking77()
    preds = pd.read_csv(f"results/{prefix}_predictions.csv")
    per_class = pd.read_csv(f"results/{prefix}_per_class.csv", index_col=0).loc[names]
    per_class["n_train"] = train["label_name"].value_counts().reindex(per_class.index)

    print(f"== {prefix}: lowest {k} intents by F1 ==")
    cols = ["precision", "recall", "f1-score", "n_train"]
    print(per_class.sort_values("f1-score").head(k)[cols].round(3).to_string())

    # H2: do small classes get lower recall?
    small = per_class[per_class["n_train"] < 60]
    rest = per_class[per_class["n_train"] >= 60]
    print(f"\nMean recall, classes with <60 train examples (n={len(small)}): {small['recall'].mean():.3f}"
          f"  | others (n={len(rest)}): {rest['recall'].mean():.3f}")
    rho = per_class["n_train"].corr(per_class["recall"], method="spearman")
    print(f"Spearman correlation, train size vs recall: {rho:.3f}")

    # H1: are the weakest intents card-related, compared with the base rate?
    low = per_class.sort_values("f1-score").head(k).index
    print(f"\nLowest-{k} intents with 'card' in the name: {sum('card' in n for n in low)}/{k}"
          f"  (base rate: {sum('card' in n for n in names)}/{len(names)} of all intents)")

    wrong = preds[preds["label_name"] != preds["pred_name"]]
    pairs = wrong.groupby(["label_name", "pred_name"]).size().sort_values(ascending=False).head(k)
    print(f"\n== Most frequent confusions (true -> predicted), {len(wrong)} errors in total ==")
    print(pairs.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")