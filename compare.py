import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data import load_banking77

MODELS = ["baseline", "embeddings", "distilbert"]
IDENTITY_TRIO = {"verify_my_identity", "why_verify_identity", "unable_to_verify_identity"}
RNG = np.random.default_rng(42)


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 2000):
    """Mean accuracy difference (b - a) with a 95% paired bootstrap interval."""
    diff = correct_b.astype(float) - correct_a.astype(float)
    idx = RNG.integers(0, len(diff), size=(n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return diff.mean(), lo, hi


def main() -> None:
    _, _, names = load_banking77()
    preds = {m: pd.read_csv(f"results/{m}_predictions.csv") for m in MODELS}
    for m in MODELS[1:]:
        assert preds[m]["text"].equals(preds["baseline"]["text"]), "test rows differ between models"
    y = preds["baseline"]["label"].to_numpy()
    correct = {m: preds[m]["pred"].to_numpy() == y for m in MODELS}

    print(f"== Overall (test, n={len(y)}) ==")
    for m in MODELS:
        f1 = f1_score(y, preds[m]["pred"], average="macro")
        print(f"{m:<11} accuracy {correct[m].mean():.4f}   macro-F1 {f1:.4f}")

    print("\n== Paired accuracy differences (95% bootstrap interval) ==")
    for a, b in [("baseline", "embeddings"), ("baseline", "distilbert"), ("embeddings", "distilbert")]:
        d, lo, hi = paired_bootstrap(correct[a], correct[b])
        verdict = "clear" if lo > 0 or hi < 0 else "NOT distinguishable from zero"
        print(f"{b} - {a}: {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    f1 = {m: pd.read_csv(f"results/{m}_per_class.csv", index_col=0).loc[names, "f1-score"]
          for m in MODELS}
    weakest20 = set(f1["baseline"].sort_values().head(20).index)
    for m in MODELS[1:]:
        gain = (f1[m] - f1["baseline"]).sort_values(ascending=False)
        top = gain.head(10)
        print(f"\n== {m}: 10 largest per-intent F1 gains over baseline ==")
        print(top.round(3).to_string())
        print(f"In the baseline's 20 weakest intents: {sum(i in weakest20 for i in top.index)}/10")
        rho = f1["baseline"].corr(gain, method="spearman")
        print(f"Spearman(baseline F1, gain) = {rho:.3f}  (strongly negative = partly regression to the mean)")
        print("Largest drops:", ", ".join(f"{i} ({v:+.3f})" for i, v in gain.tail(3).items()))

    print("\n== Specific checks ==")
    for m in MODELS:
        pc = pd.read_csv(f"results/{m}_per_class.csv", index_col=0)
        p = pc.loc["card_not_working", "precision"]
        p_df = preds[m]
        trio_err = ((p_df["label_name"] != p_df["pred_name"])
                    & p_df["label_name"].isin(IDENTITY_TRIO) & p_df["pred_name"].isin(IDENTITY_TRIO)).sum()
        print(f"{m:<11} card_not_working precision {p:.3f}   identity-trio confusions {trio_err}")


if __name__ == "__main__":
    main()
