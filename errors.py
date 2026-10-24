import sys

import pandas as pd


def main(model: str = "distilbert", n_pairs: int = 8, per_pair: int = 5, seed: int = 42) -> None:
    df = pd.read_csv(f"results/{model}_predictions.csv")
    wrong = df[df["label_name"] != df["pred_name"]]
    print(f"{model}: {len(wrong)} errors out of {len(df)} test queries")

    top = (wrong.groupby(["label_name", "pred_name"]).size()
                .sort_values(ascending=False).head(n_pairs))
    rows = []
    for (true, pred), n in top.items():
        sample = wrong[(wrong["label_name"] == true) & (wrong["pred_name"] == pred)]
        sample = sample.sample(min(per_pair, n), random_state=seed)
        print(f"\n--- true: {true}  ->  predicted: {pred}   ({n} errors) ---")
        for text in sample["text"]:
            print("  *", text)
            rows.append({"model": model, "text": text, "true": true, "pred": pred, "category": ""})

    out = f"results/{model}_error_sample.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} queries to {out}. Fill in the 'category' column by hand.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "distilbert")
