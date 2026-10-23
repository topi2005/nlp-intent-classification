from data import load_banking77


def norm(s):
    return s.str.lower().str.strip()


def main() -> None:
    train, test, names = load_banking77()
    print(f"train={len(train)}  test={len(test)}  classes={len(names)}")

    counts = train["label_name"].value_counts()
    print(f"\nExamples per class (train): min={counts.min()}  median={counts.median():.0f}  max={counts.max()}")
    print("Smallest classes:\n", counts.tail(5).to_string())

    words = train["text"].str.split().str.len()
    print("\nWords per query:\n", words.describe().round(1).to_string())

    # --- data quality / leakage checks ---
    tr, te = norm(train["text"]), norm(test["text"])
    print(f"\nDuplicate queries inside train: {tr.duplicated().sum()}")
    overlap = set(tr) & set(te)
    print(f"Queries appearing in BOTH train and test: {len(overlap)}")

    labels_per_text = train.assign(t=tr).groupby("t")["label"].nunique()
    conflicts = labels_per_text[labels_per_text > 1]
    print(f"Identical queries with conflicting labels in train: {len(conflicts)}")
    for t in conflicts.index[:5]:
        print("  ", repr(t), "->", sorted(train.loc[tr == t, "label_name"].unique()))


if __name__ == "__main__":
    main()