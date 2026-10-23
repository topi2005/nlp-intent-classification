from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")


def _from_csv() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fallback: original PolyAI CSVs placed in data/raw/ (columns: text, category)."""
    def read(path):
        df = pd.read_csv(path)
        if set(df.columns) != {"text", "category"}:   # file has no header row
            df = pd.read_csv(path, header=None, names=["text", "category"])
        return df.rename(columns={"category": "label_name"})

    train, test = read(RAW_DIR / "train.csv"), read(RAW_DIR / "test.csv")
    names = sorted(train["label_name"].unique())
    to_id = {n: i for i, n in enumerate(names)}
    for df in (train, test):
        df["label"] = df["label_name"].map(to_id)
    return train[["text", "label", "label_name"]], test[["text", "label", "label_name"]], names


def _from_hub() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    frames = []
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        text_col = "text" if "text" in df.columns else "utterance"
        df = df.rename(columns={text_col: "text"})
        df["label_name"] = df["label"].map(lambda i: names[i])
        frames.append(df[["text", "label", "label_name"]])
    return frames[0], frames[1], names


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (train, test, label_names). Uses local CSVs if present, else the HF hub."""
    if (RAW_DIR / "train.csv").exists() and (RAW_DIR / "test.csv").exists():
        return _from_csv()
    return _from_hub()