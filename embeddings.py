import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from baseline import clean
from data import load_banking77
from report import save_results, train_val_split

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
C_GRID = [1, 10, 100]


def main() -> None:
    train, test, names = load_banking77()
    train, test = clean(train, test)
    tr, val = train_val_split(train)

    encoder = SentenceTransformer(ENCODER)

    def embed(texts):
        return encoder.encode(list(texts), batch_size=128, normalize_embeddings=True,
                              show_progress_bar=False)

    X_tr, X_val, X_test = embed(tr["text"]), embed(val["text"]), embed(test["text"])

    # choose C on the validation split; the test set is not touched here
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000).fit(X_tr, tr["label"])
        f1 = f1_score(val["label"], clf.predict(X_val), average="macro")
        print(f"C={c:<4} validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    # refit on train + validation, then evaluate once on the test set
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(
        np.vstack([X_tr, X_val]), np.concatenate([tr["label"], val["label"]]))
    pred = clf.predict(X_test)
    save_results("embeddings", test, pred, names,
                 {"encoder": ENCODER, "best_C": best_c, "val_macro_f1": best_f1})


if __name__ == "__main__":
    main()
