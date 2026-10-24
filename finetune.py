import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from baseline import clean
from data import load_banking77
from report import SEED, save_results, train_val_split

MODEL = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS, LR = 64, 32, 4, 5e-5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tok, texts, labels, shuffle: bool) -> DataLoader:
    items = list(zip(list(texts), [int(y) for y in labels]))

    def collate(batch):
        t, y = zip(*batch)
        enc = tok(list(t), padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc

    return DataLoader(items, batch_size=BATCH if shuffle else 128,
                      shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(model(**batch).logits.argmax(dim=-1).cpu())
    return torch.cat(out).numpy()


def fit(model, train_dl, val_dl, val_labels, device, epochs: int):
    """Train, keep the epoch with the best validation macro-F1, and load those weights
    back into the model. Returns (best_epoch, best_validation_macro_f1)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        val_f1 = f1_score(val_labels, predict(model, val_dl, device), average="macro")
        print(f"epoch {epoch}/{epochs}   last-batch loss {loss.item():.3f}   "
              f"validation macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_epoch, best_f1


def main() -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train, test, names = load_banking77()
    train, test = clean(train, test)
    tr, val = train_val_split(train)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(names)).to(device)

    train_dl = make_loader(tok, tr["text"], tr["label"], shuffle=True)
    val_dl = make_loader(tok, val["text"], val["label"], shuffle=False)
    test_dl = make_loader(tok, test["text"], test["label"], shuffle=False)

    best_epoch, best_f1 = fit(model, train_dl, val_dl, val["label"], device, EPOCHS)

    # the epoch that did best on validation is already loaded; evaluate once on the test set
    pred = predict(model, test_dl, device)
    save_results("distilbert", test, pred, names,
                 {"base_model": MODEL, "epochs": EPOCHS, "best_epoch": best_epoch,
                  "learning_rate": LR, "max_length": MAX_LEN, "batch_size": BATCH,
                  "val_macro_f1": best_f1})


if __name__ == "__main__":
    main()
