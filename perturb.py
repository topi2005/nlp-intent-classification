import random


def _typo_word(word: str, rng: random.Random) -> str:
    """One random character edit inside a word of 4+ letters (first and last letter untouched)."""
    i = rng.randrange(1, len(word) - 2)
    op = rng.choice(["swap", "delete", "duplicate"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def add_typos(text: str, rate: float, seed: int) -> str:
    """Give each purely alphabetic word of 4+ letters a typo with probability `rate`."""
    rng = random.Random(seed)
    words = []
    for w in text.split(" "):
        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            w = _typo_word(w, rng)
        words.append(w)
    return " ".join(words)


def perturb_texts(texts, rate: float, seed: int = 42) -> list[str]:
    """Deterministic: the same (texts, rate, seed) always yields the same noisy queries,
    so every model is evaluated on exactly the same perturbed test set."""
    return [add_typos(t, rate, seed + i) for i, t in enumerate(texts)]
