"""Generate the toy corpora for the add-language worked example.

Four constructed toy languages, each defined by its own syllable inventory
(ddd_Latn shares two syllables with ccc_Latn so the added language is not
trivially separable). Deterministic (fixed seed): re-running reproduces the
committed data/ files byte-identically.

Line counts are chosen against the example's calibration constants
(head_n=500, replacement_min_n=800): aaa and bbb are replacement candidates,
ccc is re-examined (group A), and the added ddd joins group A.
"""
import random
from pathlib import Path

SYLLABLES = {
    "aaa_Latn": ["ba", "ke", "ti", "mo", "ru", "san", "pel", "dor"],
    "bbb_Latn": ["zu", "gri", "vol", "ne", "sha", "kom", "tir", "bek"],
    "ccc_Latn": ["fi", "lo", "wen", "dar", "mu", "hes", "tal", "gor"],
    "ddd_Latn": ["fi", "lo", "pra", "kli", "ven", "dus", "mar", "tol"],
}
TRAIN_LINES = {"aaa_Latn": 1000, "bbb_Latn": 1000, "ccc_Latn": 300,
               "ddd_Latn": 250}
TEST_LINES = 50
SEED = 12345


def _sentence(rng: random.Random, syllables: list) -> str:
    words = []
    for _ in range(rng.randint(4, 9)):
        n_syl = rng.randint(1, 3)
        words.append("".join(rng.choice(syllables) for _ in range(n_syl)))
    return " ".join(words) + "."


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    for lang in sorted(SYLLABLES):
        syl = SYLLABLES[lang]
        train = [_sentence(rng, syl) for _ in range(TRAIN_LINES[lang])]
        test = [_sentence(rng, syl) for _ in range(TEST_LINES)]
        (out_dir / f"{lang}_train.txt").write_text("\n".join(train) + "\n")
        (out_dir / f"{lang}_test.txt").write_text("\n".join(test) + "\n")
        print(f"{lang}: {len(train)} train, {len(test)} test lines")

    # fastText-format training file for train.py, over the three base languages
    base_langs = ["aaa_Latn", "bbb_Latn", "ccc_Latn"]
    work = Path(__file__).parent / "work"
    work.mkdir(exist_ok=True)
    with open(work / "train_base.txt", "w") as f:
        for lang in base_langs:
            for line in (out_dir / f"{lang}_train.txt").read_text().splitlines():
                f.write(f"__label__{lang} {line}\n")
    print(f"wrote {work / 'train_base.txt'}")


if __name__ == "__main__":
    main()
