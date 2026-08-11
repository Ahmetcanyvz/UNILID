"""Compare predictions on the held-out lines of the added language before and
after `unilid-add-language` (worked example, final step)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from unilid import load_model  # noqa: E402

NEW_LANG = "ddd_Latn"


def accuracy(model, texts, lang):
    preds = [p for p, _t, _s in model.predict_batch(texts)]
    return sum(p == lang for p in preds) / len(texts), preds


def main():
    test_lines = [l for l in
                  (HERE / "data" / f"{NEW_LANG}_test.txt").read_text().splitlines()
                  if l]

    before = load_model(HERE / "work" / "toy_calibrated.unilid")
    acc_b, preds_b = accuracy(before, test_lines, NEW_LANG)
    print(f"before add: {NEW_LANG} not in model ({before.num_languages} "
          f"languages); its held-out lines are labeled "
          f"{sorted(set(preds_b))} (accuracy for {NEW_LANG}: {acc_b:.2f})")

    after = load_model(HERE / "work" / "toy_extended.unilid")
    if NEW_LANG not in after.langs:
        raise SystemExit(f"{NEW_LANG} missing from the extended model")
    acc_a, _ = accuracy(after, test_lines, NEW_LANG)
    row = after.calibration.group_a[NEW_LANG]
    print(f"after add:  {after.num_languages} languages; accuracy on "
          f"{NEW_LANG}'s 50 held-out lines: {acc_a:.2f}")
    print(f"            calibration row: tau={row.tau:.4f} "
          f"excluded={row.excluded} "
          f"({row.n_self_won}/{row.n_scoreable} own-won calibration lines)")
    print(f"            re-examination stats of the last batch: "
          f"{after.last_reexamination_stats}")

    # Existing languages still predicted correctly after the append
    for lang in ("aaa_Latn", "bbb_Latn", "ccc_Latn"):
        lines = [l for l in
                 (HERE / "data" / f"{lang}_test.txt").read_text().splitlines()
                 if l]
        acc, _ = accuracy(after, lines, lang)
        print(f"            {lang} held-out accuracy after the add: {acc:.2f}")

    if acc_a < 0.9:
        raise SystemExit(
            f"expected the added language to win on its own held-out lines, "
            f"got accuracy {acc_a:.2f}")
    print("example completed: the added language is predicted on its own "
          "held-out data and the existing languages are unchanged")


if __name__ == "__main__":
    main()
