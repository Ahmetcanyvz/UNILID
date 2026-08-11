"""Build a calibration for the toy base model and bundle it (worked example,
step 3). Demonstrates that the calibration constants are part of the artifact:
the released 1,940-language model uses the paper's constants (head_n=18,000,
replacement_min_n=100,000); this toy deployment states its own, scaled to the
toy corpus sizes. The mechanism (unseen-token constant, per-language
thresholds from own training margins, replacement walk) is identical.
"""
import gc
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from unilid.calibration import (  # noqa: E402
    CAUSE_LOW_CALIBRATION,
    Calibration,
    TauRow,
    estimate_tau,
)
from unilid.model_io import UnilidModel, load_unilid_raw, write_unilid  # noqa: E402

# The toy deployment's constants. unseen_token_constant, proximity_bound,
# margin_q, and the sampling constants keep the released mechanism's values;
# the two corpus-size requirements are scaled to the toy corpus (the released
# artifact uses 18,000 and 100,000; see the repository README's caveats on
# choosing replacement_min_n for a corpus without the released data cap).
CONSTANTS = {
    "unseen_token_constant": -21.0,
    "head_n": 500,            # languages below 500 training lines are re-examined
    "replacement_min_n": 800, # languages with >= 800 lines can receive reassignments
    "proximity_bound": 21.0,
    "topk": 5,
    "margin_q": 5.0,
    "group_b_percentile": 5.0,
    "calib_max": 2000,
    "min_calib_lines": 50,
    "calib_seed": 0,
}


def main():
    model_path = HERE / "work" / "toy_base.unilid"
    out_model = HERE / "work" / "toy_calibrated.unilid"
    out_json = HERE / "work" / "calibration.json"
    data_dir = HERE / "data"

    base_tok_bytes, weights, langs = load_unilid_raw(model_path)
    counts = {}
    for lang in langs:
        lines = (data_dir / f"{lang}_train.txt").read_text().splitlines()
        counts[lang] = sum(1 for l in lines if l)
    print("training-line counts:", counts)

    group_a_langs = [l for l in langs if counts[l] < CONSTANTS["head_n"]]
    print(f"group A (fewer than {CONSTANTS['head_n']} training lines):",
          group_a_langs)

    # Thresholds need a scoring pass on the calibrated (clamped) model, which
    # needs a calibration to load. Bootstrap: bundle with placeholder rows
    # (excluded, zero lines processed), load calibrated, estimate each
    # threshold from the language's own training lines, then write the final
    # artifact. unilid.add_language does the same internally for one language.
    placeholder = TauRow(tau=float("-inf"), excluded=True,
                         cause=CAUSE_LOW_CALIBRATION, n_scoreable=0,
                         n_self_won=0)
    provisional = Calibration(
        group_a={l: placeholder for l in group_a_langs}, group_b={},
        train_counts=counts, provenance={"derived": "worked example bootstrap"},
        **CONSTANTS)
    tmp = out_model.with_name("toy_calibrated.tmp.unilid")
    weights_arr = weights.copy()
    del weights
    write_unilid(tmp, base_tok_bytes, langs, weights_arr, provisional)

    model = UnilidModel(tmp, calibrated=True)
    rows = {}
    for lang in group_a_langs:
        lines = [l for l in
                 (data_dir / f"{lang}_train.txt").read_text().splitlines() if l]
        rows[lang] = estimate_tau(model, lang, lines, counts[lang])
        print(f"{lang}: tau={rows[lang].tau:.4f} excluded={rows[lang].excluded} "
              f"({rows[lang].n_self_won}/{rows[lang].n_scoreable} own-won)")
    del model
    gc.collect()
    tmp.unlink()

    final = Calibration(
        group_a=rows, group_b={}, train_counts=counts,
        provenance={"derived": "worked example (examples/add_language), toy "
                               "constants stated in build_calibration.py"},
        **CONSTANTS)
    final.runtime_for(langs)
    final.to_json_file(out_json)
    write_unilid(out_model, base_tok_bytes, langs, weights_arr, final)
    print(f"wrote {out_json} and {out_model} (version 2)")


if __name__ == "__main__":
    main()
