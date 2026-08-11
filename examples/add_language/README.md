# Worked example: add a language to a calibrated model

This example builds a small calibrated model from scratch and then adds a
fourth language to it with `unilid-add-language`, end to end, on toy data that
ships with the repository. It needs only the installed package and the built
tokenizers extension (no model download, no SentencePiece build: every training
step uses the pure-Python EM path).

Run it:

```bash
bash examples/add_language/run_example.sh
```

Total runtime is a few minutes; everything is written under
`examples/add_language/work/` (not committed).

## What the steps do

1. **`make_data.py`** generates four constructed toy languages, each defined by
   its own syllable inventory (`data/`; deterministic, the committed files are
   reproduced byte-identically). `ddd_Latn`, the language added later, shares
   two syllables with `ccc_Latn`, so the addition is not trivially separable.
   Line counts: `aaa_Latn` 1,000, `bbb_Latn` 1,000, `ccc_Latn` 300 for the base
   model, `ddd_Latn` 250 for the addition, plus 50 held-out lines each.
2. **`train.py`** trains the base model on the three base languages
   (vocabulary 300, byte-level, per-language probabilities via the pure-Python
   soft-EM method), and **`convert.py`** packs it into a version-1 `.unilid`
   file.
3. **`build_calibration.py`** builds the calibration and bundles it into a
   version-2 file. The constants are part of the artifact: this toy deployment
   states its own corpus-size requirements (`head_n=500`,
   `replacement_min_n=800`) while keeping the released mechanism's other
   constants; the released 1,940-language model uses the paper's values
   (18,000 and 100,000). With these constants, `ccc_Latn` (300 lines) is the
   one re-examined language, and its threshold is estimated from its own
   training lines exactly as in the released calibration. The script also
   shows the bootstrap the incremental command uses internally: thresholds are
   defined on the calibrated (clamped) model, so a placeholder-excluded
   artifact is bundled first, the real thresholds are estimated against it,
   and the final artifact replaces it.
4. **`unilid-add-language`** (here invoked as `python -m unilid.add_language`)
   trains `ddd_Latn` over the existing vocabulary from its own 250 lines,
   appends its weight row (existing rows are copied byte-identically), and
   calibrates it: N=250 is below `head_n=500`, so a threshold is estimated;
   N=250 is below `replacement_min_n=800`, so it does not become a replacement
   candidate. Both facts are printed.
5. **`predict_demo.py`** compares predictions on `ddd_Latn`'s 50 held-out
   lines. Before the addition the model labels them with the three base
   languages (accuracy 0.00 by construction); after the addition the measured
   accuracy is 0.98, with the three base languages still at 1.00 on their own
   held-out lines. The script also prints the re-examination statistics of the
   final batch, where the gate is visibly active on this toy model.

## Two observed behaviors worth knowing about

- **The unseen-token constant can be a no-op for a soft-EM-trained toy model.**
  The soft-EM trainer leaves unseen tokens at the training floor
  (log 1e-12 = -27.63), which is below c = -21, so the one-sided rule leaves
  such rows unchanged, and `add_language` prints exactly that. The released
  model's rows all have unseen-token values above c (a byproduct of its
  training pipeline and data scale) and all 1,940 are lowered to c at load.
- **The two training methods differ at toy data sizes.** Re-running step 4 with
  `--method sp` (the SentencePiece path, used for the released model's
  per-language training; needs the built `spm_train` binary) also completes and
  calibrates, but on 250 toy lines it estimates a flatter distribution:
  held-out accuracy 0.60 against 0.98 for the EM path, with 186 of 250
  calibration lines own-won against 250 of 250. The release-scale evidence for
  the `sp` path comes from corpora with thousands to 100,000 lines per
  language; at toy sizes prefer `--method em`.
