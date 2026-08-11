# Reproducing the paper's results

This file maps the released artifacts to the paper's reported numbers and
describes calibrated inference in enough detail to interpret them. The paper
itself is the specification of record for the mechanism, the constants, and the
data each constant was selected on.

## Which mode reproduces which numbers

- **Base UNILID rows**: load with `calibrated=False` (or use the version-1
  polybox file). `eval.py` always scores in base mode and refuses a version-2
  (calibrated) model file unless `--base` is passed, so base numbers cannot be
  produced from a calibrated file by accident.
- **Calibrated UNILID rows**: the default (`load_model(path)` on the version-2
  file from the HuggingFace Hub repository).
- Evaluation conventions (the macro F1 and macro FPR definitions, and the
  scored-pool convention for the GlotLID-C test data) are stated in the paper's
  experimental-setup and results sections; the numbers below follow them.

Release verification: on a 250,000-line golden subset of the GlotLID-C test
pool, the packaged base path reproduces the paper pipeline's recorded base
predictions on 250,000/250,000 lines, and the packaged calibrated path
reproduces the recorded calibrated predictions on 250,000/250,000 lines.

## What calibrated inference computes

UNILID scores a text under every language and predicts the language with the
highest score. Calibrated UNILID keeps this decision rule and adds two
corrections, derived in the paper from an error analysis of the base model.

**1. A shared constant for unseen tokens.** Each language's model assigns a
log-probability to every token in the shared vocabulary, including tokens that
never appeared in that language's training data. Training never leaves a token
at probability zero: every token receives at least a minimum probability of
10^-12, and each language's probabilities are then normalized to sum to one. A
side effect of that normalization is that the probability an unseen token ends
up with differs from language to language, depending on how much training data
the language has. Prediction compares scores across languages, so these
differing unseen-token values act as a per-language offset added to every
unseen token in the text: a text containing tokens unseen by two candidate
languages is pushed toward the candidate whose unseen-token value happens to be
higher, for no linguistic reason. The correction removes the offset. At load
time, every unseen-token log-probability that lies above the shared constant
c = -21 (in natural log units) is lowered to exactly c. Values already at or
below c stay as trained, and the distributions are not renormalized afterwards.

**2. Re-examining close decisions that land in two groups of languages.** The
margin of a prediction is the best language's score minus the second-best
language's score; a small margin means the decision was close. The base model's
errors concentrate in predictions INTO two groups: languages with fewer than
18,000 training samples, whose estimated distributions stay close to their
uniform initialization and therefore give moderate probability to text from
many languages, and a group of four larger languages whose distributions are
unusually flat for their script (Scots, Banjar, Aragonese, West Flemish; the
identification criteria are in the paper). When the predicted language belongs
to either group and the margin falls below that language's own threshold, the
prediction is re-examined: it moves to the highest-ranked of the candidates
ranked 2 to 5 that has at least 100,000 training samples and a score within 21
natural-log units of the best score. If no candidate qualifies, the prediction
stays unchanged. Each threshold is a percentile of the margins that language's
own training lines produce, so a threshold can be computed for a new language
without touching any other language. 26 of the 1,080 languages in the first
group had fewer than 200 usable calibration lines; they receive no threshold
and are never re-examined.

All constants live in the calibration artifact bundled with the model (also
downloadable as `calibration.json`), not in the code.

## Measured effect of calibration

| Evaluation | Base | Calibrated |
|------------|------|------------|
| GlotLID-C test pool (45.4M lines, 1,940 languages), macro F1 | 0.929 | 0.957 |
| UDHR (parallel, near-equal per-language sample counts), macro F1 | 0.859 | 0.838 |
| CommonLID (out-of-domain web text, 109 labels), macrolanguage-aware accuracy | 0.845 | 0.860 |
| CommonLID, tag-level macro F1 | 0.723 | 0.715 |

On UDHR, re-examination also moves some correct low-margin predictions, which
lowers macro F1 on data where every language has similar sample counts. On
CommonLID (macrolanguage-aware accuracy counts a prediction as correct when it
matches the label at the language or macrolanguage level), calibration lowers
the number of lines predicted as languages outside the 109-label set from
32,901 to 25,884, which raises accuracy, while re-examination moves some
correct low-margin lines, which lowers tag-level macro F1. Together the three
results locate where the gains appear: test data whose per-language line counts
follow a collection's natural imbalance, over a label set that includes
under-resourced languages. Use `calibrated=False` where the base behavior is
wanted.

## Version 0.2.0 migration note

Calibrated inference is the default from version 0.2.0. Loading a model without
a calibration artifact (any version-1 `.unilid` file, including the original
polybox release and self-trained models) with default arguments raises
`UnilidCalibrationError`; pass `calibrated=False` or use the version-2
calibrated release file. Results published for the base model are reproduced
with `calibrated=False`.
