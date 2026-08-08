# Calibration and Testing Architecture

PI-engine is evaluated on held-out outcomes, not on whether an explanation is aesthetically compelling.

## 1. Case-level verification

Before simulation, validate:
- source/provenance exists,
- observations trace to a source,
- units/ranges are valid,
- uncertainty is represented,
- graph references are internally valid,
- models declare applicability,
- post-cutoff outcomes are not present in pre-cutoff inputs.

Future leakage is a hard test failure.

## 2. Temporal holdout

For a time-ordered case:

`T0 - T1 - T2 - T3 - T4 - T5`

Expose only information through a cutoff, predict the future window, then reveal withheld outcomes. Repeat with multiple cutoffs when possible to measure how additional evidence changes calibration and observability.

## 3. Model-level calibration

Track separately:
- applicability confidence,
- parameter confidence,
- predictive calibration,
- structural confidence.

Model performance history should include cases tested, prediction error, calibration error, false convergence/divergence, closure errors, residual classes, and failure regimes.

## 4. Competing-model evaluation

Run applicable models independently before any ensemble weighting. Preserve disagreements. When outcomes are revealed, update performance evidence rather than deleting failed models.

A disagreement that resolves under a previously omitted variable should be treated as model refinement/discovery evidence.

## 5. Synthetic ground truth

Initial fixtures should include:
- linear convergence,
- oscillation,
- deterministic divergence,
- stochastic branching,
- coupled oscillators,
- feedback instability,
- hierarchical/nested dynamics,
- random/no-structure negative control.

## 6. Negative controls

Use random graphs, shuffled time series, correlation-without-coupling, irrelevant proximity, and systems lacking paired structures. If favored patterns appear reliably in controls, the detector/model is overfitting.

## 7. Probabilistic scoring

Use proper scoring rules where appropriate:
- Brier score for binary/categorical probabilistic forecasts,
- log score/NLL only when probability assumptions are explicit and valid,
- continuous deterministic error metrics for scalar/vector trajectories,
- interval coverage for forecast envelopes.

Overconfident errors must be penalized more heavily than low-confidence misses.

## 8. Residuals

Residuals are stored as data and provisionally classified as:
- noise-like,
- parameter mismatch,
- missing variable,
- model mismatch,
- regime change,
- coupling/topology error,
- phase/timing error,
- structured unknown.

Structured residuals should become candidates for later clustering and model/invariant discovery rather than being silently absorbed as noise.

## 9. Candidate invariant evaluation

Future invariant promotion should consider:
- cross-domain coverage,
- independence of source assumptions,
- held-out predictive gain,
- compression gain without predictive loss,
- falsification survival,
- ablation/necessity,
- scale persistence,
- structured-residual reduction.

Suggested future stages:

`OBSERVED PATTERN -> REPEATED PATTERN -> STRUCTURAL CANDIDATE -> PREDICTIVE CANDIDATE -> CROSS-DOMAIN INVARIANT -> CORE INVARIANT`

No automatic promotion is implemented in v0.1.
