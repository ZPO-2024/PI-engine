# PI-engine

Predictive Invariant Engine (PI-engine) is an experimental, theory-neutral framework for observational/predictive analysis across structurally different systems.

The v0.1 goal is to normalize cases into explicit, inspectable state-space and graph representations; select applicable explicit models; simulate deterministic and stochastic trajectory ensembles; evaluate convergence, divergence, sensitivity, closure signatures, calibration, and residuals; and compare predictions against held-out outcomes.

## v0.1 principles

- Observational/predictive only.
- Explicit, inspectable models with provenance and falsifiers.
- Hybrid state-space + graph representation.
- Temporal holdout testing and calibration.
- Synthetic ground-truth systems and negative controls.
- Competing models remain separate until evidence warrants weighting or ensemble use.
- Epistemic closure and causal closure are distinct.
- Residuals are first-class data, not discarded noise.
- Hypothetical intervention support is architecturally anticipated but inactive.
- Invariant promotion, learned routing, autonomous control, and metaphysical claims are out of scope for v0.1 core.

## Key documents

- `docs/superpowers/specs/2026-08-08-pi-engine-v0.1-design.md` — approved v0.1 design.
- `docs/superpowers/plans/2026-08-08-pi-engine-v0.1.md` — Codex implementation plan.
- `docs/terminology-map.md` — translation of project language into state-estimation, MPC/SMPC, and learned-index terminology.
- `docs/calibration-and-testing.md` — evaluation architecture.
- `docs/research-boundaries.md` — epistemic and scope boundaries.

## Intended implementation stack

Initial implementation should remain lightweight and inspectable: Python, Pydantic/JSON Schema, NumPy/SciPy, NetworkX, pandas, pytest, and Hypothesis. Avoid adding ML frameworks until deterministic and similarity-based routing are demonstrably insufficient.

## Command line

`pi-engine` (or `python -m pi_engine.cli`) runs synthetic cases and negative controls end to end:

```
pi-engine list
pi-engine run linear_convergence --horizon 3 --artifact run.json
pi-engine report --artifact run.json
pi-engine reveal --artifact run.json
```

`run` is the only command that simulates. It saves a signed run artifact (a
checksummed JSON envelope) rather than printing a report. `report` and
`reveal` always replay that saved artifact — they never re-simulate — so a
rendered report reflects exactly what was run, and a tampered artifact file
fails a checksum check (`run artifact integrity check failed`) instead of
silently rendering different numbers.

## Known limitations (v0.1)

- Actual implementation needed only NumPy and Pydantic; SciPy, NetworkX, and
  pandas from the intended stack above were never required at this scope.
  Left as a forward-looking list, not a claim of current use.
- No Hypothesis/property-based tests exist yet, despite Hypothesis being
  named in the intended stack. All 524 tests are example-based. Property
  tests for the schema and numeric-analysis layers are the highest-value
  addition before relying on this for anything beyond the synthetic
  fixtures.
- `report`/`reveal` require a saved run artifact; there is no longer a
  "simulate and print a report in one step" path. This is a deliberate
  provenance guarantee (see Command line, above), not an oversight.
