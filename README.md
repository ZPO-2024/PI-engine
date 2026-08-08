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
