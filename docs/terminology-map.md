# Terminology Translation Map

This document translates PI-engine working language into adjacent established terminology. The mapping is approximate and intended to reduce reinvention, not to force different domains into false equivalence.

| PI-engine working term | State estimation / controls | MPC / stochastic MPC | Learned-index / data terminology | PI-engine usage |
|---|---|---|---|---|
| Known baseline | state `x_t`, state estimate `x_hat_t` | initial condition | query/key context | State estimate |
| Raw observation | measurement `y_t` | measured output/disturbance | record/feature | Observation |
| Hidden variable | latent/unmeasured state | uncertain state/disturbance | latent distribution feature | Latent state |
| Watcher | observer/state estimator | estimator feeding prediction | predictor/router | Estimator |
| Glue/Web | interconnection/dynamics | prediction model/coupling | mapping/model structure | Coupling umbrella |
| Relationship | topology/interconnection | coupled dynamics | dependency/feature relation | Topology/coupling |
| Information | measurement/information state | belief/uncertainty state | indexed representation | Information |
| Phase | dynamic phase/state relationship | temporal dynamics | usually no direct analogue | Phase |
| Proximity | state-space distance / interaction range | neighborhood/reachability | key/model-space distance | Effective proximity |
| Geometry | configuration/state-space geometry | feasible-set geometry | data-distribution geometry | Geometry/topology |
| Energy/resource | physical state/input/resource | effort/cost where relevant | weak analogue | Domain-dependent resource variable |
| Timeline | reachable trajectory | scenario/predicted trajectory | candidate path | Trajectory/scenario |
| Branch | reachable path | scenario | candidate | Scenario/branch |
| Feedback | innovation + feedback | receding-horizon update | error correction/retraining | Feedback/update |
| Convergence | stability/attractor | trajectory convergence | clustering | Convergence |
| Divergence | instability/sensitivity | scenario spread | regime split/error growth | Divergence |
| Closure | reachable-set contraction / constraint | constraint activation / feasible-set contraction | narrowed candidate region | Typed closure |
| Premature closure | bad state/model assumption | overconstraint/model mismatch | wrong routing/model choice | Premature closure |
| Soft closure | uncertain/revisable constraint | soft/chance constraint | approximate bound | Provisional constraint |
| Hard closure | invariant/hard constraint | hard constraint | exact bound | Hard constraint |
| Puppet master/+1 | supervisor/coordinator | optimizer/controller | model router | Orchestrator when embodied |
| Nested eggs/octaves | hierarchical/multiscale systems | hierarchical/distributed MPC | recursive/multistage index | Multiscale hierarchy |
| Invariant | structural/conserved property | model property/constraint | stable feature | Typed candidate invariant |
| Singularity | critical/bifurcation state if demonstrated | feasibility/regime boundary | discontinuity/regime shift | Avoid as formal label until measured |

## Key architectural imports

### State estimation

Use the observer/state-estimator pattern when internal state cannot be measured directly. PI-engine must explicitly represent observability and allow `UNIDENTIFIABLE FROM AVAILABLE EVIDENCE` rather than forcing a single latent-state interpretation.

### MPC

Borrow the receding-horizon logic conceptually: estimate current state, predict forward, observe new evidence, update, and repeat. v0.1 does not apply control actions.

### Stochastic MPC

Borrow explicit treatment of parameter uncertainty, disturbances, scenario ensembles, and probabilistic constraints. Do not collapse uncertainty to one confidence scalar.

### Learned indexes

Borrow hierarchical routing and model-selection concepts, not necessarily learned implementation. v0.1 uses deterministic applicability filters and explicit similarity/ranking. Learned routing is deferred until enough cases exist for a fair benchmark.

## Important non-equivalences

- A symbolic or historical system does not automatically possess the observability or mathematical regularity of an engineered plant.
- Information gain is not the same as causal state-space contraction.
- A computational trajectory branch is not evidence of a literal alternate universe.
- A closure signature is not evidence of quantum wavefunction collapse.
- Cross-domain structural analogy is useful only when its predictive value survives held-out testing and negative controls.
