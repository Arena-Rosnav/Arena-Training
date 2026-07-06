# Social-Dreamer: Structured Latent World Models for Robot Navigation in Pedestrian Environments

**Research Proposal — ICRA 2027**
Tuan Anh Roman Le · Arena Lab

---

## 1. Research Gap

Most social navigation papers report success rate against the exact pedestrian simulator whose
motion policy the RL policy was trained on. That number is not a measure of navigation quality —
it is a measure of simulator fit. Swap in a different behavior model and the rankings shift,
sometimes dramatically. The field has been comparing methods under conditions that make fair
comparison structurally impossible.

Model-free policies compound the problem. They observe positions and velocities and react; they
build no internal model of how crowds evolve or how the robot's own actions affect the people
around it. When the behavioral regime at test time differs from training, there is nothing in the
architecture to absorb that shift.

DreamerV3's RSSM has the structural capacity to resolve this — a latent model of environment
dynamics trained on imagined rollouts, that can in principle represent crowd evolution rather than
just react to it. DreamerNav [Shanks et al., Frontiers 2025] proved the architecture transfers to
indoor navigation, but kept pedestrians as anonymous obstacles with no social structure in the
latent state. This work adds that structure, evaluates across multiple behavior models, and ships
the full pipeline inside RosNav-RL and Arena 5.0.

---

## 2. Proposed Contributions

**At a glance.** Four contributions, all implemented and unit-verified; runs pending (roadmap §4.5):

| # | Contribution | One-line impact |
|---|---|---|
| C1 | Social-RSSM — GAT interaction topology inside the world-model latent | First social graph in the *latent state* rather than the policy; the structural prior H1a tests |
| C2 | cSRSSM — crowd-behavior context b as a Hidden-Parameter-MDP extension of the DreamerV3 ELBO, with prediction-driven identifiability | A *derived* objective, not a loss stack; enables counterfactual social imagination, measured (§4.3), not asserted |
| C3 | DALI-style dynamics context dₜ | Self-supervised sim-to-sim OOD interpolation, tested against a held-out driver fitted to real interaction data |
| C4 | TSSM backbone — obs-only posterior, fully parallel observe(), verified exactly equivalent to the sequential formulation | Removes the RSSM's sequential training bottleneck; completes a three-way backbone ablation |

Alongside these, the **cross-driver evaluation protocol** (train on ORCA + tuned-SFM, hold out
HSFM, report worst-case SR with seeds and CIs) is reusable beyond this paper — it measures
navigation quality where the field's standard protocol measures simulator fit (§1). Multi-driver
domain randomization itself is *enabling methodology*, not a claimed contribution: Kastner et al.
[RA-L 2025] already established its value for model-free policies; the claim here is what
structure adds *on top of* DR. The calibrated KL safety layer is scoped as a **stretch**
contribution (Phase 4, §4.5).

### 2.1 Social-RSSM: GAT-Extended Latent State (C1)

The standard DreamerV3 latent state at time t is the pair (hₜ, zₜ), where h is the deterministic
recurrent state and z a stochastic observation-conditioned representation. Pedestrian information
enters only implicitly through the flat observation encoder. The extension computes a social
context vector cₜ via a Graph Attention Network (GAT) operating over all currently observed
pedestrians, using relative positions, velocities, and estimated social states as node features.

Edges connect agents within a 4 m proximity radius or on a velocity-converging heading — relative
heading, not just distance, following the heterogeneous design of HEIGHT [Liu et al., T-ASE 2026],
since the robot-human interaction is asymmetric. The GAT is conditioned on hₜ, making it
world-model-aware: it attends differently to the same crowd configuration depending on the robot's
current recurrent context.

```
cₜ = GAT({xᵢᵖᵉᵈ}ᵢ₌₁..ₙ, hₜ)
sₜˢᵒᶜⁱᵃˡ = (hₜ, zₜ, cₜ)
```

The augmented state sₜˢᵒᶜⁱᵃˡ feeds the reward predictor, value function, and actor. The world model
can generate imagined rollouts in which crowd dynamics respond to different candidate action
sequences, enabling anticipatory rather than purely reactive behavior.

**Key claim**: pedestrian interaction topology is invariant across simulators — only avoidance
magnitude and dynamics shape differ. Encoding topology explicitly gives the RSSM a structural
prior that transfers across behavior models. This claim is empirical, and H1a is its direct test:
if topology were not the transferable component, the GAT variant would show no cross-driver
advantage over the flat encoder. Richens & Everitt [ICLR 2024] supply the complementary
theoretical motivation for the world-model route as such — agents robust to distribution shift
must learn approximate causal models — but their result does not identify *which* representation
transfers; that attribution is exactly what the ablation isolates.

#### cSRSSM: Crowd-Behavior Context Conditioning (C2)

Alongside the GAT's per-step interaction topology, a second and independently implemented
contribution addresses a different axis of structure: the crowd's *slow* behavioral regime, which
does not change within an episode but does change across simulators and driver settings. We cast
social navigation as a contextual MDP and introduce a `SocialContextEncoder`
(`rosnav_rl/model/dreamerv3/social/context.py`) that infers a latent crowd-behavior code bₜ —
capturing aggressiveness, yielding tendency, preferred speed, personal-space radius, and local
avoidance policy — once per K-step window (K = 16 in the production configuration,
`social_csrssm_config.yaml`) via a GRU encoder followed by two linear heads producing the mean and
std of a diagonal Gaussian posterior q(b | window).

Unlike cₜ, which is recomputed every step from the instantaneous pedestrian graph, bₜ is held
**fixed** across the transition and concatenated directly into `RSSM.img_step`'s input
(`networks.py`) rather than only into the readout — the same "condition the transition, not
just the observation" principle that motivates DALI's dynamics context dₜ (§2.2), but applied to
slow behavioral structure instead of fast trajectory dynamics.

Formally, this casts social navigation as a Hidden-Parameter MDP [Doshi-Velez & Konidaris,
IJCAI 2016]: episodes share state and action spaces but differ in a latent parameter (the crowd
regime) that modulates the transition kernel. The cSRSSM approximates that parameter with b and
extends the DreamerV3 ELBO accordingly. With generative model

```
p(o₁:T, z₁:T, b | a₁:T) = p(b) ∏ₜ p(zₜ | z_{t−1}, a_{t−1}, b) · p(oₜ | zₜ),   p(b) = N(0, I)
```

and variational family q(b | τ_win) ∏ₜ q(zₜ | ·), the ELBO decomposes into the standard DreamerV3
reconstruction and dynamics-KL terms — now with b-conditioned prior transitions — plus exactly one
new term, KL(q(b | τ_win) ‖ p(b)): the `kl_scale` regularizer below. The prediction head is the
VariBAD decoder analogue: without it the b-KL is minimized by posterior collapse; with it, b must
carry regime information sufficient to predict crowd evolution beyond its inference window.

Two regularizers keep the posterior well-behaved. First, an information-bottleneck KL term
KL(q(b)‖N(0,1)) (`kl_scale = 1.0`) — necessary but not sufficient, since on its own it pulls q(b)
toward the prior and would induce posterior collapse. The counter-pressure is a
**prediction-driven identifiability loss** (`pred_scale = 0.1`), VariBAD-style [Zintgraf et al.,
ICLR 2020]: a small head predicts, from (b, pooled crowd summary at step t), the pooled crowd
summary at step t+1, for steps *beyond* the window b was inferred from — so b cannot satisfy the
loss by memorizing window content and must carry regime information that extrapolates. The target
is deliberately regime-level (pooled crowd statistics, not per-pedestrian states). This is a
structural firewall, not a stylistic preference: dₜ's loss (§2.2) is computed per pedestrian on a
(B, N, F) target, while b's loss only ever sees the N-marginalized (B, F) pooled summary — there
is no tensor path by which b's gradient could substitute for dₜ's per-pedestrian job, so the
context × DALI ablation cells (Table 4.2) remain independently attributable by construction. The
known cost is signal strength: mean-pooling over N pedestrians shrinks target variance, so a
persistence baseline may sit close to the optimum and leave b under-pressured. Training therefore
logs the zero-velocity persistence floor MSE(pooled_{t+1}, pooled_t) alongside the prediction
loss; if that gap closes, the target is extended with second-moment pooled statistics (position
and heading dispersion, local density) — still N-marginalized, preserving the firewall. An InfoNCE variant (same-window halves
as positives, other sequences as negatives) is retained as a secondary ablation knob
(`infonce_scale`, default 0) — it is cheaper than the prediction head but has a known
false-negative problem under domain randomization: distinct episodes sharing the same simulator
regime are pushed apart, which is exactly wrong for a regime code.

This conditioning is what makes *counterfactual* social imagination possible: holding (s, a) fixed
and perturbing bₜ produces different imagined crowd rollouts under the same instantaneous scene,
which is the architectural hook the "scenario generation for curriculum training" downstream use in
§5 depends on and currently has no other way to express. The context on/off ablation
(`social_csrssm_config.yaml` vs `social_csrssm_no_context_config.yaml`) is already implemented and
smoke-tested; see the new row in Table 4.2 and the extended H1a/H1b wording in §3.

**Backbone ablation (C4).** The deterministic pathway is implemented as a three-way ablation axis
(config `cell_type`), all sharing the same posterior/prior heads and latent interface:

1. **GRU** (baseline): the standard DreamerV3 recurrence.
2. **Sliding-window attention cell** (`TransformerCell`, `rosnav_rl/model/dreamerv3/networks.py`):
   a causal attention cell over the last `ctx_len = 64` steps (≈6.4 s at the 10 Hz control rate),
   wrapped as a step-by-step recurrent cell inside the same sequential scan the GRU uses. Key/value
   projections of past outputs are cached rather than recomputed each step (O(ctx_len) → O(1)
   amortized, verified numerically identical to full recomputation) — an uncached first-pass
   implementation exhausted memory at the production imagination-batch size (1024) on an 8 GB
   development GPU. This variant isolates *attention-based recurrence* from *parallel training*:
   it changes what the cell computes but not the sequential training schedule.
3. **TSSM** (`TSSM`, same file; STORM-style [Zhang et al., 2023] / TransDreamer-style
   restructuring): the posterior is decoupled from the deterministic path — q(zₜ | oₜ) conditions
   on the observation embedding only — and hₜ is computed by a stack of causal attention blocks
   over the (z, a) token sequence within the same episode segment and sliding window. Because the
   posterior no longer needs hₜ, world-model training runs with **no sequential scan at all**: all
   posteriors are sampled from logged observations at once, all hₜ computed in one parallel masked
   attention pass, all priors p(zₜ | hₜ) in parallel. This is the genuine transformer
   training-speed advantage, and it is structurally unavailable to variants 1–2. The parallel pass
   is verified exactly equivalent to the step-by-step formulation (including across mid-sequence
   episode resets and window truncation) by a unit test. The trade: q(z | o) instead of q(z | h, o)
   is a real change to the probabilistic model — the ablation therefore measures the joint effect
   of attention + parallelism + posterior decoupling, which is exactly the package TransDreamerV3
   (arXiv:2506.17103) proposes.

Stated honestly rather than overclaimed: the parallel-training advantage applies to the
world-model training path (`observe`) only. Imagination rollout remains irreducibly sequential for
**any** backbone, since each imagined step consumes the model's own just-sampled stochastic
latent; TSSM runs it step-by-step with a per-layer key/value cache. TSSM's memory profile also
differs structurally: because the deterministic state is non-Markovian, every imagination start
must carry a (ctx_len × layers) key/value window, which scales imagination memory linearly in
context length and depth — the implementation threads (rather than stacks) this cache across the
horizon to keep that cost bounded.

### 2.2 DALI Dynamics Context Encoder (C3)

C1 encodes *who* interacts but not *how* they interact dynamically; C2 encodes the slow behavioral
regime. Two simulators can produce
identical interaction topology but different trajectory shapes: ORCA pedestrians deflect sharply at
the last moment; SFM pedestrians deflect gradually via social forces. An RSSM conditioned only on cₜ
still sees different dynamics in its transition targets, producing a blended dynamics model that is
inaccurate for all simulators.

A GRU-based encoder observes the trajectory buffer of the last K = 40 pedestrian trajectory steps
(≈4 s at 10 Hz — widened from the original K = 20 design during implementation for a longer
effective horizon) and produces a latent dynamics context vector dₜ, trained via a
forward-prediction auxiliary loss. No simulator label is ever provided — the encoder discovers
dynamics structure through prediction pressure alone.

```
dₜ = proj(GRU(x_buf))
L_dyn = ‖x_{t+1} − f_θ(dₜ, xₜ)‖²
sₜ = (hₜ, zₜ, cₜ, dₜ)
```

The auxiliary loss is added to the DreamerV3 ELBO with weight λ_dyn = 0.1. During DR training on
ORCA + SFM, the encoder learns a context manifold covering two distinct dynamics regimes — a
reciprocal velocity-obstacle avoider (ORCA) and a force-based translational avoider (SFM). At test
time on the held-out HSFM driver, dₜ must place a regime it never saw: HSFM adds body-heading
rotational and lateral dynamics on top of the force-based repulsion core, so it is *adjacent to*
SFM in translational behavior yet carries a structurally new degree of freedom. dₜ inferring the
nearest known regime rather than collapsing on this input is the interpolation question tested
directly.

**Research question.** Does the learned dynamics context dₜ capture structure that transfers to a
held-out driver? We test this by comparing success-rate drop on HSFM and by visualizing whether
HSFM embeddings lie closer to SFM (its dynamical parent) than to ORCA. A negative result would
still be informative — it would mark the limit of interpolation-based OOD handling in social
navigation.

### 2.3 Multi-Driver Domain Randomization (enabling methodology)

Training distribution: ORCA + SFM, uniformly sampled per episode via Arena's humansim pedestrian
stack, where the local-planner plugin is a runtime ROS parameter — zero infrastructure changes
required. Crucially, the training SFM is deliberately tuned into a parameter regime distinct from
the held-out HSFM's defaults (distinct relaxation time, repulsion strength/range, and anisotropy —
all exposed as humansim `ParamDist` gains): without this, HSFM's shared translational core would
make the held-out gap trivial, and H2 would measure parameter interpolation rather than genuine
regime transfer. The concrete tuning offsets are recorded in the experiment configs and reported.
None of C1–C3 can improve generalization if the training distribution is a single simulator; DR is
a prerequisite, not an alternative.

Held-out evaluation: **HSFM** (Headed Social Force Model), a classical, well-validated driver
already in humansim, never seen during training. It is chosen as a *safe, fully reproducible*
held-out target: no external dataset, no unreleased code, deterministic and inspectable. It adds
heading-coupled rotational/lateral avoidance dynamics absent from both training drivers, so it is a
genuine out-of-distribution regime rather than a re-parameterization. The primary result: SR_HSFM,
with the gap SR_SFM − SR_HSFM as the generalization metric.

*Note on realism.* This is a weaker realism claim than a real-data-fitted held-out driver would
support — HSFM is an analytical model, not fitted to human trajectories. We make the honest,
reproducible claim (transfer to a structurally novel *analytical* regime) rather than the stronger
but currently unsupportable one: the only real-data-fitted candidate (NeuRoSFM, PeRoI dataset) has
neither public code nor public data. humansim's GAIL-imitation driver (`socialgail`) is a possible
future real-data-fitted held-out target and is noted as such in §7.

### 2.4 Uncertainty-Aware Safety Layer (stretch)

The RSSM training objective already produces the KL divergence between the posterior and prior at
every step as a free byproduct of the ELBO objective. When cₜ encodes a semantically meaningful
interaction representation, this KL becomes an interpretable out-of-distribution signal: high KL on
a specific interaction pattern means the world model has not encountered this interaction type
before.

```
uₜ = D_KL(q_φ(z | h, o) ‖ p_θ(z | h))
```

The intended use is not to detect crowd density alone, because density-based braking can already be
handled by heuristics. Instead, the KL should respond to unexpected interaction *patterns* — e.g.
when a group suddenly splits or a pedestrian reverses direction sharply relative to recent latent
history.

Two methodological safeguards make this testable rather than anecdotal. First, raw KL magnitude is
not comparable across training stages or behavior regimes — its scale drifts as the world model
improves and differs per driver even in-distribution. The deployed signal is therefore calibrated:
per-driver KL quantiles are estimated on held-out validation episodes (split conformal calibration,
target false-alarm rate α = 0.05), and the safety layer triggers on quantile exceedance, not on raw
KL. Second, the comparison set includes not only a density-based slowdown rule but also the standard
learned alternative — disagreement across an ensemble of dynamics heads — so the claim "posterior-
prior KL is an informative OOD signal" is tested against the epistemic-uncertainty state of practice,
not only against a heuristic. Evaluation: collision rate and personal-space violations with and
without the safety layer, at matched average speed reduction across all three trigger signals.

---

## 3. Hypotheses

Hypotheses are tiered by centrality and cost. **Primary — H1a, H1b, H2**: the paper's claims stand
or fall on these; all are answered by Tier-1/2 runs (Table 4.2) plus the held-out HSFM
evaluation. **Secondary — H4**: cheap (reuses Tier-1 checkpoints, no new world-model training) and
high-value if positive, but the paper survives its failure. **Stretch — H3**: the only hypothesis
requiring substantial unbuilt machinery (ensemble dynamics heads + conformal calibration, §2.4);
it is attempted in Phase 4 (§4.5) and reported if built, dropped without harm to the core claims
otherwise.

**RQ1.** Does adding explicit social interaction structure to the world model improve cross-driver
robustness beyond flat observation encoding?

- **H1a — Architectural robustness (single-driver training).** GAT-Dreamer trained on ORCA alone
  achieves a higher worst-case cross-driver success rate (minimum SR over the unseen drivers) than
  vanilla DreamerV3 and HEIGHT trained identically, at non-inferior mean SR. Worst-case SR rather
  than SR variance: variance also shrinks under uniform degradation, which is not robustness.
  Isolates the structural contribution of the GAT. This is tested along a second structural axis
  alongside the GAT: whether cSRSSM's crowd-behavior context conditioning (§2.1) provides an
  additional, independent reduction in worst-case cross-driver degradation beyond the GAT alone.
- **H1b — DR amplification.** The SR gap between GAT-DALI-Dreamer (DR) and DR-DreamerV3 is larger
  than the gap between GAT-Dreamer (single) and vanilla DreamerV3. Tests whether relational
  structure amplifies domain randomization beyond what flat-encoder DR achieves. The
  GAT-cSRSSM-Dreamer vs GAT-Dreamer comparison (context on/off, Table 4.2) tests the same
  amplification question for the crowd-behavior context axis specifically.

**RQ2.** Does the latent dynamics context of DALI support interpolation to a held-out pedestrian
behavior model?

- **H2 — DALI interpolation in context space.** GAT-DALI-Dreamer shows a smaller SR drop from SFM
  (in-distribution) to HSFM (held-out) than GAT-Dreamer without DALI. Supporting evidence:
  t-SNE of dₜ embeddings from HSFM episodes shows HSFM vectors cluster nearer to SFM (its dynamical
  parent) than to ORCA. A negative result (HSFM isolated in context space) is still informative.

**RQ3.** Does KL-based uncertainty capture socially novel situations better than simple
crowd-density heuristics?

- **H3 (stretch).** A conformally calibrated KL safety layer (§2.4) reduces collision rate and
  personal-space violation rate more effectively than both a density-only slowdown rule and an
  ensemble-disagreement trigger, at matched average speed reduction.

**RQ4.** Does a learned social world model transfer faster than retraining from scratch when the
pedestrian driver changes?

- **H4 (secondary) — Policy transfer with frozen world model.** With a frozen Social-RSSM, re-training only the
  actor-critic on HSFM reaches near-full-retrain performance in substantially fewer environment
  steps than end-to-end retraining. Mechanism: the social representation is already learned; the
  actor-critic re-learns only the decision boundary.

---

## 4. Experimental Design

### 4.1 Training Setup

| Factor | Setting |
|---|---|
| Simulator | Arena 5.0 |
| Pedestrian drivers (train) | ORCA, SFM (tuned distinct from HSFM) — uniform per episode |
| Pedestrian drivers (held-out) | HSFM (heading-coupled, unseen in training) |
| Robot | TurtleBot3 Burger (differential drive) |
| Action space | Continuous: v_lin ∈ [0, 0.5] m/s, v_ang ∈ [−1.0, 1.0] rad/s |
| Observations | LiDAR 360°/720 rays + pedestrian detections (x, y, vₓ, v_y) for N ≤ 8 |
| Training steps | 5M environment steps per ablation run |
| Hardware | RunPod A100 40 GB, ~18 h/run |

### 4.2 Baselines

| Model | Tier | GAT | DALI | Context | DR | Notes |
|---|---|---|---|---|---|---|
| DreamerV3 (DR) | 1 | | | | ✓ | Flat-encoder anchor: DR without structural encoding |
| GAT-Dreamer (DR) | 1 | ✓ | | | ✓ | C1 + DR; "neither" cell of the 2×2 |
| GAT-cSRSSM-Dreamer (DR) | 1 | ✓ | | ✓ (K=16) | ✓ | C1 + C2; context-only cell |
| GAT-DALI-Dreamer (DR) | 1 | ✓ | ✓ | | ✓ | C1 + C3; DALI-only cell |
| GAT-DALI-cSRSSM-Dreamer (DR) | 1 | ✓ | ✓ | ✓ (K=16) | ✓ | **Full system, headline**: C1 + C2 + C3; "both" cell |
| HEIGHT (policy) | 1 | ✓ | | | ✓ | Model-free SOTA GAT baseline, matched observation space and step budget |
| DreamerV3 (single) | 2 | | | | | ORCA only; quantifies what DR alone buys |
| DALI-Dreamer (DR) | 2 | | ✓ | | ✓ | C3 without C1 — does dₜ need topology? |
| TSSM-GAT-DALI-cSRSSM-Dreamer | 2 | ✓ | ✓ | ✓ (K=16) | ✓ | C4: GRU → Transformer backbone ablation of the full system (§2.1, three-way axis): TSSM — obs-only posterior + fully parallel observe(), sequential KV-cached imagination; b conditions the token embedding. Intermediate variant (attention-in-a-scan sliding-window cell, ctx_len=64) isolates attention from parallelism; viable on the stated A100 40GB, OOMs on ≤8GB dev hardware without the KV cache |
| CrowdNav++ (policy) | opt. | ✓ | | | | Older model-free graph baseline; run only if reviewer-requested — HEIGHT supersedes it |

**Tier 1** is the paper's spine: the context × DALI 2×2 — GAT-Dreamer (neither), GAT-cSRSSM
(context only), GAT-DALI (DALI only), GAT-DALI-cSRSSM (both) — plus the flat-encoder and
model-free anchors. Each 2×2 cell removes exactly one mechanism, and the pooled-vs-per-pedestrian
target firewall (§2.1) guarantees the two axes cannot silently re-implement one another. **Tier 2**
is run only after Tier 1 clears its gate (§4.5). Compute: Tier 1 = 4 × 5 seeds + 2 × 3 seeds
= 26 runs ≈ 470 A100-hours; Tier 2 adds 9 runs ≈ 160 h. Model-free baselines are substantially
cheaper and do not dominate the budget.

### 4.3 Metrics

**Navigation efficiency**: Success Rate (SR), Collision Rate (CR), SPL (Success weighted by Path
Length) [Anderson et al., ICLR 2018], Time to Goal (TTG).

**Safety**: Near-Miss Rate (NMR) at d < 0.4 m (intimate zone, Hall 1966).

**Social compliance**: Personal Space Violation Rate (PSVR) at d < 1.2 m (personal zone,
Hall 1966) — used in CrowdNav++ and HEIGHT; minimum pedestrian distance d_min per episode
(SocNavBench, T-RO 2022); trajectory jerk (mean |a| per timestep); Time-to-Collision (TTC)
[Han, ICRA 2023].

**OOD diagnostics**: t-SNE of dₜ embeddings per simulator (500 episodes each); KL uncertainty uₜ
distribution per simulator.

**Statistical protocol**: every reported number aggregates ≥ 3 seeds (5 for the context × DALI 2×2
cells and the TSSM row); point estimates are interquartile means with 95% stratified-bootstrap
confidence intervals [Agarwal et al., NeurIPS 2021]; cross-model claims are made only where
intervals separate.

**Context identifiability diagnostics** (cSRSSM): (i) *linear probe* — a logistic-regression probe
from frozen b to the episode's driver identity (and driver parameters, where randomized), labels
never seen during training; probe accuracy is a direct measure of how much regime information b
carries, independent of downstream reward. (ii) *counterfactual consistency* — swap b between
episodes from different regimes with (s, a) held fixed and measure imagined-rollout divergence;
near-zero for same-regime swaps and growing with regime distance turns "counterfactual social
imagination" (§2.1) from a claim into a measurement. (iii) *prediction-floor gap* — the logged
margin between the b-conditioned prediction loss and the zero-velocity persistence floor (§2.1); a
vanishing gap is the early-warning signal for context collapse.

**Backbone throughput**: wall-clock world-model training throughput (observe() steps/s at the
production batch size) for GRU vs sliding-window cell vs TSSM — quantifies the parallel-training
claim instead of asserting it.

### 4.4 Deployment Feasibility

Inference requires only LiDAR (720 values) and a 2D pedestrian tracker output (N×4 relative poses
and velocities). No RGB camera, no pedestrian intent annotation. Any ROS 2-compatible laser-based
tracker (e.g., DR-SPAAM, leg_detector) is sufficient.

- RSSM step on Jetson Orin NX 16 GB: ~5 ms
- GAT forward pass (N=8): ~1 ms
- DALI GRU forward pass: ~1 ms

Total inference: ~7 ms — well within the 100 ms navigation control cycle.

### 4.5 Roadmap

| Phase | Content | Gate to next phase |
|---|---|---|
| **0 — done** | All four contributions implemented and unit-verified: TSSM parallel/sequential exact-equivalence test, KV-cache equivalence test, b-identifiability test (informative b beats shuffled b), smoke configs | — |
| **1 — diagnostics** | Instrument before spending compute: persistence-floor logging, linear probe (b → driver ID), counterfactual-consistency eval, backbone throughput benchmark; then one full-system pilot run on the A100 | Floor gap > 0 and probe accuracy > chance on the pilot — i.e., the context code demonstrably not collapsed. If the gate fails, apply the second-moment mitigation (§2.1) before Phase 2 |
| **2 — core ablations** | Tier 1 (Table 4.2): context × DALI 2×2 at 5 seeds + flat-encoder and HEIGHT anchors at 3 seeds | H1a/H1b readout with separating CIs on worst-case SR |
| **3 — OOD + transfer** | Held-out HSFM evaluation of all Tier-1 checkpoints (H2); frozen-world-model transfer (H4); Tier-2 rows incl. TSSM | H2/H4 readout |
| **4 — stretch** | Ensemble dynamics heads + conformal KL safety layer (H3); ETH/UCY zero-shot diagnostic | — |

Phases 2–4 consume the compute budget (§4.2); Phase 1 exists so that no A100-hours are spent on a
collapsed context code or an unmeasured parallelism claim.

---

## 5. Downstream Uses of the Trained World Model

Scoped by realism: one use is evaluated in the core plan (Phase 3), two are conditional on the
stretch phase (Phase 4), and two are outlook — stated as such rather than claimed.

**Evaluated in the core plan (Phase 3):**

**Sample-efficient policy transfer (H4).** A frozen full-system world model with a freshly trained
actor-critic on the held-out driver — pure simulation, reuses Tier-1 checkpoints, no new
world-model training. If the world model has learned transferable crowd structure, policy learning
inside it should need far fewer environment steps than retraining from scratch; sample-efficiency
curves (IQM + CIs, §4.3) are the readout. This is the cheapest downstream demonstration and the
one with existing infrastructure end to end.

**Conditional on Phase 4:**

**Zero-shot pedestrian trajectory prediction.** The RSSM imagines K-step trajectories for each
pedestrian by running the world model forward under the social context encoding. Zero-shot
evaluation on ETH/UCY [Pellegrini et al., ECCV 2010] uses the standard protocol (observe 8 frames,
predict 12 at 0.4 s; ADE/FDE), feeding dataset tracks through the same pedestrian-detection
interface the tracker provides at deployment, with a virtual stationary robot as ego. The domain
gap is real — the model is trained on simulated crowds around a moving robot — so this is scoped
as a diagnostic of what the learned crowd prior captures, not as a competitive-benchmark claim; it
requires no additional training either way.

**KL divergence as online safety signal.** The per-step uₜ = D_KL(q_φ‖p_θ) measures world-model
surprise. When uₜ exceeds a calibrated threshold, the velocity attenuation layer reduces command
speed proportionally — graceful degradation in unknown situations rather than a hard stop.
Requires the Phase-4 calibration and ensemble-baseline machinery (§2.4, H3).

**Outlook (not evaluated in this work):**

**Sim-to-real transfer via SimDist paradigm** [Levy et al., RSS 2026]. Freeze Social-RSSM weights
after simulation training. Collect 15–30 minutes of real-robot interaction data. Fine-tune only the
RSSM transition-dynamics head via supervised prediction loss — the GAT encoder, cₜ, and dₜ remain
frozen. This is H4's frozen-world-model mechanism applied to real-world data — realistic in
principle, but it requires robot access and 15–30 minutes of curated real interaction data, so it
is outlook rather than a promised deliverable.

**Scenario generation for curriculum training.** Steering the dₜ context vector toward high-KL
regions allows the world model to imagine adversarial crowd scenarios (sudden crossings, bottleneck
traversals, group formations) and inject them into Arena's curriculum scheduler as additional
training episodes. The cSRSSM context vector bₜ (§2.1) provides a second, complementary steering
axis for this same use: perturbing bₜ with (s, a) fixed produces different imagined crowd behavior
regimes under an otherwise identical scene, rather than different fast trajectory shapes. The
steering-and-injection machinery does not exist yet; the counterfactual-consistency metric (§4.3)
is its feasibility probe — if b-swaps produce measurably regime-consistent imagined rollouts,
steered generation is worth building.

---

## 6. Positioning Against Prior Work

| Method | Social Graph | Dynamics ID | DR | OOD Eval | Platform |
|---|---|---|---|---|---|
| DreamerNav [Frontiers 2025] | | | | | Warehouse robot |
| Wei Zhu et al. [RA-L 2023] | | | | | Ground robot |
| CrowdNav++ [ICRA 2023] | ✓ (policy) | | | | TurtleBot2 |
| HEIGHT [T-ASE 2026] | ✓ (policy) | | | | TurtleBot2 |
| DALI [NeurIPS 2025] | | ✓ (physics) | | | DMControl |
| **This work** | ✓ (world model) | ✓ | ✓ | ✓ | TurtleBot3 (held-out HSFM) |

No existing paper combines all four key dimensions. DreamerNav and Wei Zhu et al. place the world
model in navigation but omit social structure and OOD evaluation. HEIGHT and CrowdNav++ encode
social graphs in the policy, not the world model, and do not test cross-simulator generalization.
DALI provides dynamics identification and OOD robustness in physical dynamics domains with no
social structure. This work is the first to combine social graph encoding within the world model's
latent state with self-supervised dynamics context identification, validated on a held-out dataset
derived from real pedestrian-robot interaction data.

---

## 7. Limitations and Threats to Validity

- **TSSM confound (acknowledged in §2.1).** The TSSM row changes attention, parallelism, and the
  posterior family together; the sliding-window cell row is the isolating intermediate, and no
  claim attributes TSSM gains to attention alone.
- **Pooled-target signal strength.** b's identifiability pressure depends on regime differences
  surviving mean-pooling; the prediction-floor gap (§4.3) is the online health check and
  second-moment pooled targets the pre-planned mitigation (§2.1).
- **Analytical held-out driver, not real-data-fitted.** HSFM is a classical model; the
  generalization claim is transfer to a structurally novel *analytical* regime, not to fitted
  human behavior. The real-data-fitted target (NeuRoSFM / PeRoI) has neither public code nor
  public data. humansim's GAIL-imitation driver (`socialgail`) is the natural future upgrade to a
  real-data-fitted held-out set and would strengthen H2 to the original intended claim.
- **Single held-out driver.** One held-out driver (HSFM) bounds the strength of any generalization
  claim; results are reported as such. A second family-distinct held-out (e.g. a time-to-collision
  anticipatory model) is a cheap future extension — held-out drivers cost eval episodes only, no
  training compute.
- **Shared driver lineage.** ORCA, SFM, and HSFM are all reciprocal/force-based families; neither
  training nor evaluation covers qualitatively different pedestrian behavior (group intent,
  vision-conditioned, adversarial).
- **Sim-only headline results.** Real-world evidence enters only through the SimDist fine-tuning
  path (§5); the headline claims are sim-to-sim generalization, not sim-to-real performance.

---

## Appendix A — Related Work

**Social Robot Navigation (Model-Free RL).** SARL [Chen et al., ICRA 2019] introduced
self-attention over pedestrian features; it treats pedestrians as independent agents and ignores
human-human interactions. CrowdNav++ [Liu et al.] extends this with no cross-simulator
generalization tested. HEIGHT [Liu et al., T-ASE 2026] is the current model-free state of the art —
a heterogeneous interaction-graph transformer encodes social structure in the policy network, not
the world model. HumanFlow [Schaefer et al., RSS 2026] couples diffusion-based pedestrian
forecasting with MPC for MAV platforms — aerial-only, no world model.

**World Models for Navigation.** DreamerV3 [Hafner et al., Nature 2025] is the base algorithm.
DreamerNav [Shanks et al., Frontiers 2025] applies DreamerV3 to indoor navigation; pedestrians are
anonymous dynamic obstacles. Wei Zhu et al. [RA-L 2023] applied DreamerV1/2 to pedestrian scenarios
with LiDAR; no relational structure, no OOD evaluation. ELVIS [RSS 2026 #182] adds
ensemble-calibrated latent imagination for long-horizon visual MPC; relevant to the KL safety
signal design but not evaluated in crowd environments. STORM [Zhang et al., NeurIPS 2023] and
TransDreamer [Chen et al., 2022] establish the transformer state-space model design (obs-only
posterior, parallel world-model training) that the TSSM backbone variant (§2.1) follows.

**Dynamics-Aware and Causal World Models.** DALI [Röder et al., NeurIPS 2025] introduces
self-supervised context encoding for dynamics identification. The Hidden-Parameter MDP
[Doshi-Velez & Konidaris, IJCAI 2016] is the formal frame for the cSRSSM (§2.1): a family of MDPs
indexed by a latent parameter that modulates the transition kernel. VariBAD [Zintgraf et al.,
ICLR 2020] establishes the inference template the cSRSSM follows: a trajectory-window posterior
over a latent context, trained by decoding future dynamics plus a KL to a fixed prior. HAIC [RSS 2026 #13] applies
dynamics-aware world models to humanoid manipulation with unknown object dynamics — a structural
parallel to pedestrian dynamics identification. Causal World Modeling for Robot Control [RSS 2026
#16] and Richens & Everitt [ICLR 2024] provide theoretical grounding for causal vs. correlational
world model design.

**Domain Randomization and Sim-to-Real.** Kastner et al. [RA-L 2025] demonstrate that DR over
ORCA, SFM, and HSFM significantly improves social compliance compared to single-simulator training
in model-free RL. Simulation Distillation [Levy et al., RSS 2026] introduces world-model-based
rapid sim-to-real adaptation by freezing reward and value models and adapting only latent dynamics —
the template for H4 and the SimDist fine-tuning path (§5).

**Evaluation methodology.** Aggregate metrics follow rliable [Agarwal et al., NeurIPS 2021]:
interquartile means with stratified-bootstrap confidence intervals over ≥ 3 seeds. The KL safety
threshold is calibrated with split conformal prediction [Angelopoulos & Bates, 2023].

**Language and Foundation Models for Reward Design.** VLM-Social-Nav [Song et al., RA-L 2025] shows
VLM-scored social compliance improves SR by over 27% in four scenarios. ELEMENTAL [ICML 2025]
demonstrates interactive reward refinement via language feedback.
