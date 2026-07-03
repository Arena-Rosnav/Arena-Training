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
prior that transfers across behavior models. This is grounded by Richens & Everitt [ICLR 2024], who
prove that agents achieving sublinear regret under distribution shift must approximate causal world
models.

#### cSRSSM: Crowd-Behavior Context Conditioning

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

Two regularizers keep the posterior well-behaved. First, an information-bottleneck KL term
KL(q(b)‖N(0,1)) (`kl_scale = 1.0`) — necessary but not sufficient, since on its own it pulls q(b)
toward the prior and would induce posterior collapse. The counter-pressure is a
**prediction-driven identifiability loss** (`pred_scale = 0.1`), VariBAD-style [Zintgraf et al.,
ICLR 2020]: a small head predicts, from (b, pooled crowd summary at step t), the pooled crowd
summary at step t+1, for steps *beyond* the window b was inferred from — so b cannot satisfy the
loss by memorizing window content and must carry regime information that extrapolates. The target
is deliberately regime-level (pooled crowd statistics, not per-pedestrian states): per-pedestrian
next-step prediction is dₜ's job (§2.2), and sharing that target would make the two codes
redundant and the context on/off ablation uninterpretable. An InfoNCE variant (same-window halves
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

**Backbone ablation.** The deterministic pathway is implemented as a three-way ablation axis
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

### 2.2 DALI Dynamics Context Encoder (C2)

C1 encodes *who* interacts but not *how* they interact dynamically. Two simulators can produce
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
ORCA + SFM + HSFM, the encoder learns a context manifold covering three distinct dynamics regimes.
At test time on a held-out simulator (NeuRoSFM), dₜ infers the nearest known dynamics regime rather
than collapsing on unrecognized input — the interpolation question this proposal tests directly is
whether that inference is meaningfully "nearest known" rather than arbitrary under out-of-distribution
input.

**Research question.** Does the learned dynamics context dₜ capture structure that transfers to a
held-out simulator? We test this by comparing success-rate drop on NeuRoSFM and by visualizing
whether NeuRoSFM embeddings lie closer to SFM than to ORCA or HSFM. A negative result would still be
informative, because it would mark the limit of interpolation-based OOD handling in social
navigation.

### 2.3 Multi-Driver Domain Randomization (C3)

Training distribution: ORCA + SFM + HSFM, uniformly sampled per episode via Arena 5.0 /
PedsimROS — zero infrastructure changes required (one config parameter). Neither C1 nor C2 can
improve generalization if the training distribution is a single simulator; DR is a prerequisite,
not an alternative.

Held-out evaluation: NeuRoSFM [Agrawal, Dengler, Bennewitz — ICRA 2026, UniBonn], a neural SFM
fitted to the PeRL real pedestrian-robot interaction dataset. It was never seen during training and
is currently the most realistic available held-out target. The primary result: SR_NeuRoSFM, and the
gap SR_SFM − SR_NeuRoSFM as the generalization metric.

### 2.4 Uncertainty-Aware Safety Layer

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

We verify that this signal adds value beyond simple heuristics by comparing it against crowd-density
features, and by evaluating collision rate and personal-space violations with and without the safety
layer, as well as against a density-based slowdown baseline.

---

## 3. Hypotheses

**RQ1.** Does adding explicit social interaction structure to the world model improve cross-driver
robustness beyond flat observation encoding?

- **H1a — Architectural robustness (single-driver training).** GAT-Dreamer trained on ORCA alone
  shows smaller cross-driver success-rate variance than vanilla DreamerV3 and HEIGHT trained
  identically. Isolates the structural contribution of the GAT. This is tested along a second
  structural axis alongside the GAT: whether cSRSSM's crowd-behavior context conditioning (§2.1)
  provides an additional, independent reduction in cross-driver variance beyond the GAT alone.
- **H1b — DR amplification.** The SR gap between GAT-DALI-Dreamer (DR) and DR-DreamerV3 is larger
  than the gap between GAT-Dreamer (single) and vanilla DreamerV3. Tests whether relational
  structure amplifies domain randomization beyond what flat-encoder DR achieves. The
  GAT-cSRSSM-Dreamer vs GAT-Dreamer comparison (context on/off, Table 4.2) tests the same
  amplification question for the crowd-behavior context axis specifically.

**RQ2.** Does the latent dynamics context of DALI support interpolation to a held-out pedestrian
behavior model?

- **H2 — DALI interpolation in context space.** GAT-DALI-Dreamer shows a smaller SR drop from SFM
  (in-distribution) to NeuRoSFM (held-out) than GAT-Dreamer without DALI. Supporting evidence:
  t-SNE of dₜ embeddings from NeuRoSFM episodes shows NeuRoSFM vectors cluster nearer to SFM than to
  ORCA or HSFM. A negative result (NeuRoSFM isolated) is still informative.

**RQ3.** Does KL-based uncertainty capture socially novel situations better than simple
crowd-density heuristics?

- **H3.** A KL-aware safety layer reduces collision rate and personal-space violation rate more
  effectively than a density-only slowdown rule at matched average speed reduction.

**RQ4.** Does a learned social world model transfer faster than retraining from scratch when the
pedestrian driver changes?

- **H4 — Policy transfer with frozen world model.** With a frozen Social-RSSM, re-training only the
  actor-critic on NeuRoSFM reaches near-full-retrain performance in substantially fewer environment
  steps than end-to-end retraining. Mechanism: the social representation is already learned; the
  actor-critic re-learns only the decision boundary.

---

## 4. Experimental Design

### 4.1 Training Setup

| Factor | Setting |
|---|---|
| Simulator | Arena 5.0 |
| Pedestrian drivers (train) | ORCA, SFM, HSFM — uniform per episode |
| Pedestrian drivers (held-out) | NeuRoSFM (ICRA 2026) |
| Robot | TurtleBot3 Burger (differential drive) |
| Action space | Continuous: v_lin ∈ [0, 0.5] m/s, v_ang ∈ [−1.0, 1.0] rad/s |
| Observations | LiDAR 360°/720 rays + pedestrian detections (x, y, vₓ, v_y) for N ≤ 8 |
| Training steps | 5M environment steps per ablation run |
| Hardware | RunPod A100 40 GB, ~18 h/run |

### 4.2 Baselines

| Model | GAT | DALI | Context | DR | Notes |
|---|---|---|---|---|---|
| DreamerV3 (single) | | | | | ORCA only; primary baseline |
| DreamerV3 (DR) | | | | ✓ | DR without structural encoding |
| GAT-Dreamer (DR) | ✓ | | | ✓ | C1 + C3 only |
| DALI-Dreamer (DR) | | ✓ | | ✓ | Tests DALI without GAT |
| GAT-cSRSSM-Dreamer (DR) | ✓ | | ✓ (K=16) | ✓ | Tests cSRSSM crowd-behavior context conditioning (§2.1) on/off vs GAT alone |
| GAT-DALI-Dreamer (DR) | ✓ | ✓ | | ✓ | Full system: C1 + C2 + C3 |
| TSSM-GAT-DALI-Dreamer (Transformer) | ✓ | ✓ | | ✓ | GRU → Transformer backbone ablation (§2.1, three-way axis). Headline variant: TSSM — obs-only posterior + fully parallel observe(), sequential KV-cached imagination. Intermediate variant (attention-in-a-scan sliding-window cell, ctx_len=64) isolates attention from parallelism; viable on the stated A100 40GB, OOMs on ≤8GB dev hardware without the KV cache |
| CrowdNav++ (policy) | ✓ | | | | Graph in policy, not world model |
| HEIGHT (policy) | ✓ | | | | Model-free GAT baseline |

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

### 4.4 Deployment Feasibility

Inference requires only LiDAR (720 values) and a 2D pedestrian tracker output (N×4 relative poses
and velocities). No RGB camera, no pedestrian intent annotation. Any ROS 2-compatible laser-based
tracker (e.g., DR-SPAAM, leg_detector) is sufficient.

- RSSM step on Jetson Orin NX 16 GB: ~5 ms
- GAT forward pass (N=8): ~1 ms
- DALI GRU forward pass: ~1 ms

Total inference: ~7 ms — well within the 100 ms navigation control cycle.

---

## 5. Downstream Uses of the Trained World Model

**Zero-shot pedestrian trajectory prediction.** The RSSM imagines K-step trajectories for each
pedestrian by running the world model forward under the social context encoding. Zero-shot
evaluation on ETH/UCY datasets [Pellegrini et al., ECCV 2010] tests whether the imagined rollouts
constitute a standalone secondary contribution requiring no additional training.

**KL divergence as online safety signal.** The per-step uₜ = D_KL(q_φ‖p_θ) measures world-model
surprise. When uₜ exceeds a calibrated threshold, the velocity attenuation layer reduces command
speed proportionally — graceful degradation in unknown situations rather than a hard stop.

**Sim-to-real transfer via SimDist paradigm** [Levy et al., RSS 2026]. Freeze Social-RSSM weights
after simulation training. Collect 15–30 minutes of real-robot interaction data. Fine-tune only the
RSSM transition-dynamics head via supervised prediction loss — the GAT encoder, cₜ, and dₜ remain
frozen. This is H3 applied to real-world data.

**Scenario generation for curriculum training.** Steering the dₜ context vector toward high-KL
regions allows the world model to imagine adversarial crowd scenarios (sudden crossings, bottleneck
traversals, group formations) and inject them into Arena's curriculum scheduler as additional
training episodes. The cSRSSM context vector bₜ (§2.1) provides a second, complementary steering
axis for this same use: perturbing bₜ with (s, a) fixed produces different imagined crowd behavior
regimes under an otherwise identical scene, rather than different fast trajectory shapes.

---

## 6. Positioning Against Prior Work

| Method | Social Graph | Dynamics ID | DR | OOD Eval | Platform |
|---|---|---|---|---|---|
| DreamerNav [Frontiers 2025] | | | | | Warehouse robot |
| Wei Zhu et al. [RA-L 2023] | | | | | Ground robot |
| CrowdNav++ [ICRA 2023] | ✓ (policy) | | | | TurtleBot2 |
| HEIGHT [T-ASE 2026] | ✓ (policy) | | | | TurtleBot2 |
| DALI [NeurIPS 2025] | | ✓ (physics) | | | DMControl |
| **This work** | ✓ (world model) | ✓ | ✓ | ✓ | TurtleBot3 (+ NeuRoSFM) |

No existing paper combines all four key dimensions. DreamerNav and Wei Zhu et al. place the world
model in navigation but omit social structure and OOD evaluation. HEIGHT and CrowdNav++ encode
social graphs in the policy, not the world model, and do not test cross-simulator generalization.
DALI provides dynamics identification and OOD robustness in physical dynamics domains with no
social structure. This work is the first to combine social graph encoding within the world model's
latent state with self-supervised dynamics context identification, validated on a held-out dataset
derived from real pedestrian-robot interaction data.

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
self-supervised context encoding for dynamics identification. VariBAD [Zintgraf et al., ICLR 2020]
establishes the contextual-MDP template the cSRSSM follows: a trajectory-window posterior over a
latent context, trained by decoding future dynamics plus a KL to a fixed prior. HAIC [RSS 2026 #13] applies
dynamics-aware world models to humanoid manipulation with unknown object dynamics — a structural
parallel to pedestrian dynamics identification. Causal World Modeling for Robot Control [RSS 2026
#16] and Richens & Everitt [ICLR 2024] provide theoretical grounding for causal vs. correlational
world model design.

**Domain Randomization and Sim-to-Real.** Kastner et al. [RA-L 2025] demonstrate that DR over
ORCA, SFM, and HSFM significantly improves social compliance compared to single-simulator training
in model-free RL. Simulation Distillation [Levy et al., RSS 2026] introduces world-model-based
rapid sim-to-real adaptation by freezing reward and value models and adapting only latent dynamics —
the template for H3.

**Language and Foundation Models for Reward Design.** VLM-Social-Nav [Song et al., RA-L 2025] shows
VLM-scored social compliance improves SR by over 27% in four scenarios. ELEMENTAL [ICML 2025]
demonstrates interactive reward refinement via language feedback.
