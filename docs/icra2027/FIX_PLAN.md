# Social-Dreamer ICRA 2027 — Reviewer-Risk Fix Plan (revised)

Revision of an earlier tier list, corrected against the actual implementation
(`rosnav_rl/model/dreamerv3/`) and `docs/social_dreamer_proposal.md`. Each task is
scoped for independent execution by an implementation agent. Verify claims against
code before editing prose — the earlier draft contained two factual errors (noted
inline) that this revision corrects.

---

## EXECUTION STATUS (2026-07-09)

- **Task 2 — DONE.** Corrected a wrong assumption first: production
  `social_csrssm_config.yaml` runs `window: 16` with `batch_length: 64` — VALID
  (49 out-of-window targets), so the proposal's "K=16" is correct, NOT a
  transcription error. The real defect was the `context.py:24` docstring hard-
  asserting "K<=8" while production runs 16 → fixed. Added a fail-fast cross-field
  validator to `DreamerV3Cfg` (`cfg.py`) raising when `window >= batch_length` and
  `pred_scale > 0` (silent-loss-disable guard) + `tests/test_context_window_validator.py`
  (4 tests, pass). Full social/config suite: 62 passed, no regression.
- **Task 3 — VERIFIED, ready to execute.** humansim ships EIGHT selectable drivers:
  force-based `sfm`/`orca`/`hsfm`/`straight`, learned `socialgail` (GAIL) + `nsp`
  (Neural Social Physics), and DRL crowd-nav `sarl`/`cadrl`/`dsrnn`/`drlvo`. All
  import-load cleanly (verified; weights fetch at instantiation, not import).
  Recommendation: **socialgail as held-out #2** (learned/data-fitted, disjoint from
  the force-based family — directly upgrades the proposal's own stated realism
  weakness), and OPTIONALLY a DRL policy (`sarl` or `cadrl`) as held-out #3 for a
  structurally-diverse OOD sweep (analytical-force / imitation-learned / RL-policy).
  Remaining work is eval-matrix prose + one instantiation smoke per chosen driver
  (confirm weight fetch succeeds at runtime).
- **Task 4 — mostly ALREADY IN THE PROPOSAL.** The identifiability diagnostics are
  already documented as methods (§4.3: linear probe, counterfactual consistency;
  Phase-1 roadmap gate). Only remaining action is adding a results *figure* once
  Phase 1 runs — downgraded to a post-Phase-1 presentation task, no code.
- **Task 6 — DONE.** Added an EvoNav / hand-designed-reward limitation bullet to §7.
- **Task 1 — DONE (code + tests + prose); flag-on validated offline.** User chose
  Option A. Temporal-alignment convention settled in writing (below). Wired: `cfg.py`
  `gat.condition_transition` flag + tssm fail-fast; `networks.py` RSSM input widen +
  `gat_code` threaded through `img_step`/`obs_step`/`observe` (in-scan `gat_code_fn`,
  None-safe first step); `models.py` posterior call (observed peds, shifted, per-step
  in-scan GAT) + imagination reuse of readout `c_t` (`_get_augmented_feat(_return_c_t)`);
  `dreamer.py` deploy `_policy` (prev-tick crowd cache in the state tuple).
  Tests `test_gat_condition_transition.py` (5/5): width widens by out_dim, c_t moves the
  prior, None-branch baseline, observe in-scan path across is_first, and **gradient flows
  from a rollout loss into both the widened transition columns and the code-producing
  params** (the end-to-end correctness gate). Byte-identity: full rosnav_rl suite 505
  passed / 2 known-fail (flag off = bit-identical). Prose done: §1 C1 row, §2.1 (transition
  conditioning + teacher-forcing asymmetry + gated ablation), H1a third sub-ablation.
  NOTE: the flag-ON Gazebo smoke (`social_csrssm_smoke_peds_gatT.yaml`) reached clean init
  + widened-transition model construction + prefill entry with zero errors, then stalled on
  the PRE-EXISTING `task_generator_node` episode-reset busy-spin (unrelated 5th bug that
  blocks every Gazebo run). Offline gradient test is the stronger, deterministic substitute.
  OPEN (separate bug): the task_generator busy-spin needs a `SYS_PTRACE` container or in-code
  diagnostics to root-cause; it gates all in-sim runs, not just this one.
- **Task 5 — COMMIT (build H3).** User chose to build it. See the commit section below;
  it is its own Phase-4 effort, not part of the Task 1 pass.

---

## Ground truth (verified 2026-07-09, do not re-litigate without re-reading code)

- `b` (cSRSSM context) **already conditions the transition**: `RSSM.__init__` adds
  `context_size` to the GRU input width (`networks.py` ~146-155) and `img_step` /
  `obs_step` take `context=` (~398, 413). Proposal §2.1 states this correctly.
- The GAT **already runs during imagination**: the M2 pedestrian-reconstruction head
  decodes ped node-sets from the latent, and `_get_augmented_feat` applies
  decode-then-GAT at every imagined step (`models.py` ~204, ~1124-1131). Imagined
  actor/critic/reward inputs are graph-shaped.
- The genuine residual gap: `cₜ` augments the **readout feature**, not the transition
  prior p(z_{t+1} | z_t, a_t, b). But `cₜ = GAT(decode(z_t), h_t)` is a deterministic
  function of the latent state — injecting it into `img_step` adds no information the
  transition doesn't already have, only an inductive-bias skip path, and creates a
  decode→GAT→transition feedback loop during imagination.
- Window-length inconsistency is a **three-way** mismatch: proposal §2.1 says
  "K = 16 in the production configuration"; `cfg.py` default `window: 8`;
  `context.py` docstring says "K<=8"; smoke configs use `window: 6`.

---

## Task 1 — GAT-conditioned transition (Option A, revised design) — HIGHEST PRIORITY

**Problem.** Proposal line ~36 claims "First social graph in the *latent state* rather
than the policy." A careful reviewer maps this to: does the graph condition the latent
*transition*? Today it doesn't — `b` does, `cₜ` doesn't. USER DECISION: implement
Option A (make the claim true) rather than weaken the claim.

**Design constraint that shapes the implementation.** Naive Option A — feed
`cₜ = GAT(decode(zₜ))` into `img_step` — is information-free (cₜ is a deterministic
function of the latent) and creates a decode→GAT→transition feedback loop. The
revised design avoids both:

- **Posterior path (training, `obs_step`/`observe`):** compute `cₜ` from the
  **observed** pedestrian node-set (`PedestrianNodeSetSpace` input, already available
  in the batch as `_peds_bt`). This is genuine out-of-latent information — same
  epistemic status as `b`. Concatenate into the transition input alongside `b`.
- **Prior path (imagination, `img_step`):** compute `cₜ` from the **decoded**
  node-set via the existing M2 recon head + decode-then-GAT machinery
  (`models.py` `_get_augmented_feat` already does exactly this for the readout —
  reuse it, do not duplicate). Teacher-forcing asymmetry: real graph in training,
  imagined graph in rollout — standard world-model practice (same as the
  observation/decoder asymmetry itself).
- **Config gate:** `social.gat.condition_transition: bool = false` in `SocialGATCfg`
  (`cfg.py`). Default OFF so every existing config/checkpoint/ablation row is
  byte-identical. New Table 4.2 ablation row: GAT-readout vs GAT-readout+transition.

**Implementation steps (code, `rosnav_rl/model/dreamerv3/`):**

1. `cfg.py`: add `condition_transition: bool = False` to the GAT config block.
2. `networks.py` RSSM: `context_size` already widens the GRU input for `b`. Widen by
   `gat.out_dim` additionally when the flag is on (transition input becomes
   `cat(stoch, action, b, cₜ)`). Thread a `gat_code=` kwarg through `obs_step`,
   `img_step`, `observe`, `imagine_with_action` mirroring how `context=` is threaded
   today — same call sites, same None-default no-op contract.
3. `models.py` WorldModel:
   - training path (`_train`/`observe` call): compute per-step `cₜ` from observed
     `_peds_bt` with the existing GAT (masked, SE(2)-canonicalized — reuse the
     readout path's preprocessing verbatim), pass as `gat_code`.
   - imagination path (`ImagBehavior._imagine`): at each imagined step, `cₜ` from
     decoded peds is **already computed** for the readout (`_get_augmented_feat`,
     ~1124-1131) — return/capture it and pass into the next `img_step` instead of
     recomputing. Detach is NOT applied (gradient through the imagined graph is the
     point of `imag_gradient: dynamics`).
   - deploy path (`dreamer.py:_policy`): observed peds are in `obs` — compute `cₜ`
     from them, pass into `obs_step` (line ~217). One extra GAT call per control
     tick; GAT is already run per tick for the readout, so budget ~2x GAT ≈ small.
4. Tests (mirror `test_social_context.py` conventions):
   - flag off → outputs byte-identical to current (regression guard).
   - flag on → transition input width asserted; forward/backward smoke on synthetic
     batch; imagination rollout runs for `imag_horizon` steps without shape drift.
   - equivalence: posterior-path `cₜ` from observed peds == readout GAT on the same
     input (shared module, not a copy).
5. Proposal prose (§1 table, §2.1): claim becomes fully accurate — "the interaction
   graph conditions both the transition prior and the imagination-time readout;
   observed graph in training, decoded graph in imagination." Document the
   teacher-forcing asymmetry explicitly (reviewers ask). H1a gains a sub-ablation:
   readout-only vs readout+transition GAT.

**Done when:** flag-off byte-identity test passes; flag-on smoke passes; smoke config
run (`social_csrssm_smoke_peds_verify.yaml` + flag on) reaches nonzero losses in
Gazebo; proposal §2.1 updated; Table 4.2 has the new row.

**Fallback (Option B, if Option A stalls before the freeze):** prose surgery — scope
the claim to "regime latent b in the transition + relational code cₜ in
imagination-time readout" and state the design rationale. Cheap, defensible, weaker.

### ⚠ BLOCKER — two facts found while reading the code (need user sign-off before coding)

1. **Option A reverses a DELIBERATE, documented design decision.** `cfg.py` (cSRSSM
   docstring) states the context b "conditions the RSSM transition (unlike GAT/DALI,
   which only enter the feature)" explicitly for a **realtime budget** reason:
   keeping the GAT out of the transition avoids a per-imagined-step decode→GAT→
   transition loop. Threading cₜ into `img_step` adds exactly that loop to every
   imagined step of every training update AND every deploy tick. This is not fixing
   an oversight — it is trading realtime latency + imagination cost for the stronger
   novelty claim. The author already priced this and chose against it. The user
   should confirm they accept that cost.

2. **Unresolved temporal-alignment correctness question (posterior path).** `cₜ` is
   `GAT(peds_t, deter_t)` — but `deter_t` is produced BY the `observe()` scan, so on
   the posterior path cₜ cannot be precomputed like `embed` is; it must be computed
   INSIDE the scan from `prev_state["deter"]` and the correctly-lagged observed peds.
   Which crowd frame conditions which transition (cₜ₋₁ → transition into t, vs cₜ)
   is an off-by-one that, if wrong, still trains and still logs nonzero losses —
   i.e. silently produces a subtly-wrong model, the worst failure mode for a paper.
   Resolving it correctly needs either the author's stated intent or a derivation +
   a GPU byte-identity/equivalence test loop; it cannot be verified by inspection.

**Therefore:** do NOT blast a large speculative diff into the compiled hot path.
Recommended path: (a) user confirms the realtime-cost tradeoff is acceptable;
(b) settle the temporal-alignment convention explicitly (write it into this file);
(c) implement behind the config flag with the flag-off byte-identity test as the
first gate; (d) GPU-verify flag-on before any ablation run. If the user would rather
not pay the realtime cost or reopen a settled design decision, take Option B (prose)
— it is honest and reviewer-defensible on its own.

### DECISION (2026-07-09): user chose Option A ("do it carefully"). Realtime cost accepted.

### Temporal-alignment convention (THE correctness contract — settled before coding)

**Definition.** `cₜ = GAT(crowd_t, deter_t)` conditions the transition *out of* step t
(i.e. `img_step(state_t, action) → prior_{t+1}` is conditioned on `cₜ`). Equivalently:
the transition *into* step t is conditioned by `cₜ₋₁`, computed from step t-1's crowd
and deter — both of which live in `prev_state` when `img_step` runs. No forward leakage.

**Per path:**
- **Posterior / training (`observe`→`obs_step`→`img_step`):** `crowd_t` = the
  OBSERVED ground-truth ped node-set at t (`data["PedestrianNodeSetSpace"]`), SE(2)-
  canonicalized the same way the decoder targets are (anchor→current frame).
  `deter_t` = `prev_state["deter"]`. Because `deter_t` is produced by the scan, `cₜ`
  is computed INSIDE the scan lambda from `prev_state` + a per-step SHIFTED observed-
  peds input (`peds_scan[t] = observed_peds[t-1]`), not precomputed. On `is_first`,
  zero the gat_code (no prior crowd), mirroring how `prev_action` is zeroed.
- **Imagination (`_imagine`→`img_step`):** `crowd_t` = DECODED peds from `state_t`
  (teacher-forcing asymmetry, exactly like prior-vs-posterior). `cₜ` is ALREADY
  computed for the readout inside `_get_augmented_feat` (models.py ~662) — return it
  (`_return_c_t`) and pass into the same step's `img_step`; do not recompute.
- **Deploy (`dreamer.py:_policy`):** `crowd_t` = observed peds in `obs`; `deter_t` =
  current latent deter. `cₜ = GAT(observed_peds, deter)`, passed to `obs_step`.

**GAT/RSSM boundary.** The GAT lives on `WorldModel`, not `RSSM`. `img_step`/`obs_step`
take a `gat_code=` TENSOR kwarg (like `context=`), None when the flag is off. The
posterior path passes a `gat_code_fn` closure into `observe` (computed inside the scan
from prev_state + scanned peds); imagination passes the reused `cₜ` tensor directly.

**Byte-identity contract.** Flag off (`condition_transition: false`, default) ⇒
`gat_code` is None everywhere ⇒ `img_step`/`obs_step` take the existing branch ⇒
`_context_size`-only input width ⇒ outputs bit-identical to today. This is test gate #1.

### Task 5 = COMMIT (build H3). Concrete next steps (separate Phase-4 effort):
- `kl_surprise` is already exposed (`dreamer.py:_policy`, gated by
  `behavior.expose_kl_surprise`) — the observation signal is half-built.
- Unbuilt: the safety-layer policy logic that consumes `kl_surprise` + its eval
  protocol. Add both to `docs/icra2027/EXPERIMENTS.md`, schedule Phase 4 before freeze.
- This is research work, not a mechanical edit — own it as its own task, not part of
  the Task 1 code pass.

---

## Task 2 — Resolve the K = 6/8/16 three-way mismatch (code + config + prose)

**Problem.** Proposal says K=16 production; `SocialContextCfg.window` default is 8;
`context.py` docstring says "K<=8 for realtime budget"; smoke configs use 6.

**Decision to make first (one sentence of judgment, then mechanical):** pick the
production K. Recommendation: **K = 8** — it matches the code default, the docstring's
stated realtime budget, and `window < batch_length` is required (batch_length 16 in
smoke, so K=16 would leave zero out-of-window prediction targets and silently disable
`context_pred_loss`, the primary identifiability signal). K=16 in the proposal is
almost certainly a transcription error from `batch_length: 16`.

**What to do:**
1. `docs/social_dreamer_proposal.md` §2.1 (~line 89): change "K = 16" → "K = 8",
   and add the constraint sentence: K must be < batch_length so out-of-window
   prediction targets exist (`M = T_c − K_c + 1 ≥ 2`).
2. Verify `social_csrssm_config.yaml` (production config) sets `window: 8`; fix if not.
3. Add a fail-fast validation in `SocialContextCfg` (or where the trainer builds the
   model): raise if `window >= batch_length` when `pred_scale > 0`, with a message
   naming both values. This turns a silent loss-disable into a config error.
4. State the K=8 latency figure in the proposal's deployment-feasibility passage
   (§4.4-equivalent) if a measured number exists; otherwise mark TODO-measure.

**Done when:** grep for `16` near "window"/"K" in proposal returns only batch_length
references; config validator test exists (one pytest: window=16,batch_length=16,
pred_scale>0 → raises).

---

## Task 3 — Second held-out driver (cheap, high evidentiary value)

**Problem.** Every generalization claim is bounded by one OOD driver (HSFM).
Reviewers weight single-OOD-point evidence harshly on a paper whose central claim is
cross-regime generalization. The proposal itself notes adding one costs "eval episodes
only, no training compute."

**What to do:**
1. Registry checked (2026-07-09): humansim ships `sfm`, `hsfm`, `orca`, `straight`,
   **`socialgail`** (GAIL-learned policy, HF weight fetch), **`nsp`** (Neural Social
   Physics, learned). Training drivers are SFM+ORCA; held-out #1 is HSFM.
   **Recommendation: `socialgail` as held-out #2.** Rationale: it is a *learned,
   data-fitted* driver — structurally disjoint from the entire force-based family —
   which directly upgrades the proposal's own stated weakness ("a real-data-fitted
   held-out driver would support a stronger realism claim", §held-out note). NSP is
   the alternate if socialgail's weights/runtime prove flaky in eval; verify both
   load (`weights fetch` path in `local_planner/socialgail/planner.py`) before
   committing to one in prose. A parameter-perturbed SFM variant is NOT acceptable
   (parameter interpolation, which the proposal explicitly disclaims).
2. Add the second driver to the eval matrix in `docs/icra2027/EXPERIMENTS.md`:
   same seeds, same episode counts, report SR/worst-case-SR alongside HSFM.
3. One sentence in proposal §7 upgrading "one held-out driver" limitation to two.

**Done when:** EXPERIMENTS.md eval matrix has the second driver row; the driver is
selectable via existing ROS param (verified by a dry `arena launch human:=hunav` with
the module param set); proposal limitation text updated.

---

## Task 4 — Promote identifiability diagnostics to a results subsection (prose only)

**Problem.** Linear probe, counterfactual consistency, and prediction-floor gap are
framed as a Phase-1 go/no-go gate. If they pass, that is itself a citable finding
("b is empirically identifiable; meta-RL context variables usually collapse") and it
costs zero compute to present it as one.

**What to do:**
1. Add a results subsection skeleton to the proposal (or EXPERIMENTS.md): probe
   accuracy vs. chance, counterfactual divergence vs. regime distance, and
   `context_pred_gap` (already logged in metrics.jsonl — the trained loss vs. the
   zero-velocity pooled baseline `context_pred_floor`, wired in metrics as
   `context_pred_raw/floor/gap`) over training.
2. Specify one figure: x = driver-pair regime distance, y = counterfactual
   divergence, plus a probe-accuracy bar inset. Placeholder until Phase 1 runs.
3. Keep the gate function — the same numbers serve both purposes.

**Done when:** subsection exists with figure spec and named metric keys matching what
the code already logs.

---

## Task 5 — H3 commit-or-cut decision (decision, then prose)

**Problem.** H3 (uncertainty-aware safety layer, Phase 4) is a stretch contribution
whose machinery is unbuilt. Half-built stretch content in the main text reads as
padding.

**What to do:** this is a user decision, not an agent decision. Present the two
options with cost estimates (Phase-4 compute + `kl_surprise` is already exposed in
`dreamer.py:_policy` via `expose_kl_surprise`, so the plumbing is ~half done — the
unbuilt part is the safety-layer logic and its evaluation) and get an explicit call.
If CUT: move H3 to future work, delete from hypothesis table, keep `kl_surprise`
plumbing (it's gated off by default and costs nothing). If COMMIT: schedule Phase 4
before the freeze date and add its eval protocol to EXPERIMENTS.md.

**Done when:** user has decided; prose reflects the decision; no half-committed state.

---

## Task 6 — EvoNav (automated reward design, 2026) preemption sentence (trivial)

Add one sentence to Related Work or Limitations: reward *shape* is orthogonal to the
contribution (world-model structure); cite EvoNav as the current automated-reward-
design SOTA and note the TGRF-derived composite (see
`configs/social_csrssm_smoke_peds_verify.yaml` reward comment — TGRF ratios, r_pred
omitted because the world model predicts implicitly) was chosen from published
human-validated ratios, not searched. Done when the sentence exists with the citation.

---

## Priority order

| # | Task | Cost | Risk closed |
|---|------|------|-------------|
| 1 | GAT-conditioned transition (Option A revised) | Med-High | Highest — makes the core novelty claim true instead of weakening it |
| 2 | K mismatch + fail-fast validator | Low | High — silent loss-disable bug class + credibility ding |
| 3 | Second held-out driver | Low-Med | High — single-OOD-point generalization evidence |
| 4 | Identifiability results subsection | Low | Medium — free citable content |
| 5 | H3 commit-or-cut | Decision | Medium — vaporware risk |
| 6 | EvoNav sentence | Trivial | Low |

Tasks 2, 4, 6 are safe for immediate agent execution. Task 1 is the main code task
(config-gated, byte-identity regression guard required). Task 3's registry check is
done — remaining work is weight-load verification + eval-matrix prose. Task 5 blocks
on the user.

## Revision notes vs. the earlier draft

- **Naive Option A ("feed decoded-peds cₜ into img_step") was rejected**, then
  superseded by the revised design in Task 1: observed-peds cₜ on the posterior path
  (genuine out-of-latent information, like b) + decoded-peds cₜ on the prior path
  (teacher-forcing asymmetry), config-gated for byte-identical ablations. The naive
  form remains rejected — decoded-only conditioning is information-free.
- **Old Tier-1 premise "b and dₜ reach img_step but the paper doesn't support it"**
  was false — b's transition conditioning is implemented and correctly described in
  §2.1. Task 1's scope is adding cₜ, not b.
