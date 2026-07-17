# Social-Dreamer ICRA 2027 — Runbook

Single source of truth for **what to run, in what order, with which command**.
Structure/rationale lives in `../social_dreamer_proposal.md` (§4) and `EXPERIMENTS.md`;
this file is the operational sequence. Status 2026-07-16.

Execution order: **Phase T (tuning) → Gate → Tier 1 → Gate → Tier 2 → Eval matrix**.

---

## Phase T — Hyperparameter tuning (base params for ALL ablations)

Two-stage Optuna search (`configs/tuning/`). Stage A finds context/identifiability
params (kl/pred scales, b_dim, window); Stage B finds the shared core (train_ratio,
lrs, entropy, imag_horizon) + GAT dims. Winners are frozen into
`social_csrssm_config.yaml` and inherited by every ablation config — tune ONCE on the
full system, carry everywhere.

**Critical mechanics (why the configs look the way they do):**

- `tune_agent.py` **never advances the task curriculum** (trials must be
  difficulty-comparable), so the base config's `starting_stage` is the difficulty for
  the entire trial. The real config starts at Stage 0 = empty world → zero pedestrians
  → Stage A's `context_pred_gap` ≈ 0 (every trial health-pruned) and Stage B measuring
  empty-world driving. Therefore both tuning YAMLs point at
  **`social_csrssm_tuning_base.yaml`** — identical to the real config but pinned at
  **Stage 2** (moderate crowd + doorways: enough social pressure to discriminate,
  below the Stage-3 density cap where collision noise dominates).
- The sim is **persistent across all trials** in a study: launch the runtime once,
  `tune_agent.py` spawns `n_envs` (=4) envs itself and reuses them
  (`ensure_envs_healthy` respawns on failure).
- Set `agents_dir` in the tuning YAML (or accept the default) — with it set, the study
  gets sqlite storage (resumable) and a shared prefill cache across trials.

**Map choice:** tune on **training-distribution worlds only** — never `hospital_1`
(held-out layout; tuning on it leaks the holdout into hyperparameter selection and
invalidates the OOD claim).

- Stage A → `map_empty`: the signal is pedestrian-regime identifiability; geometry is
  irrelevant, and empty is the cheapest sim.
- Stage B → `office_1`: success-composite depends on geometry; office is the
  representative deployment class and is a *training* world.

**Commands** (host, one terminal each; container `arena-arena_ws-arena-1` equivalent:
`docker exec -it arena-arena_ws-arena-1 bash -lc 'source /opt/arena_ws/source && …'`):

```bash
# ── Stage A (context params, ~30 trials × 200k steps) ────────────────────
# terminal 1: persistent sim (leave running for the whole study)
arena runtime sim:=gazebo world:=map_empty headless:=true

# terminal 2: the study (resumable via sqlite if agents_dir is set)
ros2 run arena_training tune_agent.py \
    --config /opt/arena_ws/src/Arena/arena_training/configs/tuning/social_stage_a.yaml

# ── freeze Stage-A winners ────────────────────────────────────────────────
# copy best-trial values (optuna dashboard / study sqlite) into BOTH
# social_csrssm_config.yaml and social_csrssm_tuning_base.yaml (keep in lockstep)

# ── Stage B (core + GAT params, ~40 trials × 300k steps) ─────────────────
# terminal 1:
arena runtime sim:=gazebo world:=office_1 headless:=true
# terminal 2:
ros2 run arena_training tune_agent.py \
    --config /opt/arena_ws/src/Arena/arena_training/configs/tuning/social_stage_b.yaml

# ── freeze Stage-B winners into social_csrssm_config.yaml ────────────────
```

Notes:
- `--n-trials N` overrides the YAML's trial count (useful for a 2-trial pilot first).
- If pedestrians don't appear in tuning envs, append `human:=hunav` to
  `per_env_launch_args` in `scripts/tune_agent.py` (~line 490) — env-level arg, not a
  runtime arg.
- Propagate the frozen winners to the ablation variants (`*_no_context_config`,
  `social_gat_feat_config`) — same values, only the ablation switch differs.

**Gate T→1 (Phase-1 diagnostics, one pilot run):** linear probe b→driver > chance
(`scripts/probe_context.py`), `context_pred_gap` > 0 sustained, counterfactual b-swap
divergence scales with regime distance. No Tier-1 spend before this passes.

---

## Tier 1 — paper spine (6 configs × 3 seeds = 18 runs, ≈330 A100-h)

**Seed-escalation protocol:** run 3 seeds/cell. Add seeds 4–5 ONLY to cells whose
worst-case-SR IQM CIs overlap the headline's (max +4 runs → 22). Never 5 seeds
everywhere up-front.

Training command (per cell × seed):

```bash
arena train sim:=gazebo mobile:=rosnav_rl train_config:=<config> # seed via config/env
```

| # | Cell (Table 4.2) | GAT | DALI | Context | config |
|---|---|---|---|---|---|
| 1 | DreamerV3 (DR) — flat anchor | – | – | – | `dreamer_training_config.yaml` (DR variant) |
| 2 | GAT-Dreamer ("neither") | ✓ | – | – | `social_csrssm_no_context_config.yaml` (dali off) |
| 3 | GAT-cSRSSM (context only) | ✓ | – | ✓ | `social_csrssm_config.yaml` with `dali.enabled: false` |
| 4 | GAT-DALI (DALI only) | ✓ | ✓ | – | `social_csrssm_no_context_config.yaml` with `dali.enabled: true` **(config variant to add)** |
| 5 | GAT-DALI-cSRSSM — **headline** | ✓ | ✓ | ✓ | `social_csrssm_config.yaml` with `dali.enabled: true` **(config variant to add)** |
| 6 | HEIGHT — model-free anchor (cheap) | n/a | n/a | n/a | HEIGHT baseline setup (see proposal §4.2) |

All GAT cells run **transition-conditioned GAT** (`gat.condition_transition: true`) —
that is the C1 claim; readout-only is a Tier-2 attribution row. Training uses the
2-driver DR set {ORCA, de-tuned SFM} sampled per episode (**build**: driver sampler,
EXPERIMENTS.md §5.1). Worlds: `office_1` + `map_empty`/parametrized per EXPERIMENTS §1.

**Gate 1→2:** headline vs "neither" separated on worst-case SR; identifiability
diagnostics still green on headline checkpoints.

---

## Tier 2 — gated diagnostics (≤8 runs, ≈145 h, 2–3 seeds each)

| Row | Purpose | Seeds |
|---|---|---|
| DreamerV3 (single driver, ORCA) | what does DR alone buy | 2 |
| DALI-Dreamer (DR, no GAT) | does dₜ need topology | 2 |
| Full system, readout-only GAT (`gat.condition_transition: false`) | C1 attribution: transition vs readout pathway | 3 |
| TSSM backbone (full system) | C4 backbone axis | 2–3 |

---

## Evaluation matrix (episodes only — no training compute)

Every Tier-1/2 checkpoint × driver × world × scenario suite (EXPERIMENTS.md §2):

| Axis | Values |
|---|---|
| Drivers in-dist | `sfm`, `orca` |
| Drivers held-out | `hsfm` (analytical OOD), `socialgail` (learned, data-fitted) |
| Drivers appendix-reserve | `nsp`, `sarl`, `cadrl`, `dsrnn`, `drlvo` (headline ckpt only, if requested) |
| Worlds | `office_1` (seen), `hospital_1` (held-out layout) |
| Scenarios | 5 `robot_*` geometries × 3 densities × 50 seeds |

```bash
arena launch sim:=gazebo world:=hospital_1 mobile:=rosnav_rl mobile.agent:=<ckpt> \
      tm_robots:=scenario tm_obstacles:=scenario
# driver = humansim local_planner ROS param: orca|sfm|hsfm|socialgail|…
```

Verify `socialgail`/`nsp` weight-fetch once before the eval batch (registry import
already verified 2026-07-09; instantiation fetches weights).

---

## Current blockers (from EXPERIMENTS.md §5 + 2026-07-17 decisions)

1. Per-episode driver sampler + de-tuned-SFM ParamDist offsets (**training blocker**).
2. ORCA `ParamDist` (time horizon, neighbor distance) — SFM has it, ORCA is a fixed
   point in regime space; b needs both families continuous (proposal §2.3).
3. Robot = **Jackal, v_lin capped 1.0 m/s** (proposal §4.1) — cap must be enforced in
   the training action space, not just prose.
4. Tier-1 cells 4/5 need the two `dali.enabled: true` config variants (trivial copies).
5. Episode-record → parquet pipeline in `arena_evaluation` (all quantitative figures).
6. Scenario density variants (YAML cloning).

Items 1–4 are specced for delegation in
`~/.claude/plans/you-are-an-expert-quizzical-porcupine.md` (Sonnet backlog).
