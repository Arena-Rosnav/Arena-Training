# Social-Dreamer ICRA 2027 — Experiment Execution Plan

Companion to `main.tex`. Everything here maps to concrete Arena machinery that exists
today; items marked **(build)** are the pieces still to be written. Status 2026-07-07.

---

## 1. Maps

**Training (two worlds, both axes of clutter):**

| World | Why |
|---|---|
| `office_1` | Structured indoor layout — rooms, corridors, doorways. The "typical deployment" geometry class. |
| `map_empty` + `tm_obstacles:=parametrized` | Procedurally randomized static obstacles per episode. Prevents layout memorization; cheap diversity. |

**Evaluation:**

| World | Role |
|---|---|
| `office_1` | Seen layout — isolates the driver axis (behavior generalization with layout held fixed). |
| `hospital_1` | **Held-out layout** — long corridors, wide halls, door bottlenecks. Natural pedestrian-contention geometry. Never in training. |

Rationale: two independent holdout axes (layout, driver) reported as separate cells,
never pooled. `hospital_2`, `airport`, `reception` stay in reserve for a
reviewer-requested robustness appendix — they cost eval episodes only.

All worlds live in `arena_simulation_setup/worlds/`; no new assets needed.

---

## 2. Scenario Structure

Two layers, both already in the stack:

**(a) humansim scenario YAMLs** (`humansim/arena_humansim/config/scenarios/`) — fix
pedestrian spawn poses, goal sequences, velocities, seed. Schema per file:

```yaml
name: robot_simple_crossing
simulation: {seed: 42, dt: 0.05, max_ticks: 2000, ...}
agents:
  - agent_id: 0
    spawn_pose: {x: 10.0, y: 0.0, theta: 3.14159}
    goal_sequence: [{x: -10.0, y: 0.0}]
    waypoint_mode: reverse
    desired_velocity: 1.2
  # ...
```

**Evaluation suite = the existing `robot_*` scenarios:**

| Scenario | Stress pattern |
|---|---|
| `robot_simple_crossing` | 8 peds converge in star pattern; robot crosses the focus |
| `robot_corridor` | bidirectional flow, passing convention |
| `robot_bottleneck` | door contention, yielding |
| `robot_t_junction` | occluded merging |
| `robot_queue` | standing group + stragglers, static-social mix |

**Scenario axes** for the paper grid: geometry class (5 above) × pedestrian density
(low/mid/high — clone each YAML at 4/8/12 agents) × driver (ORCA/SFM/HSFM) × seed
(50). Density variants **(build)**: mechanical YAML cloning, ~1 h scripting.

**(b) task_generator episode axes** — `tm_robots:=scenario` / `tm_obstacles:=scenario`
replays fixed scenarios; `tm_robots:=random` + `tm_obstacles:=parametrized` for
training and free-roam eval. Driver selection is a runtime ROS param on humansim
(local_planner plugin: `orca` / `sfm` / `hsfm`), sampled per episode during training
— **(build)**: per-episode driver sampler + ParamDist offsets for the de-tuned SFM,
recorded in the experiment config.

---

## 3. Required Visualizations

Plotting deps already in root venv (`matplotlib`, `seaborn`, `plotly`, `polars`);
bag reading via `rosbags`/`mcap` (no ROS needed). Landing package:
`arena_evaluation` (currently empty — this is its charter). humansim's
`utils/renderer.py` (matplotlib top-down) covers the qualitative figures.
Per-figure map (numbers = paper figures in `main.tex`):

| # | Figure | Answers | Tool |
|---|---|---|---|
| 1 | Worst-case SR per method, IQM + 95% bootstrap CI (rliable-style) | H1a/H1b headline | matplotlib + rliable protocol |
| 2 | Paired SR drop SFM→HSFM per method | H2 | matplotlib paired dots |
| 3 | Linear-probe accuracy b→driver vs chance | C2 identifiability | existing `scripts/probe_context.py` output |
| 4 | Counterfactual b-swap divergence vs regime distance | counterfactual imagination | existing `social/counterfactual.py` output |
| 5 | 2D embedding (t-SNE/UMAP) of d_t per driver, HSFM highlighted | H2 mechanism | sklearn + matplotlib |
| 6 | Persistence-floor gap over training | collapse alarm | wandb export → polars → matplotlib |
| 7 | Frozen-WM transfer sample-efficiency curves | H4 | wandb export |
| 8 | Trajectory overlays (crossing, bottleneck): robot path + ped tracks, full system vs flat DreamerV3, matched seeds | qualitative | humansim `renderer.py` |
| — | Calibration curve: empirical false-alarm vs α; trigger-comparison bars | H3 | existing `safety.py` calibration output |
| — | Backbone throughput bars + TSSM imagination memory vs horizon | C4 | benchmark harness pattern (`utils/benchmark`) |

**(build)**: one episode-record → parquet pipeline (mcap in, polars frame out:
per-step robot pose, ped states, driver label, per-episode metrics). Every figure
above reads that one format. Estimate: the single biggest missing piece, ~2–3 days.

---

## 4. How to Run It with Arena

```bash
# training (per Tier-1 cell; config selects ablation)
arena train sim:=gazebo mobile:=rosnav_rl train_config:=social_csrssm_config.yaml
# context-off cell: train_config:=social_csrssm_no_context_config.yaml
# smoke first: social_csrssm_smoke.yaml (see gazebo-train-smoke skill)

# evaluation episodes (per driver × scenario × world)
arena launch sim:=gazebo world:=hospital_1 mobile:=rosnav_rl mobile.agent:=<ckpt> \
      tm_robots:=scenario tm_obstacles:=scenario
# driver switch = humansim local_planner param (orca|sfm|hsfm)

# record for analysis
# mcap recording of /arena state topics + ped AgentStates + cmd_vel (build: topic list)
```

Compute: Tier 1 = 18 runs ≈ 330 A100-h (3 seeds/cell + CI-triggered escalation to 5,
worst case 22 runs; see proposal §4.2); Tier 2 ≤8 runs ≈ 145 h (RunPod A100 40 GB,
~18 h/run). Gates before spending: Phase-1 diagnostics (probe > chance, floor gap > 0)
on one pilot run — machinery exists (`probe_context.py`, floor logging, counterfactual
eval). **Operational sequence + all commands: `RUNBOOK.md` (tuning → tiers → eval).**

## 5. Build-List Summary (ordered)

1. Per-episode driver sampler + de-tuned SFM ParamDist config (training blocker).
2. Episode-record → parquet pipeline in `arena_evaluation` (all quantitative figures).
3. Scenario density variants (YAML cloning).
4. Metrics module: SR/collision/SPL/TTG/NMR/PSVR/d_min/jerk from parquet.
5. Figure scripts 1–8 (thin, on top of 2+4).
6. Ensemble dynamics heads (H3 stretch only; conformal calibration already exists).
```
