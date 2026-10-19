# Arena Training

ROS 2 package implementing the Deep Reinforcement Learning training pipeline for Arena-Rosnav. It wraps the [rosnav_rl](deps/rosnav_rl) framework and provides the gym environments, trainer classes, configuration schemas, and scripts needed to train and evaluate navigation agents.

## Package Structure

```
arena_training/
├── agents/                        # Trained agent storage (training_config.yaml + best_model.zip)
├── arena_training/
│   ├── arena_rosnav_rl/           # Arena-specific wiring (cfg, trainer, node, tools)
│   └── environments/              # Gym environments (Gazebo, Flatland, …)
├── deps/
│   └── rosnav_rl/                 # Git submodule — rosnav_rl framework
├── scripts/
│   ├── train_agent.py             # Main training entry point
│   └── test_agent.py              # Pipeline smoke-test (no full training)
├── pyproject.toml                 # Python deps managed by uv
├── package.xml
└── CMakeLists.txt
```

## Installation

The canonical way is through the Arena workspace tooling, which handles submodule init, Python deps, and colcon build in one step:

```bash
cd ~/arena5_ws     # replace with your actual workspace path
source arena
arena feature training install
```

### Manual installation

```bash
# 1. Initialize the rosnav_rl submodule (shallow clone to avoid fetching large history)
cd /path/to/arena5_ws/src/Arena/arena_training
git submodule update --init --depth 1 deps/rosnav_rl

# 2. Install Python dependencies into the Arena venv
cd /path/to/arena5_ws/src/Arena
uv pip install -e arena_training -e arena_training/deps/rosnav_rl/rosnav_rl

# 3. Build ROS 2 packages
cd /path/to/arena5_ws
arena build
source arena
```

## Usage

### Launching the full stack (recommended)

`arena train` brings up arena and training in one step. It runs [`launch/training.launch.py`](launch/training.launch.py), which includes `arena_bringup/launch/arena_runtime.launch.py` (runtime-only, no auto-spawn) and starts `train_agent.py`. The training script reads `n_envs` from the config YAML and spawns envs via `/arena/spawn_env`, each with `task.episode.spawn.robot_clearance` set to `arena_cfg.general.safety_distance`, so a robot standing at its spawn is clear of the safe-distance penalty.

```bash
# Config name resolved from arena_training/configs/
arena train sim:=gazebo robot.mobile:=rosnav_rl train_config:=dreamer_training_config.yaml

# Or absolute path
arena train sim:=gazebo robot.mobile:=rosnav_rl train_config:=/path/to/dreamer_training_config.yaml
```

All launch args (`sim`, `world`, `robot`, `robot.mobile`, `robot.mobile.agent`, ...) flow through to `arena_runtime.launch.py` via [`IncludeLaunchDescriptionForward`](../arena_bringup/arena_bringup/actions.py). Fleet size is controlled by `arena_cfg.general.n_envs` in the YAML.

Gym steps are paced on the sim clock at `arena_cfg.general.control_hz`. With `arena_cfg.general.lockstep: true` the envs hold the sim instead and a gym step advances it by `arena_cfg.general.lockstep_step_seconds` through `/arena/sim_lifecycle/step`. With several envs every env's action is applied, the sim is stepped once and then every env observes, in the SB3 vec env and in the DreamerV3 `simulate` loop alike.

`agent_config.action_chunk: H` (SB3 only) makes the policy emit H commands per decision, run one per control tick by `VecActionChunk`. `total_timesteps`, `eval_freq` and `max_num_moves_per_eps` stay in control ticks, PPO discounts with gamma^H and gae_lambda^H per decision, and an env whose episode ends inside a chunk idles for the rest of it. The inference node plays a chunk out the same way. `agent_config.mirror: canonical | random` reflects observations and commands across the robot's forward axis (nearest pedestrian on the left, or whole episodes with probability one half). `arena_cfg.general.env_seed` gives env i the episode seed stream `run_seed:=<env_seed>-<i>`, and `arena_cfg.general.env_launch_args` gives env i entry i (cycled) of a list of extra launch args, such as its world or pedestrian driver.

`arena train` is sugar for `arena feature training launch ...`. You can visualize the running simulation via `arena viz --all`.

See [`arena_bringup`](../arena_bringup) for all available launch arguments. For agent, observation space, reward and curriculum configuration refer to the [rosnav_rl README](deps/rosnav_rl/README.md).

### Standalone launch file

If you want to bypass the `arena` CLI (e.g. when running inside an already-set-up environment, or driving from another launch file), invoke the launch file directly:

```bash
ros2 launch arena_training training.launch.py sim:=gazebo robot.mobile:=rosnav_rl train_config:=dreamer_training_config.yaml
```

Same arg surface as `arena train` — the CLI verb is just a thin wrapper.

### Training script (standalone)

For development without a simulator (no env spawning, no service calls), invoke the script directly. This runs the trainer logic against whatever envs already exist; `train_agent.py` will fail at `/arena/spawn_env` if arena_node is not running.

```bash
# Default config (sb_training_config.yaml)
ros2 run arena_training train_agent.py

# Specific config by name — resolved from arena_training/configs/
ros2 run arena_training train_agent.py --config dreamer_training_config.yaml

# Absolute path
ros2 run arena_training train_agent.py --config /path/to/my_config.yaml
```

Config resolution order:
1. Absolute path — used directly
2. Relative path from current working directory
3. Name only — looked up in `arena_training/configs/`

Training metrics are logged to **Weights & Biases** automatically. Trained agents are saved to `agents/<agent_name>/` (`training_config.yaml` + `best_model.zip`).

### Available training configs (`arena_training/configs/`)

| File | Description |
|---|---|
| `sb_training_config.yaml` | Stable Baselines3 — PPO / SAC / TD3 (default) |
| `dreamer_training_config.yaml` | DreamerV3 model-based RL |
| `observations/observations.yaml` | Observation space definitions |

### Agent output directory

Trained agents (` training_config.yaml` + `best_model.zip`) are saved under `arena_training/agents/<agent_name>/`. Resolution order:

1. `agents_dir` field in the training config YAML (highest priority)
2. `ROSNAV_AGENTS_DIR` environment variable
3. Default: `arena_training/agents/`

### Deployment

Trained agents are loaded by the `rosnav_rl` action server at inference time. See the [rosnav_rl README](deps/rosnav_rl/README.md) for deployment instructions.

Quick test without a simulator:
```bash
ros2 run arena_training test_agent
```

## MCP domain

`arena_training/mcp.py` contributes the `training` domain to the Arena MCP server (`python3 -m arena_mcp ls training` lists its tools). Configs are named as in `arena train` (a file in the share `configs/` dir, or an absolute path) and every config goes through `load_training_config`, so the listing, the parsed `TrainingCfg` JSON with its schema (also the resource `arena://training/schema`) and validation of a file or of YAML text report what training would see. A relative `observations_config` in YAML text resolves against the share configs dir unless `base_dir` is given. Starting training runs `arena train train_config:=<config> [key:=value ...]` as a job and is refused while `/arena/register_env` answers, because training brings up its own runtime. `jobs_cancel` stops it gracefully. Agents are read from `ROSNAV_AGENTS_DIR`, else the arena_training default, with their checkpoints, the latest tensorboard scalars and the log tail of the training job that wrote them.

## Supported RL Frameworks

| Framework | Algorithms |
|---|---|
| **Stable Baselines3** | PPO, SAC, TD3, A2C, DDPG |
| **DreamerV3** | Model-based RL with world models |

## Configuration Structure

Configs are Pydantic-validated YAML. Top-level keys:

```yaml
arena_cfg:       # simulation / task / robot / monitoring settings
agent_cfg:       # rosnav_rl agent: framework, observation spaces, action space
```

## Troubleshooting

**`rosnav_rl` not found**
```bash
cd /path/to/arena5_ws/src/Arena/arena_training
git submodule update --init --depth 1 deps/rosnav_rl
cd /path/to/arena5_ws/src/Arena
uv pip install -e arena_training -e arena_training/deps/rosnav_rl/rosnav_rl
```

**Config file not found**
```bash
# Configs ship with arena_training — list available ones:
ls /path/to/arena5_ws/src/Arena/arena_training/configs/
```
Pass the file name only (e.g. `dreamer_training_config.yaml`) and it will be found automatically, or supply an absolute path.

## Related Packages

- [`arena_bringup`](../arena_bringup) — launch files and training configs
- [`arena_simulation_setup`](../arena_simulation_setup) — simulation environment setup
- [`task_generator`](../task_generator) — dynamic task generation
- [`arena_evaluation`](../arena_evaluation) — evaluation utilities
- [`rosnav_rl`](deps/rosnav_rl) — core RL framework (submodule)
