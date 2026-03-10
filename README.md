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
# 1. Initialize the rosnav_rl submodule
cd /path/to/arena5_ws/src/Arena/arena_training
git submodule update --init --recursive

# 2. Install Python dependencies (uses uv)
cd /path/to/arena5_ws/src/Arena
uv sync

# 3. Build ROS 2 packages
cd /path/to/arena5_ws
colcon build --packages-up-to arena_training
source install/setup.bash
```

## Usage

### Launching the full stack (recommended)

The easiest way to start training is through `arena launch`. Providing `train_config` automatically implies `train_mode:=true` and launches `train_agent.py` in parallel with the simulation:

```bash
# Start simulation + training together (train_mode implied)
arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 \
    train_config:=/path/to/dreamer_training_config.yaml

# Or with a config name resolved from arena_bringup/configs/training/
arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 \
    train_config:=dreamer_training_config.yaml

# Start simulation only in train_mode (no trainer process)
arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 train_mode:=true
```

> **Note:** `train_config` and `train_mode` are independent - `train_config` sets up the trainer process, while `train_mode` controls simulation behaviour (direct `cmd_vel` publishing, nav2 controller silenced). Providing `train_config` sets `train_mode` to `true` automatically.

See [`arena_bringup`](../arena_bringup) for all available launch arguments. For agent, observation space, reward and curriculum configuration refer to the [rosnav_rl README](deps/rosnav_rl/README.md).

### Training script (standalone)

```bash
# Default config (sb_training_config.yaml)
ros2 run arena_training train_agent

# Specific config by name — resolved from arena_bringup/configs/training/
ros2 run arena_training train_agent --config dreamer_training_config.yaml

# Absolute path
ros2 run arena_training train_agent --config /path/to/my_config.yaml
```

Training metrics are logged to **Weights & Biases** automatically. Trained agents are saved to `agents/<agent_name>/` (`training_config.yaml` + `best_model.zip`).

### Available training configs (`arena_bringup/configs/training/`)

| File | Description |
|---|---|
| `sb_training_config.yaml` | Stable Baselines3 (default) |
| `dreamer_training_config.yaml` | DreamerV3 |
| `semantic_training_config.yaml` | Semantic observations |
| `unity_rgbd_training_config.yaml` | Unity RGB-D |

### Deployment

Trained agents are loaded by the `rosnav_rl` action server at inference time. See the [rosnav_rl README](deps/rosnav_rl/README.md) for deployment instructions.

Quick test without a simulator:
```bash
ros2 run arena_training test_agent
```

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
cd arena_training && git submodule update --init --recursive
cd /path/to/arena5_ws/src/Arena && uv sync
```

**Config file not found**
```bash
# Verify arena_bringup is installed and the file exists
ls $(ros2 pkg prefix arena_bringup)/share/arena_bringup/configs/training/
```

## Related Packages

- [`arena_bringup`](../arena_bringup) — launch files and training configs
- [`arena_simulation_setup`](../arena_simulation_setup) — simulation environment setup
- [`task_generator`](../task_generator) — dynamic task generation
- [`arena_evaluation`](../arena_evaluation) — evaluation utilities
- [`rosnav_rl`](deps/rosnav_rl) — core RL framework (submodule)
