# Arena Training Package

This package implements the Deep Reinforcement Learning (DRL) training pipeline for Arena-Rosnav using the rosnav_rl framework.

## Overview

The `arena_training` package provides the main entry point for training DRL agents in the Arena-Rosnav simulation environment. It supports multiple RL frameworks including:

- **Stable Baselines3** - Popular RL algorithms (PPO, SAC, TD3, etc.)
- **DreamerV3** - Model-based RL algorithm

## Package Structure

```
arena_training/
├── arena_training/          # Python package directory
│   └── __init__.py
├── deps/                    # Dependencies
│   └── rosnav_rl/          # Git submodule - rosnav_rl framework
├── resource/               # Package resources
│   └── arena_training
├── scripts/                # Executable scripts
│   └── train_agent.py     # Main training script
├── CMakeLists.txt         # Build configuration
├── package.xml            # ROS 2 package manifest
├── setup.py               # Python package setup
├── setup.cfg              # Python package configuration
└── README.md              # This file
```

## Dependencies

### ROS 2 Dependencies
- `rclpy` - ROS 2 Python client library
- `rclcpp` - ROS 2 C++ client library
- `geometry_msgs` - Geometry message types
- `nav_msgs` - Navigation message types
- `sensor_msgs` - Sensor message types
- `tf2`, `tf2_geometry_msgs`, `tf2_ros` - Transform library

### Python Dependencies
- `torch` - PyTorch for deep learning
- `stable-baselines3` - RL algorithms (if using SB3 framework)
- `gymnasium` - OpenAI Gym interface
- `numpy` - Numerical computing
- `pyyaml` - YAML configuration parsing

### Arena Dependencies
- `rl_utils` - Arena RL utilities package (required)
- `rosnav_rl` - Core RL framework (included as submodule)
- `arena_bringup` - Launch files and configurations

## Installation

### Quick Installation

The easiest way to install arena_training with all dependencies is through the main Arena workspace:

```bash
cd /home/le/arena5_ws/src/Arena

# Install with training dependencies
uv pip install -e ".[training]"

# Install rosnav_rl submodule
cd arena_training/deps/rosnav_rl/rosnav_rl
uv pip install -e .

# Build the ROS 2 packages (this will build rl_utils and arena_training)
cd /home/le/arena5_ws
colcon build --packages-up-to arena_training
source install/setup.bash
```

For detailed installation instructions, dependency management, and troubleshooting, see [INSTALL.md](INSTALL.md).

**Note:** The `arena_training` package depends on:
- `rl_utils` package (will be built automatically by colcon)
- `rosnav_rl` framework (must be installed from submodule as shown above)

### 1. Initialize Git Submodules

The package uses `rosnav_rl` as a git submodule. Initialize it with:

```bash
cd /path/to/arena5_ws/src/Arena/arena_training
git submodule update --init --recursive
```

Or if you've already cloned the Arena repository with submodules:

```bash
cd /path/to/arena5_ws/src/Arena
git submodule update --init --recursive
```

### 2. Build the Package

```bash
cd /path/to/arena5_ws
colcon build --packages-select arena_training
source install/setup.bash
```

## Usage

### Basic Training

To start training with the default configuration:

```bash
ros2 run arena_training train_agent.py
```

### Training with Custom Configuration

To use a specific configuration file:

```bash
ros2 run arena_training train_agent.py --config training_config.yaml
```

To use an absolute path to a configuration file:

```bash
ros2 run arena_training train_agent.py --config /path/to/custom_config.yaml
```

### Configuration Files

Configuration files are stored in `arena_bringup/configs/training/`. The package will automatically look for configuration files in this location.

Available configurations:
- `training_config.yaml` - Default training configuration
- `semantic_training_config.yaml` - Semantic observation training
- `unity_rgbd_training_config.yaml` - Unity RGBD training

## Configuration Structure

Training configurations are YAML files with the following structure:

```yaml
arena_cfg:
  # Arena-specific configuration
  general:
    # General training parameters
  robot:
    # Robot configuration
  task:
    # Task configuration
  monitor:
    # Monitoring configuration

agent_cfg:
  # Agent configuration
  framework:
    name: "stable_baselines3"  # or "dreamer_v3"
    algorithm:
      # Algorithm-specific parameters
  observation_space:
    # Observation space configuration
  action_space:
    # Action space configuration
```

## Training Pipeline

The training pipeline follows these steps:

1. **Initialization** - Initialize ROS 2 and parse arguments
2. **Configuration Loading** - Load training configuration from YAML
3. **Trainer Creation** - Create the appropriate trainer (SB3 or DreamerV3)
4. **Environment Setup** - Set up training and evaluation environments
5. **Agent Setup** - Initialize the RL agent with the specified algorithm
6. **Training Loop** - Execute the training process
7. **Monitoring** - Log metrics and save checkpoints

## Supported RL Frameworks

### Stable Baselines3
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

### DreamerV3
- Model-based reinforcement learning
- World model learning
- Latent imagination

## Troubleshooting

### Import Errors

If you encounter import errors for `rosnav_rl` or `rl_utils`:

1. Ensure the git submodule is initialized:
   ```bash
   cd arena_training
   git submodule update --init --recursive
   ```

2. Verify that `rl_utils` is built and sourced:
   ```bash
   colcon build --packages-select rl_utils
   source install/setup.bash
   ```

### Configuration Not Found

If the training script can't find the configuration file:

1. Check that `arena_bringup` is installed:
   ```bash
   ros2 pkg list | grep arena_bringup
   ```

2. Verify the configuration file exists:
   ```bash
   ros2 pkg prefix arena_bringup
   ls $(ros2 pkg prefix arena_bringup)/share/arena_bringup/configs/training/
   ```

3. Use an absolute path to the configuration file as a workaround.

## Development

### Adding New Features

When adding new training features:

1. Implement the feature in `rosnav_rl` or `rl_utils`
2. Update configuration schemas if needed
3. Test with the training script
4. Update documentation

### Testing

To test the training pipeline without full training:

```bash
# Use a test configuration with reduced training steps
ros2 run arena_training train_agent.py --config test_config.yaml
```

## Related Packages

- `arena_bringup` - Launch files and configurations
- `arena_simulation_setup` - Simulation environment setup
- `task_generator` - Dynamic task generation
- `arena_evaluation` - Model evaluation utilities
- `rl_utils` - RL utilities and wrappers

## License

MIT License - See LICENSE.md in the Arena repository root.

## Maintainer

voshch <dev@voshch.dev>

## References

- [Arena-Rosnav](https://github.com/Arena-Rosnav)
- [rosnav-rl](https://github.com/Arena-Rosnav/rosnav-rl)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
