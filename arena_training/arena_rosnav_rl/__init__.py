# Arena ROS Navigation RL Package
# Training utilities and components for Arena-Rosnav

import sys
import types

# Keep tensorboard on its TF stub: a real tensorflow import exports llvm::*
# symbols that crash triton's dlopen inside torch._dynamo.
sys.modules["tensorboard.compat.notf"] = types.ModuleType("tensorboard.compat.notf")
