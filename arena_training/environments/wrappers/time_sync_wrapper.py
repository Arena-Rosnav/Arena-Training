import gymnasium as gym
import time
from rclpy.node import Node
from rclpy.time import Time  # Import Time for type hinting

from rosnav_rl.model.dreamerv3.envs.wrappers import _GymDelegatingWrapper


class TimeSyncWrapper(_GymDelegatingWrapper):
    def __init__(self, env, control_hz: float = 10.0, warning_slop: float = 0.1):
        """
        A Gym Wrapper to synchronize step calls to a specific control frequency using ROS 2 time.

        Args:
            env: The Gym environment to wrap.
            control_hz: The desired control frequency in Hz.
            warning_slop: A factor to allow for a small deviation in control frequency before issuing a warning.
                          E.g., a value of 0.1 means that a warning will be issued if the actual frequency
                          is off by more than 10% of the desired interval.
        """
        super().__init__(env)
        if not isinstance(env.node, Node):
            raise ValueError("A valid rclpy.node.Node must be provided.")
        self.node = env.node
        self.clock = self.node.get_clock()

        if control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        # Store interval in nanoseconds for precise comparison with rclpy.time.Time objects
        self.control_interval_nanosec = int(1e9 / control_hz)
        self.warning_slop_nanosec = int(self.control_interval_nanosec * warning_slop)

        # Time when the last env.step() was allowed to initiate.
        # Initialized to current time, so the first step call will also adhere to the interval logic.
        self.last_step_initiation_time: Time = self.clock.now()
        self._skip_frequency_check = True  # Don't warn on the very first step
        # Rate object for efficient sim-clock-aware sleeping (signals via threading.Event
        # set directly by the /clock callback, bypassing spin_once queue starvation).
        self._rate = env.node.create_rate(control_hz, clock=self.clock)
        self._last_step_exit_wall = time.monotonic()

    def _initialize_environment(self):
        """Delegates to the wrapped env's _initialize_environment."""
        return self.env._initialize_environment()

    def _now(self) -> Time:
        """Returns the current ROS time as an rclpy.time.Time object."""
        return self.clock.now()

    def step(self, action):
        """
        Executes a step in the environment, ensuring the control frequency is respected.
        If called too frequently, this method will block (while spinning the node)
        until the control interval has passed since the last step initiation.
        """
        _t_entry_wall = time.monotonic()
        current_time = self._now()
        elapsed_nanosec = (current_time - self.last_step_initiation_time).nanoseconds

        # Warn if the actual interval is longer than the desired one, indicating a missed control frequency.
        # Skip the check right after a reset — the gap is caused by the reset itself
        # (or SubprocVecEnv waiting on sibling envs), not a real control loop miss.
        if self._skip_frequency_check:
            self._skip_frequency_check = False
        elif elapsed_nanosec > self.control_interval_nanosec + self.warning_slop_nanosec:
            desired_hz = 1e9 / self.control_interval_nanosec
            actual_hz = 1e9 / elapsed_nanosec
            wall_gap_ms = (_t_entry_wall - self._last_step_exit_wall) * 1000
            # throttle_duration_sec=5.0: rcutils suppresses DDS publish within window
            # (check happens before the rosout publish, so suppressed calls have zero DDS overhead)
            self.node.get_logger().warn(
                f"Control frequency missed! "
                f"Desired: {desired_hz:.2f}Hz, Actual: {actual_hz:.2f}Hz, wall_gap={wall_gap_ms:.0f}ms",
                throttle_duration_sec=5.0,
            )

        # Wait if the time elapsed since the last step initiation is less than the control interval.
        # Use Rate.sleep() which signals via a threading.Event set directly by the /clock callback,
        # bypassing spin_once queue starvation entirely.  This is the canonical ROS2 approach.
        if elapsed_nanosec < self.control_interval_nanosec:
            self._rate.sleep()  # blocks until next period boundary per sim clock

        # Update the initiation time for the current step (which is now allowed to proceed)
        # It's important to take a fresh timestamp here, after the wait loop.
        self.last_step_initiation_time = self._now()

        _result = self.env.step(action)
        self._last_step_exit_wall = time.monotonic()
        return _result

    def reset(self, **kwargs):
        """
        Resets the environment. Also resets the step timing mechanism for the first step
        after reset, similar to __init__.
        """
        # Reset last_step_initiation_time so the first step after reset
        # doesn't try to align with the time before the reset call.
        # It will enforce its interval relative to the actual time of reset completion.
        reset_return_value = self.env.reset(**kwargs)
        self.last_step_initiation_time = self._now()
        self._skip_frequency_check = True  # First step after reset: don't warn
        return reset_return_value
