import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import threading

from ..cfg import TrainingCfg


class SupervisorNode(Node):
    # This node is responsible for supervising the training process in a ROS2 environment.
    # stores training parameters and configurations.
    # subscribes to relevant topics and manages the training lifecycle.
    # communicates with other nodes to coordinate training tasks (task reset, curriculum management, etc.).

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.get_logger().info(f"{node_name} has been started.")
        self._shutdown_event = threading.Event()

        # SingleThreadedExecutor: no pool threads, no GIL competition from
        # background executor threads.  spin_once() releases the GIL during
        # rcl_wait so the worker's main thread can run Python code concurrently.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_loop)

        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])


    def start_spinning(self):
        """Starts the spin loop in a background thread."""
        if not self._spin_thread.is_alive():
            self._shutdown_event.clear()
            self._spin_thread.start()
            self.get_logger().info(
                "SupervisorNode spinning started with SingleThreadedExecutor."
            )

    def stop_spinning(self):
        """Stops the spin loop."""
        if self._spin_thread.is_alive():
            self._shutdown_event.set()
            self._spin_thread.join()
            self.get_logger().info("SupervisorNode spinning stopped.")

    def _spin_loop(self):
        """Continuously spins the ROS2 node in a background thread with SingleThreadedExecutor."""
        import traceback as _tb
        while not self._shutdown_event.is_set():
            try:
                # Use the executor to handle all callback groups
                self._executor.spin_once(timeout_sec=0.001)
            except Exception as e:
                # Never let the spin thread die silently: dying makes every
                # async service call wait forever on its completion event.
                try:
                    self.get_logger().error(
                        f"[SupervisorNode._spin_loop] exception: {e!r}\n{_tb.format_exc()}"
                    )
                except Exception:
                    import sys as _sys
                    print(
                        f"[SupervisorNode._spin_loop] exception: {e!r}\n{_tb.format_exc()}",
                        file=_sys.stderr, flush=True,
                    )
                # If the rclpy context died, exit cleanly instead of spinning hot.
                try:
                    import rclpy as _rclpy
                    if not _rclpy.ok():
                        break
                except Exception:
                    break

    def destroy_node(self):
        self.stop_spinning()
        self._executor.remove_node(self)
        self._executor.shutdown()
        super().destroy_node()


def main(args=None):
    from arena_rclpy_mixins.spin import spin_node

    rclpy.init(args=args)
    spin_node(SupervisorNode("basic_node"))


if __name__ == "__main__":
    main()
