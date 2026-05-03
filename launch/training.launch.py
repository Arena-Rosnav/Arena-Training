import launch
from arena_bringup.actions import IncludeLaunchDescriptionForward
from arena_bringup.substitutions import LaunchArgument
from launch.actions import ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    train_config = LaunchArgument(
        name='train_config',
        default_value='',
        description='Path to training config YAML.',
    )

    arena_launch = IncludeLaunchDescriptionForward(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arena_bringup'),
                'launch', 'arena.launch.py',
            ]),
        ),
        overrides={'env_n': '0', 'auto_reset': 'false'},
    )

    train_agent = ExecuteProcess(
        cmd=['ros2', 'run', 'arena_training', 'train_agent.py',
             '--config', train_config.substitution],
        output='screen',
        on_exit=launch.actions.Shutdown(reason='train_agent exited'),
    )

    return launch.LaunchDescription([
        *ld_items,
        arena_launch,
        train_agent,
    ])
