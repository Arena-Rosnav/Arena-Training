import pytest

from arena_training.arena_rosnav_rl.cfg.arena_cfg.robot import RobotCfg
from arena_training.arena_rosnav_rl.utils.envs import set_laser_topic


@pytest.mark.parametrize(
    ("model", "topic"),
    [("jackal", "lidar"), ("boxer", "front_laser"), ("rbtheron", "scan"), ("mpo700", "scan"), ("rbvogui", "scan")],
)
def test_robot_laser_topic_is_namespace_relative(model: str, topic: str):
    assert RobotCfg(robot_model=model).laser_topic == topic


def test_collector_without_topic_takes_the_robot_laser_topic():
    config = {"datasources": {"front_laser": {"type": "sensor_msgs/LaserScan", "params": {"up_to_date_required": True}}}}
    set_laser_topic(config, "front_laser")
    assert config["datasources"]["front_laser"]["params"] == {"up_to_date_required": True, "topic": "front_laser"}


def test_collector_with_topic_keeps_it():
    config = {"datasources": {"scan_rear": {"type": "sensor_msgs/LaserScan", "params": {"topic": "scan_rear"}}}}
    set_laser_topic(config, "front_laser")
    assert config["datasources"]["scan_rear"]["params"]["topic"] == "scan_rear"


def test_other_collectors_are_untouched():
    config = {"datasources": {"odom": {"type": "nav_msgs/Odometry", "params": {"topic": "odom"}}}}
    set_laser_topic(config, "front_laser")
    assert config["datasources"]["odom"]["params"] == {"topic": "odom"}


def test_robot_without_laser_cannot_fill_the_topic():
    config = {"datasources": {"front_laser": {"type": "sensor_msgs/LaserScan", "params": {}}}}
    with pytest.raises(ValueError, match="laserscan"):
        set_laser_topic(config, None)
