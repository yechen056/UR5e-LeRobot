#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_ur5e_pgi import BiUR5ePGI, BiUR5ePGIConfig
from lerobot.robots.ur5e_pgi import UR5ePGI, UR5ePGIConfig

TEST_CALIBRATION_DIR = Path("/tmp/lerobot-test-calibration/ur5e_pgi")


def test_ur5e_pgi_config_is_registered():
    assert "ur5e_pgi" in UR5ePGIConfig.get_known_choices()


def test_make_robot_from_config_without_gripper():
    config = UR5ePGIConfig(
        robot_ip="192.168.1.3",
        disable_gripper=True,
        calibration_dir=TEST_CALIBRATION_DIR,
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, UR5ePGI)
    assert robot.name == "ur5e_pgi"
    assert list(robot.action_features) == [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow.pos",
        "wrist_1.pos",
        "wrist_2.pos",
        "wrist_3.pos",
    ]


def test_make_robot_from_config_with_gripper():
    config = UR5ePGIConfig(
        robot_ip="192.168.1.3",
        gripper_ip="192.168.1.8",
        calibration_dir=TEST_CALIBRATION_DIR,
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, UR5ePGI)
    assert "gripper.pos" in robot.action_features
    assert "gripper.pos" in robot.observation_features


def test_make_robot_from_config_eef_action_mode():
    config = UR5ePGIConfig(
        robot_ip="192.168.1.3",
        gripper_ip="192.168.1.8",
        action_mode="eef",
        calibration_dir=TEST_CALIBRATION_DIR,
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, UR5ePGI)
    assert list(robot.action_features) == [
        "tcp.x",
        "tcp.y",
        "tcp.z",
        "tcp.rx",
        "tcp.ry",
        "tcp.rz",
        "gripper.pos",
    ]


def test_make_bimanual_robot_from_config():
    config = BiUR5ePGIConfig(
        left_arm_config=UR5ePGIConfig(robot_ip="192.168.1.3", gripper_ip="192.168.1.8"),
        right_arm_config=UR5ePGIConfig(robot_ip="192.168.1.5", gripper_ip="192.168.1.7"),
        calibration_dir=TEST_CALIBRATION_DIR,
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, BiUR5ePGI)
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.observation_features


def test_make_bimanual_robot_from_config_eef_action_mode():
    config = BiUR5ePGIConfig(
        left_arm_config=UR5ePGIConfig(
            robot_ip="192.168.1.3",
            gripper_ip="192.168.1.8",
            action_mode="eef",
        ),
        right_arm_config=UR5ePGIConfig(
            robot_ip="192.168.1.5",
            gripper_ip="192.168.1.7",
            action_mode="eef",
        ),
        calibration_dir=TEST_CALIBRATION_DIR,
    )

    robot = make_robot_from_config(config)

    assert isinstance(robot, BiUR5ePGI)
    assert "left_tcp.x" in robot.action_features
    assert "left_gripper.pos" in robot.action_features
    assert "right_tcp.rz" in robot.action_features
    assert "right_gripper.pos" in robot.action_features
