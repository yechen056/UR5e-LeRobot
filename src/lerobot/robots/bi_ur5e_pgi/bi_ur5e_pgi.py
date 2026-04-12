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

from functools import cached_property

from lerobot.robots.ur5e_pgi import UR5ePGI, UR5ePGIConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_ur5e_pgi import BiUR5ePGIConfig


class BiUR5ePGI(Robot):
    config_class = BiUR5ePGIConfig
    name = "bi_ur5e_pgi"

    def __init__(self, config: BiUR5ePGIConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = UR5ePGIConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            robot_ip=config.left_arm_config.robot_ip,
            action_mode=config.left_arm_config.action_mode,
            gripper_ip=config.left_arm_config.gripper_ip,
            gripper_port=config.left_arm_config.gripper_port,
            gripper_unit_id=config.left_arm_config.gripper_unit_id,
            gripper_force=config.left_arm_config.gripper_force,
            gripper_speed=config.left_arm_config.gripper_speed,
            disable_gripper=config.left_arm_config.disable_gripper,
            max_relative_target=config.left_arm_config.max_relative_target,
            max_relative_translation=config.left_arm_config.max_relative_translation,
            max_relative_rotation=config.left_arm_config.max_relative_rotation,
            cameras=config.left_arm_config.cameras,
            joint_names=config.left_arm_config.joint_names,
            servo_dt=config.left_arm_config.servo_dt,
            lookahead_time=config.left_arm_config.lookahead_time,
            gain=config.left_arm_config.gain,
            velocity=config.left_arm_config.velocity,
            acceleration=config.left_arm_config.acceleration,
            tcp_offset_pose=config.left_arm_config.tcp_offset_pose,
            payload_mass=config.left_arm_config.payload_mass,
            payload_cog=config.left_arm_config.payload_cog,
        )
        right_arm_config = UR5ePGIConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            robot_ip=config.right_arm_config.robot_ip,
            action_mode=config.right_arm_config.action_mode,
            gripper_ip=config.right_arm_config.gripper_ip,
            gripper_port=config.right_arm_config.gripper_port,
            gripper_unit_id=config.right_arm_config.gripper_unit_id,
            gripper_force=config.right_arm_config.gripper_force,
            gripper_speed=config.right_arm_config.gripper_speed,
            disable_gripper=config.right_arm_config.disable_gripper,
            max_relative_target=config.right_arm_config.max_relative_target,
            max_relative_translation=config.right_arm_config.max_relative_translation,
            max_relative_rotation=config.right_arm_config.max_relative_rotation,
            cameras=config.right_arm_config.cameras,
            joint_names=config.right_arm_config.joint_names,
            servo_dt=config.right_arm_config.servo_dt,
            lookahead_time=config.right_arm_config.lookahead_time,
            gain=config.right_arm_config.gain,
            velocity=config.right_arm_config.velocity,
            acceleration=config.right_arm_config.acceleration,
            tcp_offset_pose=config.right_arm_config.tcp_offset_pose,
            payload_mass=config.right_arm_config.payload_mass,
            payload_cog=config.right_arm_config.payload_cog,
        )

        self.left_arm = UR5ePGI(left_arm_config)
        self.right_arm = UR5ePGI(right_arm_config)
        self.cameras = {
            **{f"left_{k}": v for k, v in self.left_arm.cameras.items()},
            **{f"right_{k}": v for k, v in self.right_arm.cameras.items()},
        }

    @property
    def _motor_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @property
    def _observation_motor_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motor_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motor_features.items()},
        }

    @property
    def _camera_features(self) -> dict[str, tuple]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._camera_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm._camera_features.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._observation_motor_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motor_features

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate=calibrate)
        self.right_arm.connect(calibrate=calibrate)

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    @check_if_not_connected
    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()
        return {
            **{f"left_{key}": value for key, value in left_obs.items()},
            **{f"right_{key}": value for key, value in right_obs.items()},
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
