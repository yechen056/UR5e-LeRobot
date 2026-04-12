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

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..gello_leader import GelloLeader, GelloLeaderConfig
from ..teleoperator import Teleoperator
from .config_bi_gello_leader import BiGelloLeaderConfig


class BiGelloLeader(Teleoperator):
    config_class = BiGelloLeaderConfig
    name = "bi_gello_leader"

    def __init__(self, config: BiGelloLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = GelloLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            motors=config.left_arm_config.motors,
            start_joints=config.left_arm_config.start_joints,
            joint_signs=config.left_arm_config.joint_signs,
            gripper_open_correction_deg=config.left_arm_config.gripper_open_correction_deg,
            gripper_close_delta_deg=config.left_arm_config.gripper_close_delta_deg,
            offset_search_range_pi=config.left_arm_config.offset_search_range_pi,
        )
        right_arm_config = GelloLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            motors=config.right_arm_config.motors,
            start_joints=config.right_arm_config.start_joints,
            joint_signs=config.right_arm_config.joint_signs,
            gripper_open_correction_deg=config.right_arm_config.gripper_open_correction_deg,
            gripper_close_delta_deg=config.right_arm_config.gripper_close_delta_deg,
            offset_search_range_pi=config.right_arm_config.offset_search_range_pi,
        )

        self.left_arm = GelloLeader(left_arm_config)
        self.right_arm = GelloLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        left_action = self.left_arm.get_action()
        right_action = self.right_arm.get_action()
        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        del feedback
        raise NotImplementedError("Feedback is not implemented for bimanual Gello leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
