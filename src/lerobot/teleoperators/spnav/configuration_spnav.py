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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spnav")
@dataclass
class SpnavTeleopConfig(TeleoperatorConfig):
    robot_ip: str
    action_mode: str = "joint"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
            "gripper",
        ]
    )
    max_value: int = 350
    deadzone: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.08, 0.08, 0.12])
    translation_step_m: list[float] = field(default_factory=lambda: [0.008, 0.008, 0.008])
    rotation_step_rad: list[float] = field(default_factory=lambda: [0.12, 0.12, 0.12])
    gripper_step: float = 0.6
    gripper_initial_pos: float = 0.0
    use_gripper: bool = True
    open_gripper_button: int = 1
    close_gripper_button: int = 0
    position_only: bool = False
    max_ik_position_error: float = 1e-3
    max_ik_orientation_error: float = 1e-2

    def __post_init__(self):
        if self.action_mode not in ("joint", "eef"):
            raise ValueError("`action_mode` must be either 'joint' or 'eef'.")
        if len(self.deadzone) != 6:
            raise ValueError("`deadzone` must contain 6 values.")
        if len(self.translation_step_m) != 3:
            raise ValueError("`translation_step_m` must contain 3 values.")
        if len(self.rotation_step_rad) != 3:
            raise ValueError("`rotation_step_rad` must contain 3 values.")
        if len(self.joint_names) not in (6, 7):
            raise ValueError("`joint_names` must contain 6 arm joints and an optional gripper joint.")
