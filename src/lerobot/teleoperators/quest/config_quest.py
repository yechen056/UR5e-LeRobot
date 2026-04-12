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


@TeleoperatorConfig.register_subclass("quest")
@dataclass
class QuestTeleopConfig(TeleoperatorConfig):
    robot_ip: str | None = None
    left_robot_ip: str | None = None
    right_robot_ip: str | None = None
    action_mode: str = "joint"
    bimanual: bool = False
    single_hand: str = "r"
    translation_scale: float = 3.0
    use_gripper: bool = True
    gripper_initial_pos: float = 1.0
    max_ik_position_error: float = 1e-3
    max_ik_orientation_error: float = 1e-2
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

    def __post_init__(self):
        if self.action_mode not in ("joint", "eef"):
            raise ValueError("`action_mode` must be either 'joint' or 'eef'.")
        if self.single_hand not in ("l", "r"):
            raise ValueError("`single_hand` must be either 'l' or 'r'.")
        if len(self.joint_names) not in (6, 7):
            raise ValueError("`joint_names` must contain 6 arm joints and an optional gripper joint.")
        if self.translation_scale <= 0.0:
            raise ValueError("`translation_scale` must be positive.")
        if self.bimanual:
            if self.left_robot_ip is None or self.right_robot_ip is None:
                raise ValueError("Bimanual Quest teleop requires `left_robot_ip` and `right_robot_ip`.")
        elif self.robot_ip is None:
            raise ValueError("Single-arm Quest teleop requires `robot_ip`.")
