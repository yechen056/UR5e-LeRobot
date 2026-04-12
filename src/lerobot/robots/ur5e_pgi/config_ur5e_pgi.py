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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class UR5ePGIConfigBase:
    robot_ip: str
    action_mode: str = "joint"
    gripper_ip: str | None = None
    gripper_port: int = 8888
    gripper_unit_id: int = 1
    gripper_force: int = 30
    gripper_speed: int = 100
    disable_gripper: bool = False
    max_relative_target: float | dict[str, float] | None = None
    max_relative_translation: float | None = None
    max_relative_rotation: float | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "gripper",
    )
    servo_dt: float = 1.0 / 500.0
    lookahead_time: float = 0.2
    gain: int = 100
    velocity: float = 0.5
    acceleration: float = 0.5
    tcp_offset_pose: tuple[float, float, float, float, float, float] | None = None
    payload_mass: float | None = None
    payload_cog: tuple[float, float, float] | None = None

    @property
    def has_gripper(self) -> bool:
        return not self.disable_gripper and self.gripper_ip is not None

    def __post_init__(self):
        if self.action_mode not in ("joint", "eef"):
            raise ValueError("`action_mode` must be either 'joint' or 'eef'.")


@RobotConfig.register_subclass("ur5e_pgi")
@dataclass
class UR5ePGIConfig(RobotConfig, UR5ePGIConfigBase):
    pass
