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


@dataclass
class GelloLeaderConfigBase:
    port: str
    motors: dict[str, tuple[int, str]] = field(
        default_factory=lambda: {
            "shoulder_pan": (1, "xl330-m288"),
            "shoulder_lift": (2, "xl330-m288"),
            "elbow": (3, "xl330-m288"),
            "wrist_1": (4, "xl330-m288"),
            "wrist_2": (5, "xl330-m288"),
            "wrist_3": (6, "xl330-m288"),
            "gripper": (7, "xl330-m077"),
        }
    )
    start_joints: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_signs: tuple[float, ...] = (1.0, 1.0, -1.0, 1.0, 1.0, 1.0)
    gripper_open_correction_deg: float = 0.2
    gripper_close_delta_deg: float = 42.0
    offset_search_range_pi: int = 8

    def __post_init__(self):
        if len(self.start_joints) != 6:
            raise ValueError(f"Expected 6 start_joints entries, got {len(self.start_joints)}")
        if len(self.joint_signs) != 6:
            raise ValueError(f"Expected 6 joint_signs entries, got {len(self.joint_signs)}")
        for idx, sign in enumerate(self.joint_signs):
            if sign not in (-1, 1, -1.0, 1.0):
                raise ValueError(f"joint_signs[{idx}] should be -1 or 1, got {sign}")


@TeleoperatorConfig.register_subclass("gello_leader")
@dataclass
class GelloLeaderConfig(TeleoperatorConfig, GelloLeaderConfigBase):
    pass
