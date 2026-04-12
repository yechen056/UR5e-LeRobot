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

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.bi_gello_leader import BiGelloLeader, BiGelloLeaderConfig
from lerobot.teleoperators.gello_leader import GelloLeader, GelloLeaderConfig


def test_gello_leader_config_is_registered():
    assert "gello_leader" in GelloLeaderConfig.get_known_choices()


def test_make_gello_leader_from_config():
    teleop = make_teleoperator_from_config(GelloLeaderConfig(port="/dev/ttyUSB0"))

    assert isinstance(teleop, GelloLeader)
    assert "shoulder_pan.pos" in teleop.action_features
    assert "gripper.pos" in teleop.action_features


def test_make_bimanual_gello_leader_from_config():
    teleop = make_teleoperator_from_config(
        BiGelloLeaderConfig(
            left_arm_config=GelloLeaderConfig(port="/dev/ttyUSB0"),
            right_arm_config=GelloLeaderConfig(port="/dev/ttyUSB1"),
        )
    )

    assert isinstance(teleop, BiGelloLeader)
    assert "left_shoulder_pan.pos" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features
