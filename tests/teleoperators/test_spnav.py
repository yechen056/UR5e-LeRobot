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
from lerobot.teleoperators.spnav import SpnavTeleop, SpnavTeleopConfig


def test_spnav_config_is_registered():
    assert "spnav" in SpnavTeleopConfig.get_known_choices()


def test_make_spnav_from_config():
    teleop = make_teleoperator_from_config(SpnavTeleopConfig(robot_ip="192.168.0.10"))

    assert isinstance(teleop, SpnavTeleop)
    assert "shoulder_pan.pos" in teleop.action_features
    assert "wrist_3.pos" in teleop.action_features
    assert "gripper.pos" in teleop.action_features
