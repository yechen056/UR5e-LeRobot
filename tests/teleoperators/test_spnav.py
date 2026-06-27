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

import numpy as np

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


class _FakeReceive:
    def __init__(self):
        self.tcp_reads = 0
        self.joint_reads = 0

    def getActualTCPPose(self):  # noqa: N802
        self.tcp_reads += 1
        return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

    def getActualQ(self):  # noqa: N802
        self.joint_reads += 1
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_spnav_episode_reference_uses_current_robot_state():
    teleop = SpnavTeleop(SpnavTeleopConfig(robot_ip="192.168.0.10"))
    receive = _FakeReceive()
    teleop._rtde_receive = receive
    teleop._is_connected = True
    teleop._motion_state[:] = 1.0
    teleop._button_state[0] = True

    teleop.reset_reference_from_robot()

    np.testing.assert_array_equal(teleop._motion_state, np.zeros(6))
    assert not teleop._button_state
    assert receive.tcp_reads == 1
    assert receive.joint_reads == 1
