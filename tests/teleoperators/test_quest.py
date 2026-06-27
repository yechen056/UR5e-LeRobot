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

import numpy as np

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.quest import QuestTeleop, QuestTeleopConfig
from lerobot.utils.rotation import Rotation

TEST_CALIBRATION_DIR = Path("/tmp/lerobot-test-calibration/quest")


class _FakeReader:
    def __init__(self, packets):
        self.packets = list(packets)
        self.stopped = False

    def get_transformations_and_buttons(self):
        if len(self.packets) == 1:
            return self.packets[0]
        return self.packets.pop(0)

    def stop(self):
        self.stopped = True


class _FakeReceive:
    def __init__(self, tcp=None, joints=None):
        self.tcp = np.zeros(6, dtype=np.float64) if tcp is None else np.asarray(tcp, dtype=np.float64)
        self.joints = (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
            if joints is None
            else np.asarray(joints, dtype=np.float64)
        )

    def getActualTCPPose(self):  # noqa: N802
        return self.tcp.tolist()

    def getActualQ(self):  # noqa: N802
        return self.joints.tolist()


class _FakeControl:
    def __init__(self, solutions):
        self.solutions = list(solutions)
        self.requests = []
        self.servo_stopped = False
        self.script_stopped = False

    def getInverseKinematics(self, target_tcp, *args):  # noqa: N802
        self.requests.append((target_tcp, args))
        if len(self.solutions) == 1:
            return self.solutions[0]
        return self.solutions.pop(0)

    def servoStop(self):  # noqa: N802
        self.servo_stopped = True

    def stopScript(self):  # noqa: N802
        self.script_stopped = True


def _quest_pose(position, rotation=None):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    pose[:3, :3] = np.eye(3) if rotation is None else rotation
    return pose


def _connect_fake(teleop, packets, ik_solutions=None):
    teleop._oculus_reader = _FakeReader(packets)
    teleop._is_connected = True
    for arm in teleop._arms:
        arm.rtde_receive = _FakeReceive()
        arm.rtde_control = _FakeControl([np.arange(10.0, 16.0)] if ik_solutions is None else ik_solutions)
    return teleop


def test_quest_config_is_registered():
    assert "quest" in QuestTeleopConfig.get_known_choices()


def test_make_quest_from_config_defaults_to_joint_actions():
    teleop = make_teleoperator_from_config(
        QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)
    )

    assert isinstance(teleop, QuestTeleop)
    assert teleop.config.action_mode == "joint"
    assert "shoulder_pan.pos" in teleop.action_features
    assert "wrist_3.pos" in teleop.action_features
    assert "gripper.pos" in teleop.action_features


def test_quest_eef_action_features():
    teleop = QuestTeleop(
        QuestTeleopConfig(robot_ip="192.168.0.10", action_mode="eef", calibration_dir=TEST_CALIBRATION_DIR)
    )

    assert list(teleop.action_features) == [
        "tcp.x",
        "tcp.y",
        "tcp.z",
        "tcp.rx",
        "tcp.ry",
        "tcp.rz",
        "gripper.pos",
    ]


def test_bimanual_quest_action_features():
    teleop = QuestTeleop(
        QuestTeleopConfig(
            bimanual=True,
            left_robot_ip="192.168.0.10",
            right_robot_ip="192.168.0.11",
            calibration_dir=TEST_CALIBRATION_DIR,
        )
    )

    assert "left_shoulder_pan.pos" in teleop.action_features
    assert "left_gripper.pos" in teleop.action_features
    assert "right_shoulder_pan.pos" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features


def test_bimanual_quest_eef_action_features():
    teleop = QuestTeleop(
        QuestTeleopConfig(
            bimanual=True,
            left_robot_ip="192.168.0.10",
            right_robot_ip="192.168.0.11",
            action_mode="eef",
            calibration_dir=TEST_CALIBRATION_DIR,
        )
    )

    assert "left_tcp.x" in teleop.action_features
    assert "left_gripper.pos" in teleop.action_features
    assert "right_tcp.rz" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features


def test_quest_pose_mapping_matches_policyconsensus_formula():
    teleop = QuestTeleop(
        QuestTeleopConfig(
            robot_ip="192.168.0.10",
            action_mode="eef",
            translation_scale=3.0,
            calibration_dir=TEST_CALIBRATION_DIR,
        )
    )
    arm = teleop._arms[0]
    arm.reference_quest_pose = _quest_pose([0.0, 0.0, 0.0])
    arm.reference_tcp_pos = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    arm.reference_tcp_rot = Rotation.from_rotvec(np.array([0.1, 0.2, -0.1])).as_matrix()

    current_quest_pose = _quest_pose(
        [0.01, -0.02, 0.03],
        Rotation.from_rotvec(np.array([0.2, -0.1, 0.05])).as_matrix(),
    )

    target_tcp = teleop._map_quest_pose_to_tcp(arm, current_quest_pose)

    delta_pos = current_quest_pose[:3, 3] - arm.reference_quest_pose[:3, 3]
    expected_delta_pos = teleop._quest2ur @ delta_pos * teleop.config.translation_scale
    expected_delta_pos[0] *= -1.0
    expected_delta_pos[1] *= -1.0
    delta_rot = current_quest_pose[:3, :3] @ np.linalg.inv(arm.reference_quest_pose[:3, :3])
    expected_delta_rot = teleop._quest2ur @ delta_rot @ teleop._ur2quest
    expected_delta_rotvec = Rotation.from_matrix(expected_delta_rot).as_rotvec()
    expected_delta_rotvec *= -1.0
    expected_delta_rot = Rotation.from_rotvec(expected_delta_rotvec).as_matrix()
    expected_tcp = np.zeros(6, dtype=np.float64)
    expected_tcp[:3] = arm.reference_tcp_pos + expected_delta_pos
    expected_tcp[3:] = Rotation.from_matrix(expected_delta_rot @ arm.reference_tcp_rot).as_rotvec()

    np.testing.assert_allclose(target_tcp, expected_tcp)


def test_quest_trigger_gates_motion_and_empty_ik_keeps_last_joint_target():
    packets = [
        ({"r": _quest_pose([0.0, 0.0, 0.0])}, {"rightGrip": (1.0,), "rightTrig": (0.0,)}),
        ({"r": _quest_pose([0.1, 0.0, 0.0])}, {"rightGrip": (1.0,), "rightTrig": (0.0,)}),
    ]
    teleop = _connect_fake(
        QuestTeleop(QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)),
        packets,
        ik_solutions=[[]],
    )

    anchored_action = teleop.get_action()
    moved_action = teleop.get_action()

    assert anchored_action["shoulder_pan.pos"] == 1.0
    assert moved_action["shoulder_pan.pos"] == 1.0
    assert moved_action["wrist_3.pos"] == 6.0
    assert teleop._arms[0].fallback_used


def test_quest_front_trigger_controls_gripper_proportionally():
    teleop = _connect_fake(
        QuestTeleop(QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)),
        [
            ({}, {"rightGrip": (0.0,), "rightTrig": (0.0,)}),
            ({}, {"rightGrip": (0.0,), "rightTrig": (0.35,)}),
            ({}, {"rightGrip": (0.0,), "rightTrig": (1.0,)}),
        ],
    )

    opened_action = teleop.get_action()
    partially_closed_action = teleop.get_action()
    closed_action = teleop.get_action()

    assert opened_action["gripper.pos"] == 1.0
    assert np.isclose(partially_closed_action["gripper.pos"], 0.65)
    assert closed_action["gripper.pos"] == 0.0


def test_quest_front_trigger_does_not_enable_arm_motion():
    teleop = _connect_fake(
        QuestTeleop(QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)),
        [({"r": _quest_pose([0.2, 0.0, 0.0])}, {"rightGrip": (0.0,), "rightTrig": (1.0,)})],
    )

    action = teleop.get_action()

    assert action["shoulder_pan.pos"] == 1.0
    assert action["gripper.pos"] == 0.0
    assert not teleop._arms[0].control_active


def test_bimanual_quest_front_triggers_control_their_own_grippers():
    teleop = _connect_fake(
        QuestTeleop(
            QuestTeleopConfig(
                bimanual=True,
                left_robot_ip="192.168.0.10",
                right_robot_ip="192.168.0.11",
                calibration_dir=TEST_CALIBRATION_DIR,
            )
        ),
        [
            (
                {},
                {
                    "leftGrip": (0.0,),
                    "rightGrip": (0.0,),
                    "leftTrig": (0.25,),
                    "rightTrig": (0.8,),
                },
            )
        ],
    )

    action = teleop.get_action()

    assert np.isclose(action["left_gripper.pos"], 0.75)
    assert np.isclose(action["right_gripper.pos"], 0.2)


def test_quest_episode_reference_uses_current_robot_state():
    teleop = _connect_fake(
        QuestTeleop(QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)),
        [({}, {})],
    )
    arm = teleop._arms[0]
    arm.control_active = True
    arm.reference_quest_pose = _quest_pose([1.0, 2.0, 3.0])
    arm.last_target_tcp = np.full(7, 99.0)
    arm.last_joint_action = np.full(7, 99.0)
    arm.rtde_receive = _FakeReceive(
        tcp=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        joints=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    teleop.reset_reference_from_robot()

    assert not arm.control_active
    assert arm.reference_quest_pose is None
    np.testing.assert_allclose(arm.last_target_tcp[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(arm.last_joint_action[:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_quest_disconnect_stops_reader_and_rtde_session():
    teleop = _connect_fake(
        QuestTeleop(QuestTeleopConfig(robot_ip="192.168.0.10", calibration_dir=TEST_CALIBRATION_DIR)),
        [({}, {})],
    )
    reader = teleop._oculus_reader
    control = teleop._arms[0].rtde_control

    teleop.disconnect()

    assert reader.stopped
    assert control.servo_stopped
    assert control.script_stopped
    assert not teleop.is_connected
    assert teleop._oculus_reader is None
