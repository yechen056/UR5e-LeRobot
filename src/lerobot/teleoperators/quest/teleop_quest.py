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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from ..teleoperator import Teleoperator
from .config_quest import QuestTeleopConfig

logger = logging.getLogger(__name__)


@dataclass
class _QuestArmState:
    hand: str
    prefix: str
    robot_ip: str
    gripper_target: float
    rtde_control: Any = None
    rtde_receive: Any = None
    control_active: bool = False
    reference_quest_pose: np.ndarray | None = None
    reference_tcp_pos: np.ndarray | None = None
    reference_tcp_rot: np.ndarray | None = None
    last_target_tcp: np.ndarray | None = None
    last_joint_action: np.ndarray | None = None
    fallback_used: bool = field(default=False, init=False)

    @property
    def pose_key(self) -> str:
        return self.hand

    @property
    def trigger_key(self) -> str:
        return "leftTrig" if self.hand == "l" else "rightTrig"

    @property
    def gripper_open_key(self) -> str:
        return "Y" if self.hand == "l" else "B"

    @property
    def gripper_close_key(self) -> str:
        return "X" if self.hand == "l" else "A"


class QuestTeleop(Teleoperator):
    """Meta Quest teleoperator for UR robots using LeRobot action dictionaries."""

    config_class = QuestTeleopConfig
    name = "quest"
    _tcp_action_names = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")
    _quest2ur = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _ur2quest = np.linalg.inv(_quest2ur)

    def __init__(self, config: QuestTeleopConfig):
        super().__init__(config)
        self.config = config
        self._oculus_reader = None
        self._is_connected = False
        self._arm_joint_names = tuple(config.joint_names[:6])
        self._joint_names = tuple(
            config.joint_names[: 6 + int(config.use_gripper and len(config.joint_names) > 6)]
        )
        self._arms = self._make_arm_states()

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode == "eef":
            features = {}
            for arm in self._arms:
                features.update({f"{arm.prefix}{name}": float for name in self._tcp_action_names})
                if self.config.use_gripper and len(self._joint_names) == 7:
                    features[f"{arm.prefix}gripper.pos"] = float
            return features

        features = {}
        for arm in self._arms:
            features.update({f"{arm.prefix}{joint}.pos": float for joint in self._joint_names})
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate

        try:
            from oculus_reader.reader import OculusReader
        except ImportError as exc:
            raise ImportError("Quest teleoperation requires the `oculus_reader` Python package.") from exc

        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError(
                "Quest teleoperation for UR robots requires `ur-rtde` in the current environment."
            ) from exc

        self._oculus_reader = OculusReader()
        for arm in self._arms:
            arm.rtde_control = RTDEControlInterface(arm.robot_ip)
            arm.rtde_receive = RTDEReceiveInterface(arm.robot_ip)
        self._is_connected = True
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        assert self._oculus_reader is not None

        pose_data, button_data = self._oculus_reader.get_transformations_and_buttons()
        action: dict[str, float] = {}
        for arm in self._arms:
            arm_action = self._get_arm_action(arm, pose_data, button_data)
            action.update(arm_action)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        self._oculus_reader = None
        for arm in self._arms:
            arm.rtde_control = None
            arm.rtde_receive = None
            arm.control_active = False
            arm.reference_quest_pose = None
        self._is_connected = False

    def _make_arm_states(self) -> list[_QuestArmState]:
        initial_gripper = float(np.clip(self.config.gripper_initial_pos, 0.0, 1.0))
        if self.config.bimanual:
            assert self.config.left_robot_ip is not None
            assert self.config.right_robot_ip is not None
            return [
                _QuestArmState("l", "left_", self.config.left_robot_ip, initial_gripper),
                _QuestArmState("r", "right_", self.config.right_robot_ip, initial_gripper),
            ]

        assert self.config.robot_ip is not None
        return [_QuestArmState(self.config.single_hand, "", self.config.robot_ip, initial_gripper)]

    def _get_arm_action(
        self,
        arm: _QuestArmState,
        pose_data: dict[str, Any] | None,
        button_data: dict[str, Any] | None,
    ) -> dict[str, float]:
        assert arm.rtde_control is not None
        assert arm.rtde_receive is not None

        current_tcp = np.asarray(arm.rtde_receive.getActualTCPPose(), dtype=np.float64)
        current_joints = np.asarray(arm.rtde_receive.getActualQ(), dtype=np.float64)
        self._initialize_hold_targets(arm, current_tcp, current_joints)
        self._update_gripper_target(arm, button_data)

        hold_tcp = arm.last_target_tcp.copy()
        hold_joint = arm.last_joint_action.copy()
        if self.config.use_gripper and len(self._joint_names) == 7:
            hold_tcp[6] = arm.gripper_target
            hold_joint[6] = arm.gripper_target

        trigger_value = self._trigger_value(arm, button_data)
        if trigger_value <= 0.5 or not pose_data or arm.pose_key not in pose_data:
            arm.control_active = False
            arm.reference_quest_pose = None
            arm.last_target_tcp = hold_tcp
            arm.last_joint_action = hold_joint
            return self._format_action(arm, hold_tcp, hold_joint)

        current_quest_pose = np.asarray(pose_data[arm.pose_key], dtype=np.float64)
        if not arm.control_active:
            arm.control_active = True
            arm.reference_quest_pose = current_quest_pose.copy()
            arm.reference_tcp_pos = current_tcp[:3].copy()
            arm.reference_tcp_rot = Rotation.from_rotvec(current_tcp[3:6]).as_matrix()
            anchored_tcp = self._tcp_with_gripper(current_tcp, arm.gripper_target)
            anchored_joint = self._joint_with_gripper(current_joints, arm.gripper_target)
            arm.last_target_tcp = anchored_tcp
            arm.last_joint_action = anchored_joint
            return self._format_action(arm, anchored_tcp, anchored_joint)

        target_tcp = self._map_quest_pose_to_tcp(arm, current_quest_pose)
        if self.config.use_gripper and len(self._joint_names) == 7:
            target_tcp = np.append(target_tcp, arm.gripper_target)

        if self.config.action_mode == "eef":
            arm.last_target_tcp = target_tcp.astype(np.float64)
            return self._format_action(arm, arm.last_target_tcp, hold_joint)

        target_joint = self._inverse_kinematics(arm, target_tcp[:6], current_joints, hold_joint)
        if self.config.use_gripper and len(self._joint_names) == 7:
            target_joint = np.append(target_joint[:6], arm.gripper_target)

        arm.last_target_tcp = target_tcp.astype(np.float64)
        arm.last_joint_action = target_joint.astype(np.float64)
        return self._format_action(arm, arm.last_target_tcp, arm.last_joint_action)

    def _initialize_hold_targets(
        self,
        arm: _QuestArmState,
        current_tcp: np.ndarray,
        current_joints: np.ndarray,
    ) -> None:
        if arm.last_target_tcp is None:
            arm.last_target_tcp = self._tcp_with_gripper(current_tcp, arm.gripper_target)
        if arm.last_joint_action is None:
            arm.last_joint_action = self._joint_with_gripper(current_joints, arm.gripper_target)

    def _tcp_with_gripper(self, tcp_pose: np.ndarray, gripper_target: float) -> np.ndarray:
        if self.config.use_gripper and len(self._joint_names) == 7:
            return np.append(np.asarray(tcp_pose[:6], dtype=np.float64), float(gripper_target))
        return np.asarray(tcp_pose[:6], dtype=np.float64)

    def _joint_with_gripper(self, joints: np.ndarray, gripper_target: float) -> np.ndarray:
        if self.config.use_gripper and len(self._joint_names) == 7:
            return np.append(np.asarray(joints[:6], dtype=np.float64), float(gripper_target))
        return np.asarray(joints[:6], dtype=np.float64)

    def _map_quest_pose_to_tcp(self, arm: _QuestArmState, current_quest_pose: np.ndarray) -> np.ndarray:
        assert arm.reference_quest_pose is not None
        assert arm.reference_tcp_pos is not None
        assert arm.reference_tcp_rot is not None

        delta_rot = current_quest_pose[:3, :3] @ np.linalg.inv(arm.reference_quest_pose[:3, :3])
        delta_pos = current_quest_pose[:3, 3] - arm.reference_quest_pose[:3, 3]

        delta_pos_ur = self._quest2ur @ delta_pos * self.config.translation_scale
        delta_pos_ur[0] *= -1.0
        delta_pos_ur[1] *= -1.0

        delta_rot_ur = self._quest2ur @ delta_rot @ self._ur2quest
        delta_rotvec_ur = Rotation.from_matrix(delta_rot_ur).as_rotvec()
        delta_rotvec_ur *= -1.0
        delta_rot_ur = Rotation.from_rotvec(delta_rotvec_ur).as_matrix()

        next_tcp = np.zeros((6,), dtype=np.float64)
        next_tcp[:3] = arm.reference_tcp_pos + delta_pos_ur
        next_tcp[3:] = Rotation.from_matrix(delta_rot_ur @ arm.reference_tcp_rot).as_rotvec()
        return next_tcp

    def _inverse_kinematics(
        self,
        arm: _QuestArmState,
        target_tcp: np.ndarray,
        current_joints: np.ndarray,
        hold_joint: np.ndarray,
    ) -> np.ndarray:
        assert arm.rtde_control is not None
        arm.fallback_used = False
        try:
            ik_solution = arm.rtde_control.getInverseKinematics(
                target_tcp.tolist(),
                current_joints[:6].tolist(),
                self.config.max_ik_position_error,
                self.config.max_ik_orientation_error,
            )
        except TypeError:
            ik_solution = arm.rtde_control.getInverseKinematics(target_tcp.tolist())

        if len(ik_solution) == 0:
            arm.fallback_used = True
            logger.debug(
                "UR inverse kinematics failed for pose %s; keeping last Quest joint target.", target_tcp
            )
            return np.asarray(hold_joint[:6], dtype=np.float64)

        return np.asarray(ik_solution[:6], dtype=np.float64)

    def _update_gripper_target(self, arm: _QuestArmState, button_data: dict[str, Any] | None) -> None:
        if not self.config.use_gripper or not button_data:
            return
        if bool(button_data.get(arm.gripper_open_key, False)):
            arm.gripper_target = 1.0
        if bool(button_data.get(arm.gripper_close_key, False)):
            arm.gripper_target = 0.0
        arm.gripper_target = float(np.clip(arm.gripper_target, 0.0, 1.0))

    def _trigger_value(self, arm: _QuestArmState, button_data: dict[str, Any] | None) -> float:
        if not button_data:
            return 0.0
        state = button_data.get(arm.trigger_key, (0.0,))
        if isinstance(state, (list, tuple, np.ndarray)):
            return float(state[0]) if len(state) > 0 else 0.0
        return float(state)

    def _format_action(
        self,
        arm: _QuestArmState,
        target_tcp: np.ndarray,
        target_joint: np.ndarray,
    ) -> dict[str, float]:
        if self.config.action_mode == "eef":
            action = {
                f"{arm.prefix}{name}": float(value)
                for name, value in zip(self._tcp_action_names, target_tcp[:6], strict=True)
            }
            if self.config.use_gripper and len(self._joint_names) == 7:
                action[f"{arm.prefix}gripper.pos"] = float(target_tcp[6])
            return action

        action = {
            f"{arm.prefix}{joint_name}.pos": float(value)
            for joint_name, value in zip(self._arm_joint_names, target_joint[:6], strict=True)
        }
        if self.config.use_gripper and len(self._joint_names) == 7:
            action[f"{arm.prefix}gripper.pos"] = float(target_joint[6])
        return action
