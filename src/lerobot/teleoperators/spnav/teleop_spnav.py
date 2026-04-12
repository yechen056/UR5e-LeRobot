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
from collections import defaultdict
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from ..teleoperator import Teleoperator
from .configuration_spnav import SpnavTeleopConfig

logger = logging.getLogger(__name__)


class SpnavTeleop(Teleoperator):
    """SpaceMouse teleoperator that outputs UR joint targets through UR RTDE IK."""

    config_class = SpnavTeleopConfig
    name = "spnav"
    _tcp_action_names = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")

    def __init__(self, config: SpnavTeleopConfig):
        super().__init__(config)
        self.config = config
        self._rtde_control = None
        self._rtde_receive = None
        self._spnav = None
        self._is_connected = False

        self._motion_state = np.zeros(6, dtype=np.float64)
        self._button_state: dict[int, bool] = defaultdict(bool)
        self._gripper_target = float(np.clip(config.gripper_initial_pos, 0.0, 1.0))
        self._spnav_to_robot_axes = np.array(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        self._arm_joint_names = tuple(config.joint_names[:6])
        self._joint_names = tuple(config.joint_names[: 6 + int(self.config.use_gripper and len(config.joint_names) > 6)])

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode == "eef":
            features = {name: float for name in self._tcp_action_names}
            if self.config.use_gripper and len(self._joint_names) == 7:
                features["gripper.pos"] = float
            return features

        return {f"{joint}.pos": float for joint in self._joint_names}

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
            import spnav
        except ImportError as exc:
            raise ImportError("spnav teleoperation requires the `spnav` Python package.") from exc

        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError(
                "spnav teleoperation for UR robots requires `ur-rtde` in the current environment."
            ) from exc

        self._spnav = spnav
        self._rtde_control = RTDEControlInterface(self.config.robot_ip)
        self._rtde_receive = RTDEReceiveInterface(self.config.robot_ip)
        self._spnav.spnav_open()
        self._is_connected = True
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        assert self._spnav is not None
        assert self._rtde_control is not None
        assert self._rtde_receive is not None

        self._drain_events()
        close_pressed = self._button_state.get(self.config.close_gripper_button, False)
        open_pressed = self._button_state.get(self.config.open_gripper_button, False)

        current_pose = np.asarray(self._rtde_receive.getActualTCPPose(), dtype=np.float64)
        current_joints = np.asarray(self._rtde_receive.getActualQ(), dtype=np.float64)

        delta_translation = self._transform_motion(self._motion_state[:3]) * np.asarray(
            self.config.translation_step_m, dtype=np.float64
        )
        delta_rotation = self._transform_motion(self._motion_state[3:]) * np.asarray(
            self.config.rotation_step_rad, dtype=np.float64
        )

        # Keep roll / pitch available but more damped than translation and yaw.
        delta_translation *= 1.2
        delta_rotation[:2] = 0.0
        delta_rotation[1] *= -1.0
        delta_rotation[2] *= -1.25

        target_pose = current_pose.copy()
        target_pose[:3] = target_pose[:3] + delta_translation

        if not self.config.position_only:
            current_rot = Rotation.from_rotvec(current_pose[3:]).as_matrix()
            delta_rot = Rotation.from_rotvec(delta_rotation).as_matrix()
            target_pose[3:] = Rotation.from_matrix(current_rot @ delta_rot).as_rotvec()

        self._update_gripper_target()

        if self.config.action_mode == "eef":
            action = {name: float(value) for name, value in zip(self._tcp_action_names, target_pose, strict=True)}
            if self.config.use_gripper and len(self._joint_names) == 7:
                action["gripper.pos"] = self._gripper_target
            return action

        try:
            ik_solution = self._rtde_control.getInverseKinematics(
                target_pose.tolist(),
                current_joints.tolist(),
                self.config.max_ik_position_error,
                self.config.max_ik_orientation_error,
            )
        except TypeError:
            # Older ur-rtde builds only expose the pose-only overload.
            ik_solution = self._rtde_control.getInverseKinematics(target_pose.tolist())

        if len(ik_solution) == 0:
            logger.debug("UR inverse kinematics failed for pose %s; keeping current joints.", target_pose.tolist())
            ik_solution = current_joints.tolist()

        action = {
            f"{joint_name}.pos": float(value)
            for joint_name, value in zip(self._arm_joint_names, ik_solution[:6], strict=True)
        }
        if self.config.use_gripper and len(self._joint_names) == 7:
            action["gripper.pos"] = self._gripper_target

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self._spnav is not None:
                self._spnav.spnav_close()
        finally:
            self._rtde_control = None
            self._rtde_receive = None
            self._spnav = None
            self._is_connected = False

    def _drain_events(self) -> None:
        assert self._spnav is not None

        while True:
            event = self._spnav.spnav_poll_event()
            if event is None:
                break
            if isinstance(event, self._spnav.SpnavMotionEvent):
                motion = np.array(event.translation + event.rotation, dtype=np.float64)
                motion = motion / float(self.config.max_value)
                deadzone = np.asarray(self.config.deadzone, dtype=np.float64)
                motion[np.abs(motion) < deadzone] = 0.0
                self._motion_state = motion
            elif isinstance(event, self._spnav.SpnavButtonEvent):
                self._button_state[int(event.bnum)] = bool(event.press)

    def _transform_motion(self, motion: np.ndarray) -> np.ndarray:
        return self._spnav_to_robot_axes @ motion

    def _update_gripper_target(self) -> None:
        if not self.config.use_gripper:
            return

        close_pressed = self._button_state.get(self.config.close_gripper_button, False)
        open_pressed = self._button_state.get(self.config.open_gripper_button, False)

        if close_pressed and not open_pressed:
            self._gripper_target = 0.0
        elif open_pressed and not close_pressed:
            self._gripper_target = 1.0

        self._gripper_target = float(np.clip(self._gripper_target, 0.0, 1.0))
