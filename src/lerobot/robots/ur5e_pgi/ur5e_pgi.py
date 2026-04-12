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
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_ur5e_pgi import UR5ePGIConfig

logger = logging.getLogger(__name__)


class _PgiTcpGripper:
    def __init__(self):
        self.client = None
        self.unit = 1

    def connect(self, ip: str, port: int, unit: int = 1, timeout: float = 1.0) -> bool:
        from pymodbus.client.sync import ModbusTcpClient

        self.unit = unit
        self.client = ModbusTcpClient(host=ip, port=port, timeout=timeout)
        return bool(self.client.connect())

    def disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def init_gripper(self) -> None:
        assert self.client is not None
        self.client.write_register(0x0100, 1, unit=self.unit)
        for _ in range(200):
            status = self.client.read_holding_registers(0x0200, 1, unit=self.unit)
            if hasattr(status, "registers") and status.registers[0] == 1:
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out while waiting for PGI gripper initialization.")

    def set_force(self, force: int) -> None:
        assert self.client is not None
        self.client.write_register(0x0101, int(force), unit=self.unit)

    def set_position(self, position: int) -> None:
        assert self.client is not None
        self.client.write_register(0x0103, int(position), unit=self.unit)

    def set_speed(self, speed: int) -> None:
        assert self.client is not None
        self.client.write_register(0x0104, int(speed), unit=self.unit)

    def read_current_position(self) -> int:
        assert self.client is not None
        result = self.client.read_holding_registers(0x0202, 1, unit=self.unit)
        if not hasattr(result, "registers"):
            raise RuntimeError("Failed to read current PGI gripper position.")
        return int(result.registers[0])


class UR5ePGI(Robot):
    """LeRobot wrapper for a UR5e arm with an optional PGI TCP gripper."""

    config_class = UR5ePGIConfig
    name = "ur5e_pgi"
    _tcp_action_names = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")

    def __init__(self, config: UR5ePGIConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._rtde_control = None
        self._rtde_receive = None
        self._gripper = None
        self._is_connected = False
        self._owns_rtde_session = True

        self._arm_joint_names = tuple(config.joint_names[:6])
        self._joint_names = tuple(config.joint_names[: 6 + int(config.has_gripper)])

    @property
    def _motor_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self._joint_names}

    @property
    def _tcp_features(self) -> dict[str, type]:
        features = {name: float for name in self._tcp_action_names}
        if self.config.has_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def _camera_features(self) -> dict[str, tuple]:
        return {
            camera_name: (camera_config.height, camera_config.width, 3)
            for camera_name, camera_config in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motor_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode == "eef":
            return self._tcp_features
        return self._motor_features

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(camera.is_connected for camera in self.cameras.values())
        return self._is_connected and cameras_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate

        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError(
                "UR5e support requires `ur-rtde`. Install it in the current environment first."
            ) from exc

        logger.info("Connecting UR5e arm at %s", self.config.robot_ip)
        reused_rtde_session = self._rtde_control is not None and self._rtde_receive is not None
        if reused_rtde_session:
            logger.info("Reusing existing UR RTDE session for %s", self.config.robot_ip)
            self._owns_rtde_session = False
        else:
            self._rtde_control = RTDEControlInterface(self.config.robot_ip)
            self._rtde_receive = RTDEReceiveInterface(self.config.robot_ip)
            self._owns_rtde_session = True

        if self.config.has_gripper:
            self._gripper = self._make_gripper()
            connected = self._gripper.connect(
                ip=self.config.gripper_ip,
                port=self.config.gripper_port,
                unit=self.config.gripper_unit_id,
            )
            if not connected:
                raise ConnectionError(
                    f"Failed to connect PGI gripper at {self.config.gripper_ip}:{self.config.gripper_port}"
                )
            self._gripper.init_gripper()
            self._gripper.set_speed(self.config.gripper_speed)
            self._gripper.set_force(self.config.gripper_force)

        for camera in self.cameras.values():
            camera.connect()

        self._is_connected = True
        self.configure()
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        logger.info("UR5e PGI robot does not require LeRobot-side calibration.")

    @check_if_not_connected
    def configure(self) -> None:
        assert self._rtde_control is not None

        self._rtde_control.endFreedriveMode()

        if self.config.tcp_offset_pose is not None:
            self._rtde_control.setTcp(list(self.config.tcp_offset_pose))

        if self.config.payload_mass is not None:
            if self.config.payload_cog is not None:
                self._rtde_control.setPayload(self.config.payload_mass, list(self.config.payload_cog))
            else:
                self._rtde_control.setPayload(self.config.payload_mass)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self._rtde_receive is not None

        obs: dict[str, Any] = {}
        joints = np.asarray(self._rtde_receive.getActualQ(), dtype=np.float32)
        if self.config.has_gripper:
            joints = np.append(joints, np.float32(self._read_gripper_position()))

        for joint_name, value in zip(self._joint_names, joints, strict=True):
            obs[f"{joint_name}.pos"] = float(value)

        for camera_name, camera in self.cameras.items():
            obs[camera_name] = camera.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self._rtde_control is not None

        expected_keys = list(self.action_features)
        missing_keys = [key for key in expected_keys if key not in action]
        if missing_keys:
            raise KeyError(f"Missing action keys for UR5e PGI: {missing_keys}")

        if self.config.action_mode == "eef":
            safe_action = self._apply_eef_safety_limits(action)
            tcp_target = np.asarray([safe_action[name] for name in self._tcp_action_names], dtype=np.float64)
            self._rtde_control.servoL(
                tcp_target.tolist(),
                self.config.velocity,
                self.config.acceleration,
                self.config.servo_dt,
                self.config.lookahead_time,
                self.config.gain,
            )
        else:
            current_obs = self.get_observation()
            safe_action = self._apply_safety_limits(action, current_obs)
            joint_targets = np.asarray(
                [safe_action[f"{name}.pos"] for name in self._arm_joint_names], dtype=np.float64
            )

            self._rtde_control.servoJ(
                joint_targets.tolist(),
                self.config.velocity,
                self.config.acceleration,
                self.config.servo_dt,
                self.config.lookahead_time,
                self.config.gain,
            )

        if self.config.has_gripper:
            assert self._gripper is not None
            gripper_target = float(safe_action["gripper.pos"])
            gripper_target = min(max(gripper_target, 0.0), 1.0)
            self._gripper.set_position(int(round(gripper_target * 1000.0)))

        return {key: float(safe_action[key]) for key in expected_keys}

    @check_if_not_connected
    def disconnect(self) -> None:
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Failed to disconnect camera cleanly.")

        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                logger.exception("Failed to disconnect PGI gripper cleanly.")
            finally:
                self._gripper = None

        if self._rtde_control is not None and self._owns_rtde_session:
            try:
                self._rtde_control.servoStop()
            except Exception:
                logger.exception("Failed to stop UR servo cleanly.")
            try:
                self._rtde_control.stopScript()
            except Exception:
                logger.exception("Failed to stop UR RTDE script cleanly.")

        if self._owns_rtde_session:
            self._rtde_control = None
            self._rtde_receive = None
        self._is_connected = False
        self._owns_rtde_session = True

    @check_if_not_connected
    def move_to_tcp_pose(
        self,
        tcp_pose: list[float] | tuple[float, float, float, float, float, float],
        speed: float | None = None,
        acceleration: float | None = None,
        gripper_target: float | None = None,
    ) -> None:
        assert self._rtde_control is not None

        try:
            self._rtde_control.servoStop()
        except Exception:
            logger.exception("Failed to stop UR servo cleanly before moveL.")

        self._rtde_control.moveL(
            list(tcp_pose),
            self.config.velocity if speed is None else speed,
            self.config.acceleration if acceleration is None else acceleration,
            False,
        )

        if gripper_target is not None and self.config.has_gripper and self._gripper is not None:
            gripper_target = min(max(float(gripper_target), 0.0), 1.0)
            self._gripper.set_position(int(round(gripper_target * 1000.0)))

    def _apply_safety_limits(self, action: RobotAction, current_obs: RobotObservation) -> RobotAction:
        if self.config.max_relative_target is None:
            return action

        goal_present = {
            key: (float(action[key]), float(current_obs[key]))
            for key in self.action_features
            if key in current_obs
        }
        safe_positions = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        return {**action, **safe_positions}

    def _apply_eef_safety_limits(self, action: RobotAction) -> RobotAction:
        assert self._rtde_receive is not None

        translation_cap = self.config.max_relative_translation
        rotation_cap = self.config.max_relative_rotation
        if self.config.max_relative_target is not None:
            if translation_cap is None and isinstance(self.config.max_relative_target, float):
                translation_cap = self.config.max_relative_target
            if rotation_cap is None and isinstance(self.config.max_relative_target, float):
                rotation_cap = self.config.max_relative_target

        if translation_cap is None and rotation_cap is None:
            return action

        current_pose = np.asarray(self._rtde_receive.getActualTCPPose(), dtype=np.float64)
        caps = {
            "tcp.x": translation_cap,
            "tcp.y": translation_cap,
            "tcp.z": translation_cap,
            "tcp.rx": rotation_cap,
            "tcp.ry": rotation_cap,
            "tcp.rz": rotation_cap,
        }
        goal_present = {
            key: (float(action[key]), float(current_pose[index]))
            for index, key in enumerate(self._tcp_action_names)
            if caps[key] is not None
        }
        if not goal_present:
            return action

        safe_positions = ensure_safe_goal_position(goal_present, {key: caps[key] for key in goal_present})
        return {**action, **safe_positions}

    def _make_gripper(self):
        try:
            from pymodbus.client.sync import ModbusTcpClient  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PGI support requires `pymodbus`. Install it in the current environment first."
            ) from exc

        return _PgiTcpGripper()

    def _read_gripper_position(self) -> float:
        if self._gripper is None:
            return 0.0

        try:
            position = float(self._gripper.read_current_position())
        except Exception:
            logger.exception("Failed to read PGI gripper position, returning 0.0.")
            return 0.0

        return min(max(position / 1000.0, 0.0), 1.0)
