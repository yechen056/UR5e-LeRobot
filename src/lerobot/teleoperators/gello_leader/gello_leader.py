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

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode, TorqueMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_gello_leader import GelloLeaderConfig

logger = logging.getLogger(__name__)


class GelloLeader(Teleoperator):
    config_class = GelloLeaderConfig
    name = "gello_leader"
    calibration_format_version = 2
    calibration_mode = "ur5e_joint_alignment"

    def __init__(self, config: GelloLeaderConfig):
        self.joint_offsets: np.ndarray | None = None
        self.gripper_open_close_rad: tuple[float, float] | None = None
        self._loaded_calibration_format_version: int | None = None
        self._loaded_calibration_mode: str | None = None
        super().__init__(config)
        self.config = config
        self._start_joints = np.asarray(self.config.start_joints, dtype=np.float64)
        self._joint_signs = np.asarray(self.config.joint_signs, dtype=np.float64)
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                motor_name: Motor(motor_id, model, MotorNormMode.DEGREES)
                for motor_name, (motor_id, model) in self.config.motors.items()
            },
        )
        self._joint_names = tuple(self.config.motors)
        self._arm_joint_names = tuple(name for name in self._joint_names if name != "gripper")

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self._joint_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return bool(
            self._loaded_calibration_format_version == self.calibration_format_version
            and self._loaded_calibration_mode == self.calibration_mode
            and self.joint_offsets is not None
            and self.joint_offsets.shape == (6,)
            and np.all(np.isfinite(self.joint_offsets))
            and self.gripper_open_close_rad is not None
            and len(self.gripper_open_close_rad) == 2
            and np.all(np.isfinite(self.gripper_open_close_rad))
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # PortHandler.openPort() opens the serial device using this value.
        self.bus.port_handler.baudrate = self.config.baudrate
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("No gello offset calibration found for %s", self)
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        raise RuntimeError(
            "GELLO calibration requires live UR5e joint positions. Run lerobot-teleoperate with a "
            "gello_leader/ur5e_pgi pair instead of calibrating the GELLO device by itself."
        )

    def calibrate_to_ur5e(self, ur5e_joint_positions: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """Align this GELLO's current pose with a UR5e's current six joint positions."""
        offsets, gripper_range = self._compute_ur5e_alignment(ur5e_joint_positions)
        self._apply_ur5e_alignment(offsets, gripper_range)

    def _compute_ur5e_alignment(
        self, ur5e_joint_positions: np.ndarray | list[float] | tuple[float, ...]
    ) -> tuple[np.ndarray, tuple[float, float]]:
        ur5e_joints = np.asarray(ur5e_joint_positions, dtype=np.float64)
        if ur5e_joints.shape != (6,):
            raise ValueError(f"Expected 6 UR5e joint positions, got shape {ur5e_joints.shape}.")
        if not np.all(np.isfinite(ur5e_joints)):
            raise ValueError("UR5e joint positions must all be finite.")

        raw_joint_radians = self._read_raw_joint_radians()
        if raw_joint_radians.shape != (7,) or not np.all(np.isfinite(raw_joint_radians)):
            raise ValueError("Expected 7 finite raw GELLO joint positions.")

        joint_offsets = raw_joint_radians[:6] - self._joint_signs * ur5e_joints

        raw_gripper_rad = float(raw_joint_radians[-1])
        gripper_open_close_rad = (
            raw_gripper_rad - math.radians(self.config.gripper_open_correction_deg),
            raw_gripper_rad - math.radians(self.config.gripper_close_delta_deg),
        )
        return joint_offsets, gripper_open_close_rad

    def _apply_ur5e_alignment(
        self, joint_offsets: np.ndarray, gripper_open_close_rad: tuple[float, float]
    ) -> None:
        self.joint_offsets = np.asarray(joint_offsets, dtype=np.float64)
        self.gripper_open_close_rad = tuple(float(value) for value in gripper_open_close_rad)
        self._loaded_calibration_format_version = self.calibration_format_version
        self._loaded_calibration_mode = self.calibration_mode
        self._save_calibration()
        logger.info("Calibration saved to %s", self.calibration_fpath)

    @check_if_not_connected
    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        self._reset_torque_mode()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        raw_joint_radians = self._read_raw_joint_radians()

        if self.joint_offsets is None:
            raise RuntimeError("Gello leader joint offsets are not calibrated.")
        if self.gripper_open_close_rad is None:
            raise RuntimeError("Gello leader gripper open/close calibration is missing.")

        calibrated_arm = (raw_joint_radians[:6] - self.joint_offsets) * self._joint_signs
        action = {
            f"{motor_name}.pos": float(value_rad)
            for motor_name, value_rad in zip(self._arm_joint_names, calibrated_arm, strict=True)
        }
        action["gripper.pos"] = self._normalize_gripper(raw_joint_radians[-1])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read action: %.1fms", self, dt_ms)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        del feedback
        raise NotImplementedError("Feedback is not implemented for Gello leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)

    def _reset_torque_mode(self) -> None:
        self.bus.write("Torque_Enable", "gripper", TorqueMode.DISABLED.value, normalize=False)
        for motor_name in self._arm_joint_names:
            self.bus.write(
                "Operating_Mode", motor_name, OperatingMode.EXTENDED_POSITION.value, normalize=False
            )
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value, normalize=False)

    def _read_raw_positions(self) -> np.ndarray:
        raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        return np.asarray([raw_positions[motor_name] for motor_name in self._joint_names], dtype=np.int32)

    def _read_raw_joint_radians(self) -> np.ndarray:
        raw_positions = self._read_raw_positions()
        raw_joint_radians = []
        for motor_name, position in zip(self._joint_names, raw_positions, strict=True):
            model = self.bus.motors[motor_name].model
            max_res = self.bus.model_resolution_table[model] - 1
            raw_joint_radians.append(float(position) / max_res * 2 * math.pi)
        return np.asarray(raw_joint_radians, dtype=np.float64)

    def _normalize_gripper(self, raw_gripper_rad: float) -> float:
        assert self.gripper_open_close_rad is not None
        open_rad, close_rad = self.gripper_open_close_rad
        denominator = open_rad - close_rad
        if abs(denominator) < 1e-6:
            return 0.0
        normalized = (raw_gripper_rad - close_rad) / denominator
        return float(min(max(normalized, 0.0), 1.0))

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f:
            data = json.load(f)

        self._loaded_calibration_format_version = data.get("format_version")
        self._loaded_calibration_mode = data.get("calibration_mode")
        joint_offsets = data.get("joint_offsets")
        self.joint_offsets = (
            np.asarray(joint_offsets, dtype=np.float64) if joint_offsets is not None else None
        )
        gripper_open_close = data.get("gripper_open_close_rad")
        self.gripper_open_close_rad = tuple(gripper_open_close) if gripper_open_close is not None else None

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        payload = {
            "format_version": self.calibration_format_version,
            "calibration_mode": self.calibration_mode,
            "joint_offsets": self.joint_offsets.tolist() if self.joint_offsets is not None else None,
            "joint_signs": self._joint_signs.tolist(),
            "start_joints": self._start_joints.tolist(),
            "gripper_open_close_rad": list(self.gripper_open_close_rad)
            if self.gripper_open_close_rad is not None
            else None,
        }
        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2)
