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

    def __init__(self, config: GelloLeaderConfig):
        self.joint_offsets: np.ndarray | None = None
        self.gripper_open_close_rad: tuple[float, float] | None = None
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
        return (
            self.joint_offsets is not None
            and self.joint_offsets.shape == (6,)
            and self.gripper_open_close_rad is not None
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("No gello offset calibration found for %s", self)
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        if self.is_calibrated:
            user_input = input(
                f"Press ENTER to use the existing offset calibration for {self.id}, or type 'c' and press ENTER to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                return

        logger.info("Running gello_get_offset-style calibration for %s", self)
        self.bus.disable_torque()
        self._reset_torque_mode()
        input(
            "\nCalibration: start pose\n"
            "Place the 6 arm joints at the configured start_joints pose and put the gripper in its open reference pose, then press ENTER..."
        )

        raw_joint_radians = self._read_raw_joint_radians()
        self.joint_offsets = self._estimate_joint_offsets(raw_joint_radians[:6])

        raw_gripper_rad = float(raw_joint_radians[-1])
        self.gripper_open_close_rad = (
            raw_gripper_rad - math.radians(self.config.gripper_open_correction_deg),
            raw_gripper_rad - math.radians(self.config.gripper_close_delta_deg),
        )

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

    def _estimate_joint_offsets(self, arm_joint_radians: np.ndarray) -> np.ndarray:
        best_offsets = []
        candidate_offsets = np.linspace(
            -self.config.offset_search_range_pi * math.pi,
            self.config.offset_search_range_pi * math.pi,
            self.config.offset_search_range_pi * 4 + 1,
        )
        for joint_index, current_joint in enumerate(arm_joint_radians):
            best_offset = 0.0
            best_error = float("inf")
            for offset in candidate_offsets:
                calibrated_joint = self._joint_signs[joint_index] * (current_joint - offset)
                error = abs(calibrated_joint - self._start_joints[joint_index])
                if error < best_error:
                    best_error = error
                    best_offset = float(offset)
            best_offsets.append(best_offset)
        return np.asarray(best_offsets, dtype=np.float64)

    def _normalize_gripper(self, raw_gripper_rad: float) -> float:
        assert self.gripper_open_close_rad is not None
        open_rad, close_rad = self.gripper_open_close_rad
        denominator = close_rad - open_rad
        if abs(denominator) < 1e-6:
            return 0.0
        normalized = (raw_gripper_rad - open_rad) / denominator
        return float(min(max(normalized, 0.0), 1.0))

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f:
            data = json.load(f)

        joint_offsets = data.get("joint_offsets")
        self.joint_offsets = (
            np.asarray(joint_offsets, dtype=np.float64) if joint_offsets is not None else None
        )
        gripper_open_close = data.get("gripper_open_close_rad")
        self.gripper_open_close_rad = tuple(gripper_open_close) if gripper_open_close is not None else None

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        payload = {
            "joint_offsets": self.joint_offsets.tolist() if self.joint_offsets is not None else None,
            "joint_signs": self._joint_signs.tolist(),
            "start_joints": self._start_joints.tolist(),
            "gripper_open_close_rad": list(self.gripper_open_close_rad)
            if self.gripper_open_close_rad is not None
            else None,
        }
        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2)
