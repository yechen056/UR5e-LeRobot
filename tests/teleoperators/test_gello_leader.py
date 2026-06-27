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

import numpy as np
import pytest

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.bi_gello_leader import BiGelloLeader, BiGelloLeaderConfig
from lerobot.teleoperators.gello_leader import GelloLeader, GelloLeaderConfig


def test_gello_leader_config_is_registered():
    assert "gello_leader" in GelloLeaderConfig.get_known_choices()


def test_make_gello_leader_from_config():
    teleop = make_teleoperator_from_config(GelloLeaderConfig(port="/dev/ttyUSB0"))

    assert isinstance(teleop, GelloLeader)
    assert teleop.config.baudrate == 3_000_000
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
    assert teleop.left_arm.config.baudrate == 3_000_000
    assert teleop.right_arm.config.baudrate == 3_000_000
    assert "left_shoulder_pan.pos" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features


def test_gello_applies_configured_baudrate_before_connect(tmp_path, monkeypatch):
    teleop = GelloLeader(
        GelloLeaderConfig(
            port="/dev/ttyUSB0", id="paired", calibration_dir=tmp_path, baudrate=3_000_000
        )
    )
    observed_baudrates = []
    teleop.bus.port_handler.is_open = False
    monkeypatch.setattr(
        teleop.bus,
        "connect",
        lambda: observed_baudrates.append(teleop.bus.port_handler.baudrate),
    )
    monkeypatch.setattr(teleop, "configure", lambda: None)

    teleop.connect(calibrate=False)

    assert observed_baudrates == [3_000_000]


def test_gello_calibrates_to_live_ur5e_joint_positions(tmp_path, monkeypatch):
    joint_signs = (1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    teleop = GelloLeader(
        GelloLeaderConfig(
            port="/dev/ttyUSB0",
            id="paired",
            calibration_dir=tmp_path,
            joint_signs=joint_signs,
            gripper_open_correction_deg=0.0,
            gripper_close_delta_deg=42.0,
        )
    )
    ur5e_joints = np.asarray([0.4, -1.2, 1.7, -0.8, 0.3, 2.1])
    raw_gello = np.asarray([4.1, 2.3, 5.2, -1.0, 3.7, 7.4, 1.5])
    monkeypatch.setattr(teleop, "_read_raw_joint_radians", lambda: raw_gello.copy())

    teleop.calibrate_to_ur5e(ur5e_joints)

    calibrated = (raw_gello[:6] - teleop.joint_offsets) * np.asarray(joint_signs)
    np.testing.assert_allclose(calibrated, ur5e_joints)
    assert teleop.is_calibrated
    assert teleop._normalize_gripper(raw_gello[-1]) == pytest.approx(1.0)
    close_rad = raw_gello[-1] - np.deg2rad(42.0)
    assert teleop._normalize_gripper(close_rad) == pytest.approx(0.0)
    assert teleop._normalize_gripper(raw_gello[-1] + 1.0) == pytest.approx(1.0)
    assert teleop._normalize_gripper(close_rad - 1.0) == pytest.approx(0.0)

    saved = json.loads((tmp_path / "paired.json").read_text())
    assert saved["format_version"] == GelloLeader.calibration_format_version
    assert saved["calibration_mode"] == GelloLeader.calibration_mode


def test_gello_legacy_calibration_is_invalidated(tmp_path):
    legacy_path = tmp_path / "paired.json"
    legacy_path.write_text(
        json.dumps(
            {
                "joint_offsets": [0.0] * 6,
                "gripper_open_close_rad": [1.0, 0.0],
            }
        )
    )

    teleop = GelloLeader(
        GelloLeaderConfig(port="/dev/ttyUSB0", id="paired", calibration_dir=tmp_path)
    )

    assert not teleop.is_calibrated


@pytest.mark.parametrize(
    "ur5e_joints",
    ([0.0] * 5, [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]),
)
def test_gello_rejects_invalid_ur5e_joint_positions(tmp_path, monkeypatch, ur5e_joints):
    teleop = GelloLeader(
        GelloLeaderConfig(port="/dev/ttyUSB0", id="paired", calibration_dir=tmp_path)
    )
    monkeypatch.setattr(teleop, "_read_raw_joint_radians", lambda: np.zeros(7))

    with pytest.raises(ValueError):
        teleop.calibrate_to_ur5e(ur5e_joints)


def test_standalone_gello_calibration_requires_ur5e_reference(tmp_path):
    teleop = GelloLeader(
        GelloLeaderConfig(port="/dev/ttyUSB0", id="paired", calibration_dir=tmp_path)
    )

    with pytest.raises(RuntimeError, match="live UR5e joint positions"):
        teleop.calibrate()
