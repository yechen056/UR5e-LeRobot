#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lerobot.robots.ur5e_pgi import UR5ePGI
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    _capture_robot_start_poses,
    _move_robot_to_start_poses,
    record,
    record_loop,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators.keyboard import KeyboardUR5eTeleop
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


@pytest.mark.parametrize(
    ("already_calibrated", "recalibrate", "expected_calls"),
    ((False, False, 1), (True, False, 0), (True, True, 1)),
)
def test_teleoperate_gello_pair_calibration_trigger(
    already_calibrated, recalibrate, expected_calls
):
    cfg = TeleoperateConfig(
        robot=MockRobotConfig(),
        teleop=MockTeleopConfig(calibrated=already_calibrated),
        teleop_time_s=0.01,
        recalibrate=recalibrate,
    )
    with (
        patch("lerobot.scripts.lerobot_teleoperate._is_gello_ur5e_pair", return_value=True),
        patch("lerobot.scripts.lerobot_teleoperate._validate_gello_ur5e_pair"),
        patch("lerobot.scripts.lerobot_teleoperate._run_gello_ur5e_calibration") as calibrate_pair,
    ):
        teleoperate(cfg)

    assert calibrate_pair.call_count == expected_calls


def test_recalibrate_rejected_for_non_gello_pair():
    cfg = TeleoperateConfig(
        robot=MockRobotConfig(),
        teleop=MockTeleopConfig(),
        teleop_time_s=0.01,
        recalibrate=True,
    )

    with pytest.raises(ValueError, match="only supported"):
        teleoperate(cfg)


@pytest.mark.parametrize("terminal_event", ("delete_last_episode", "stop_episode"))
def test_manual_record_loop_preserves_terminal_event_for_outer_state_machine(terminal_event):
    events = {
        "delete_last_episode": terminal_event == "delete_last_episode",
        "stop_episode": terminal_event == "stop_episode",
        "exit_early": True,
    }

    record_loop(
        robot=object(),
        events=events,
        fps=30,
        teleop_action_processor=None,
        robot_action_processor=None,
        robot_observation_processor=None,
        manual_episode_control=True,
    )

    assert events[terminal_event]
    assert not events["exit_early"]


def test_manual_idle_loop_keeps_teleoperation_active_until_c(monkeypatch):
    robot = MockRobot(MockRobotConfig(static_values=[0.0, 0.0, 0.0], random_values=False))
    teleop = MockTeleop(MockTeleopConfig(static_values=[1.0, 2.0, 3.0], random_values=False))
    robot.connect()
    teleop.connect()
    events = {
        "start_episode": False,
        "delete_last_episode": False,
        "stop_episode": False,
        "exit_early": False,
    }
    sent_actions = []

    def send_action(action):
        sent_actions.append(action)
        events["start_episode"] = True
        return action

    monkeypatch.setattr(robot, "send_action", send_action)

    record_loop(
        robot=robot,
        events=events,
        fps=1000,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        teleop=teleop,
        manual_episode_control=True,
        wait_for_episode_start=True,
        record_frames=False,
    )

    assert len(sent_actions) == 1
    assert events["start_episode"]


def test_keyboard_start_pose_is_captured_once_and_used_for_automatic_return():
    robot = object.__new__(UR5ePGI)
    robot.config = SimpleNamespace(has_gripper=True)
    robot.get_tcp_pose = lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    robot._read_gripper_position = lambda: 0.75
    move_requests = []
    robot.move_to_tcp_pose = lambda pose, gripper_target=None: move_requests.append(
        (tuple(pose), gripper_target)
    )
    keyboard = object.__new__(KeyboardUR5eTeleop)

    start_poses = _capture_robot_start_poses(robot, keyboard)
    robot.get_tcp_pose = lambda: [9.0] * 6
    _move_robot_to_start_poses(robot, keyboard, start_poses)

    assert move_requests == [((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 0.75)]


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)
