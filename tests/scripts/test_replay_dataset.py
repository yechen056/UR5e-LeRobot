#!/usr/bin/env python

from argparse import Namespace

import numpy as np

from scripts.replay_dataset import (
    ReplayPlan,
    infer_action_mode,
    infer_bimanual,
    interpolate_actions,
    make_robot,
    quality_report,
    validate_ur5e_action_names,
)


def test_infers_bimanual_joint_replay_from_action_names():
    names = (
        "left_shoulder_pan.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_gripper.pos",
    )

    assert infer_action_mode(names) == "joint"
    assert infer_bimanual(names)


def test_infers_single_eef_replay_from_action_names():
    names = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz", "gripper.pos")

    assert infer_action_mode(names) == "eef"
    assert not infer_bimanual(names)


def test_rejects_non_ur_joint_schema():
    names = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    with np.testing.assert_raises(ValueError):
        validate_ur5e_action_names(names, "joint", False)


def test_preparation_interpolation_bounds_joint_and_gripper_steps():
    current = {"shoulder_pan.pos": 0.0, "gripper.pos": 0.0}
    target = {"shoulder_pan.pos": 0.31, "gripper.pos": 0.7}

    actions = interpolate_actions(current, target, "joint")
    trajectory = [current, *actions]

    assert actions[-1] == target
    assert max(
        abs(trajectory[index + 1]["shoulder_pan.pos"] - trajectory[index]["shoulder_pan.pos"])
        for index in range(len(trajectory) - 1)
    ) <= 0.03
    assert max(
        abs(trajectory[index + 1]["gripper.pos"] - trajectory[index]["gripper.pos"])
        for index in range(len(trajectory) - 1)
    ) <= 0.0500001


def test_quality_report_flags_large_action_jump():
    plan = ReplayPlan(
        action_names=("shoulder_pan.pos",),
        actions=np.asarray([[0.0], [0.5]], dtype=np.float64),
        action_mode="joint",
        bimanual=False,
        has_gripper=False,
        fps=15.0,
    )

    assert any("WARNING" in line for line in quality_report(plan))


def test_bimanual_robot_uses_project_hardware_defaults():
    args = Namespace(
        no_gripper=False,
        max_joint_step=0.12,
        max_translation_step=0.02,
        max_rotation_step=0.15,
        robot_ip="192.168.1.5",
        gripper_ip="192.168.1.7",
        gripper_port=8887,
        left_robot_ip="192.168.1.3",
        right_robot_ip="192.168.1.5",
        left_gripper_ip="192.168.1.8",
        left_gripper_port=8888,
        right_gripper_ip="192.168.1.7",
        right_gripper_port=8887,
    )
    plan = ReplayPlan(
        action_names=(
            "left_shoulder_pan.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_gripper.pos",
        ),
        actions=np.zeros((2, 4), dtype=np.float64),
        action_mode="joint",
        bimanual=True,
        has_gripper=True,
        fps=15.0,
    )

    robot = make_robot(args, plan)

    assert robot.left_arm.config.robot_ip == "192.168.1.3"
    assert robot.left_arm.config.gripper_ip == "192.168.1.8"
    assert robot.left_arm.config.gripper_port == 8888
    assert robot.right_arm.config.robot_ip == "192.168.1.5"
    assert robot.right_arm.config.gripper_ip == "192.168.1.7"
    assert robot.right_arm.config.gripper_port == 8887
