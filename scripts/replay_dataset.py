#!/usr/bin/env python

"""Safely replay one local LeRobot episode on a single or bimanual UR5e setup."""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_ur5e_pgi import BiUR5ePGI, BiUR5ePGIConfig
from lerobot.robots.ur5e_pgi import UR5ePGI, UR5ePGIConfig, UR5ePGIConfigBase
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep


class _ReplayClampWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Relative goal position magnitude had to be clamped to be safe." not in record.getMessage()


def silence_replay_clamp_warnings() -> None:
    logging.getLogger().addFilter(_ReplayClampWarningFilter())


@dataclass(frozen=True)
class ReplayPlan:
    action_names: tuple[str, ...]
    actions: np.ndarray
    action_mode: str
    bimanual: bool
    has_gripper: bool
    fps: float


def infer_action_mode(action_names: Sequence[str]) -> str:
    has_tcp = any(name.removeprefix("left_").removeprefix("right_").startswith("tcp.") for name in action_names)
    return "eef" if has_tcp else "joint"


def infer_bimanual(action_names: Sequence[str]) -> bool:
    return any(name.startswith("left_") for name in action_names) and any(
        name.startswith("right_") for name in action_names
    )


def validate_ur5e_action_names(action_names: Sequence[str], action_mode: str, bimanual: bool) -> None:
    if action_mode == "joint":
        base_names = (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow.pos",
            "wrist_1.pos",
            "wrist_2.pos",
            "wrist_3.pos",
        )
    else:
        base_names = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")
    prefixes = ("left_", "right_") if bimanual else ("",)
    required = {f"{prefix}{name}" for prefix in prefixes for name in base_names}
    present = set(action_names)
    missing = sorted(required - present)
    allowed = required | {f"{prefix}gripper.pos" for prefix in prefixes}
    unexpected = sorted(present - allowed)
    if missing or unexpected:
        raise ValueError(
            "Dataset actions are not compatible with UR5e replay. "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_replay_plan(
    dataset_root: str | Path,
    episode: int,
    repo_id: str = "local/replay",
    playback_speed: float = 1.0,
    filter_static: bool = False,
) -> ReplayPlan:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir() or not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset directory: {root}")
    if episode < 0:
        raise ValueError("episode must be non-negative")
    if not 0.0 < playback_speed <= 1.0:
        raise ValueError("playback_speed must be in (0, 1]")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=[episode],
        download_videos=False,
    )
    if dataset.num_frames == 0:
        raise ValueError(f"Episode {episode} contains no frames.")

    feature = dataset.features.get(ACTION)
    if feature is None or not feature.get("names"):
        raise ValueError("Dataset does not contain named action features.")
    action_names = tuple(feature["names"])
    selected = dataset.select_columns(ACTION)
    actions = np.asarray([selected[index][ACTION] for index in range(dataset.num_frames)], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(action_names):
        raise ValueError(
            f"Action shape {actions.shape} does not match {len(action_names)} action names."
        )
    if not np.all(np.isfinite(actions)):
        bad = np.argwhere(~np.isfinite(actions))[0]
        raise ValueError(f"Non-finite action at frame={bad[0]}, dimension={bad[1]}.")
    if filter_static and len(actions) > 1:
        moving = np.max(np.abs(np.diff(actions, axis=0)), axis=1) > 1e-6
        keep = np.concatenate(([True], moving))
        actions = actions[keep]

    action_mode = infer_action_mode(action_names)
    bimanual = infer_bimanual(action_names)
    validate_ur5e_action_names(action_names, action_mode, bimanual)
    return ReplayPlan(
        action_names=action_names,
        actions=actions,
        action_mode=action_mode,
        bimanual=bimanual,
        has_gripper=any(name.endswith("gripper.pos") for name in action_names),
        fps=float(dataset.fps) * playback_speed,
    )


def action_dict(plan: ReplayPlan, index: int) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(plan.action_names, plan.actions[index], strict=True)
    }


def quality_report(plan: ReplayPlan) -> list[str]:
    duration = len(plan.actions) / plan.fps
    lines = [
        f"mode={plan.action_mode}, bimanual={plan.bimanual}, gripper={plan.has_gripper}",
        f"frames={len(plan.actions)}, replay_fps={plan.fps:.2f}, duration={duration:.2f}s",
    ]
    if len(plan.actions) <= 1:
        return lines

    deltas = np.abs(np.diff(plan.actions, axis=0))
    max_indices = np.argmax(deltas, axis=0)
    largest = sorted(
        (
            (float(deltas[frame, dim]), plan.action_names[dim], int(frame + 1))
            for dim, frame in enumerate(max_indices)
        ),
        reverse=True,
    )
    lines.append("largest per-frame jumps:")
    lines.extend(f"  {name}: {value:.5f} at frame {frame}" for value, name, frame in largest[:8])

    warning_threshold = 0.35 if plan.action_mode == "joint" else 0.10
    suspicious = [(value, name, frame) for value, name, frame in largest if value > warning_threshold]
    if suspicious:
        lines.append(
            f"WARNING: {len(suspicious)} action dimensions exceed the {warning_threshold:g} "
            "per-frame jump threshold. Inspect the episode before real replay."
        )
    return lines


def _current_action(
    robot: UR5ePGI | BiUR5ePGI,
    plan: ReplayPlan,
    fallback_action: dict[str, float] | None = None,
) -> dict[str, float]:
    if plan.action_mode == "joint":
        observation = robot.get_observation()
        return {
            name: float(observation[name])
            if name in observation
            else float((fallback_action or {})[name])
            for name in plan.action_names
        }

    def arm_tcp(arm: UR5ePGI, prefix: str) -> dict[str, float]:
        tcp = arm.get_tcp_pose()
        values = {
            f"{prefix}tcp.x": float(tcp[0]),
            f"{prefix}tcp.y": float(tcp[1]),
            f"{prefix}tcp.z": float(tcp[2]),
            f"{prefix}tcp.rx": float(tcp[3]),
            f"{prefix}tcp.ry": float(tcp[4]),
            f"{prefix}tcp.rz": float(tcp[5]),
        }
        gripper_name = f"{prefix}gripper.pos"
        if gripper_name in plan.action_names:
            values[gripper_name] = arm._read_gripper_position()
        return values

    if isinstance(robot, UR5ePGI):
        values = arm_tcp(robot, "")
    else:
        values = {**arm_tcp(robot.left_arm, "left_"), **arm_tcp(robot.right_arm, "right_")}
    return {name: values[name] for name in plan.action_names}


def interpolation_steps(
    current: dict[str, float], target: dict[str, float], action_mode: str
) -> int:
    steps = 1
    for name, target_value in target.items():
        delta = abs(target_value - current[name])
        if name.endswith("gripper.pos"):
            max_step = 0.05
        elif action_mode == "joint":
            max_step = 0.03
        elif name.endswith(("tcp.x", "tcp.y", "tcp.z")):
            max_step = 0.005
        else:
            max_step = 0.03
        steps = max(steps, int(math.ceil(delta / max_step)))
    return steps


def interpolate_actions(
    current: dict[str, float], target: dict[str, float], action_mode: str
) -> list[dict[str, float]]:
    count = interpolation_steps(current, target, action_mode)
    return [
        {
            name: float(current[name] + (target[name] - current[name]) * step / count)
            for name in target
        }
        for step in range(1, count + 1)
    ]


def make_robot(args: argparse.Namespace, plan: ReplayPlan) -> UR5ePGI | BiUR5ePGI:
    has_gripper = plan.has_gripper and not args.no_gripper

    def arm_config(robot_ip: str, gripper_ip: str, gripper_port: int) -> UR5ePGIConfigBase:
        return UR5ePGIConfigBase(
            robot_ip=robot_ip,
            action_mode=plan.action_mode,
            gripper_ip=gripper_ip if has_gripper else None,
            gripper_port=gripper_port,
            max_relative_target=args.max_joint_step if plan.action_mode == "joint" else None,
            max_relative_translation=args.max_translation_step if plan.action_mode == "eef" else None,
            max_relative_rotation=args.max_rotation_step if plan.action_mode == "eef" else None,
        )

    if plan.bimanual:
        return BiUR5ePGI(
            BiUR5ePGIConfig(
                id="dataset_replay",
                left_arm_config=arm_config(args.left_robot_ip, args.left_gripper_ip, args.left_gripper_port),
                right_arm_config=arm_config(
                    args.right_robot_ip, args.right_gripper_ip, args.right_gripper_port
                ),
            )
        )
    return UR5ePGI(
        UR5ePGIConfig(
            id="dataset_replay",
            **vars(arm_config(args.robot_ip, args.gripper_ip, args.gripper_port)),
        )
    )


def replay(args: argparse.Namespace) -> None:
    plan = load_replay_plan(
        args.dataset_root,
        args.episode,
        repo_id=args.repo_id,
        playback_speed=args.playback_speed,
        filter_static=args.filter_static,
    )
    print(f"Dataset: {Path(args.dataset_root).expanduser().resolve()}")
    print(f"Episode: {args.episode}")
    for line in quality_report(plan):
        print(line)

    if args.dry_run:
        print("dry-run: robot connection and motion skipped.")
        return

    if not args.yes:
        confirmation = input(
            "Real robot replay will move the arm(s). Clear the workspace and type REPLAY to continue: "
        )
        if confirmation.strip() != "REPLAY":
            print("Replay cancelled.")
            return

    robot = make_robot(args, plan)
    robot.connect()
    try:
        first = action_dict(plan, 0)
        current = _current_action(robot, plan, fallback_action=first)
        preparation = interpolate_actions(current, first, plan.action_mode)
        print(f"Moving to first action with {len(preparation)} safe interpolation steps...")
        for action in preparation:
            start = time.perf_counter()
            robot.send_action(action)
            precise_sleep(max(1.0 / args.preparation_fps - (time.perf_counter() - start), 0.0))

        print("Starting episode replay. Press Ctrl-C for emergency stop.")
        for index in range(len(plan.actions)):
            start = time.perf_counter()
            robot.send_action(action_dict(plan, index))
            if index % max(1, int(plan.fps)) == 0:
                print(f"  frame {index}/{len(plan.actions) - 1}")
            precise_sleep(max(1.0 / plan.fps - (time.perf_counter() - start), 0.0))
        print("Replay completed.")
    except KeyboardInterrupt:
        print("Replay interrupted by user.")
    finally:
        robot.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Local LeRobot dataset directory")
    parser.add_argument("episode", type=int, help="Episode index")
    parser.add_argument("--repo-id", default="local/replay", help="Logical dataset repo id")
    parser.add_argument("--playback-speed", type=float, default=0.5, help="Speed multiplier in (0, 1]")
    parser.add_argument("--filter-static", action="store_true", help="Drop exactly repeated action frames")
    parser.add_argument("--dry-run", action="store_true", help="Inspect data without connecting to robots")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive real-motion confirmation")
    parser.add_argument("--preparation-fps", type=float, default=30.0)
    parser.add_argument("--max-joint-step", type=float, default=0.12)
    parser.add_argument("--max-translation-step", type=float, default=0.02)
    parser.add_argument("--max-rotation-step", type=float, default=0.15)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--robot-ip", default="192.168.1.5")
    parser.add_argument("--gripper-ip", default="192.168.1.7")
    parser.add_argument("--gripper-port", type=int, default=8887)
    parser.add_argument("--left-robot-ip", default="192.168.1.3")
    parser.add_argument("--right-robot-ip", default="192.168.1.5")
    parser.add_argument("--left-gripper-ip", default="192.168.1.8")
    parser.add_argument("--left-gripper-port", type=int, default=8888)
    parser.add_argument("--right-gripper-ip", default="192.168.1.7")
    parser.add_argument("--right-gripper-port", type=int, default=8887)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    silence_replay_clamp_warnings()
    if args.preparation_fps <= 0:
        parser.error("preparation_fps must be positive")
    try:
        replay(args)
    except (FileNotFoundError, IndexError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
