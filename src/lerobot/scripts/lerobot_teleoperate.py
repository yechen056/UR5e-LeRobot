# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import os
import select
import sys
import termios
import time
import tty
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    ur5e_pgi,
)
from lerobot.robots.bi_ur5e_pgi import BiUR5ePGI  # noqa: F401
from lerobot.robots.ur5e_pgi import UR5ePGI, UR5ePGIConfig  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_gello_leader,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    gello_leader,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    quest,
    reachy2_teleoperator,
    so_leader,
    spnav,
    unitree_g1,
)
from lerobot.teleoperators.quest import QuestTeleop, QuestTeleopConfig  # noqa: F401
from lerobot.teleoperators.spnav import SpnavTeleop, SpnavTeleopConfig  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)
UR5E_PGI_HOME_TCP_POSE = (-0.11130, -0.48927, 0.22326, 3.152, -0.007, -0.001)


class _TerminalKeyReader:
    def __init__(self):
        self._enabled = False
        self._fd = None
        self._old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self

        try:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except Exception:
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        if not self._enabled:
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None

        return sys.stdin.read(1)


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    profile_loop = os.getenv("LEROBOT_TELEOP_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    profile_window = 30
    profile_acc = {"obs": 0.0, "action": 0.0, "send": 0.0, "loop": 0.0, "count": 0}
    with _TerminalKeyReader() as key_reader:
        while True:
            loop_start = time.perf_counter()

            if isinstance(robot, UR5ePGI):
                key = key_reader.read_key()
                if key is not None and key.lower() == "h":
                    logger.info("`h` pressed. Moving UR5e to the configured home TCP pose.")
                    robot.move_to_tcp_pose(UR5E_PGI_HOME_TCP_POSE, gripper_target=1.0)
                    continue

            # Get robot observation
            # Not really needed for now other than for visualization
            # teleop_action_processor can take None as an observation
            # given that it is the identity processor as default
            obs_start = time.perf_counter()
            obs = robot.get_observation()
            obs_dt = time.perf_counter() - obs_start

            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Get teleop action
            action_start = time.perf_counter()
            raw_action = teleop.get_action()
            action_dt = time.perf_counter() - action_start

            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))

            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

            # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
            send_start = time.perf_counter()
            _ = robot.send_action(robot_action_to_send)
            send_dt = time.perf_counter() - send_start

            if display_data:
                # Process robot observation through pipeline
                obs_transition = robot_observation_processor(obs)

                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            if profile_loop:
                profile_acc["obs"] += obs_dt
                profile_acc["action"] += action_dt
                profile_acc["send"] += send_dt
                profile_acc["loop"] += loop_s
                profile_acc["count"] += 1

                if profile_acc["count"] >= profile_window:
                    avg_obs_ms = profile_acc["obs"] / profile_acc["count"] * 1e3
                    avg_action_ms = profile_acc["action"] / profile_acc["count"] * 1e3
                    avg_send_ms = profile_acc["send"] / profile_acc["count"] * 1e3
                    avg_loop_ms = profile_acc["loop"] / profile_acc["count"] * 1e3
                    avg_hz = profile_acc["count"] / profile_acc["loop"]
                    logger.info(
                        "Teleop profile avg over %d loops | obs=%.2fms action=%.2fms send=%.2fms loop=%.2fms (%.1f Hz)",
                        profile_acc["count"],
                        avg_obs_ms,
                        avg_action_ms,
                        avg_send_ms,
                        avg_loop_ms,
                        avg_hz,
                    )
                    profile_acc = {"obs": 0.0, "action": 0.0, "send": 0.0, "loop": 0.0, "count": 0}

            if duration is not None and time.perf_counter() - start >= duration:
                return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()

    if (
        isinstance(teleop, SpnavTeleop)
        and isinstance(robot, UR5ePGI)
        and teleop.config.robot_ip == robot.config.robot_ip
    ):
        robot._rtde_control = teleop._rtde_control
        robot._rtde_receive = teleop._rtde_receive
        robot._owns_rtde_session = False
    elif isinstance(teleop, QuestTeleop):
        if (
            isinstance(robot, UR5ePGI)
            and not teleop.config.bimanual
            and len(teleop._arms) == 1
            and teleop._arms[0].robot_ip == robot.config.robot_ip
        ):
            robot._rtde_control = teleop._arms[0].rtde_control
            robot._rtde_receive = teleop._arms[0].rtde_receive
            robot._owns_rtde_session = False
        elif isinstance(robot, BiUR5ePGI) and teleop.config.bimanual:
            arms_by_prefix = {arm.prefix: arm for arm in teleop._arms}
            left_arm = arms_by_prefix.get("left_")
            right_arm = arms_by_prefix.get("right_")
            if left_arm is not None and left_arm.robot_ip == robot.left_arm.config.robot_ip:
                robot.left_arm._rtde_control = left_arm.rtde_control
                robot.left_arm._rtde_receive = left_arm.rtde_receive
                robot.left_arm._owns_rtde_session = False
            if right_arm is not None and right_arm.robot_ip == robot.right_arm.config.robot_ip:
                robot.right_arm._rtde_control = right_arm.rtde_control
                robot.right_arm._rtde_receive = right_arm.rtde_receive
                robot.right_arm._owns_rtde_session = False

    robot.connect()

    if isinstance(teleop, SpnavTeleop) and isinstance(robot, UR5ePGI):
        teleop._gripper_target = 1.0
        if robot.config.has_gripper and robot._gripper is not None:
            robot._gripper.set_position(1000)
            logger.info("Opened the PGI gripper for teleoperation startup.")
    elif isinstance(teleop, QuestTeleop):
        if isinstance(robot, UR5ePGI):
            for arm in teleop._arms:
                arm.gripper_target = 1.0
            if robot.config.has_gripper and robot._gripper is not None:
                robot._gripper.set_position(1000)
                logger.info("Opened the PGI gripper for Quest teleoperation startup.")
        elif isinstance(robot, BiUR5ePGI):
            for arm in teleop._arms:
                arm.gripper_target = 1.0
            if robot.left_arm.config.has_gripper and robot.left_arm._gripper is not None:
                robot.left_arm._gripper.set_position(1000)
            if robot.right_arm.config.has_gripper and robot.right_arm._gripper is not None:
                robot.right_arm._gripper.set_position(1000)
            logger.info("Opened the PGI grippers for Quest teleoperation startup.")

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        robot.disconnect()
        teleop.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
