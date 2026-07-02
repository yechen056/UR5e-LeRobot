<div align="center">
  <h1 align="center"> UR5e-LeRobot </h1>
</div>

<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

# 📖 Introduction

**UR5e-LeRobot (UR5e + PGI)** is a practical real-world robotics extension focused on lowering the barrier to data-driven robot learning with **LeRobot**.

We focus on an end-to-end workflow for real hardware, from teleoperation to deployment:

- 🦾 **Robot Setup Support**: Single-arm and bimanual UR5e setups with PGI gripper support.
- 🕹️ **Teleoperation Pipelines**: SpaceMouse (`spnav`), Quest 3 VR (`quest`), GELLO (`gello_leader`, `bi_gello_leader`), and Keyboard (`keyboard_ur5e`) for UR5e.
- 📦 **Data-to-Policy Workflow**: Unified examples for data collection, policy training, and real-world evaluation.
- 💻 **LeRobot-Native Experience**: Keep the upstream CLI workflow so teams can move quickly without heavy customization.

| Category | Supported Options |
|---|---|
| Hardware | `UR5e`, `PGI gripper` |
| Robot types | `Single UR5e`, `Bimanual UR5e` |
| Teleoperation backends | `SpaceMouse`, `Quest 3 (VR)`, `Gello`, `Keyboard` |

> [!IMPORTANT]
> Keep `action_mode` consistent across data collection and evaluation (`joint` or `eef`).
> If not specified, default is `joint` (`joint-space`).
>
> Data collection:
>
> - `--robot.action_mode=joint --teleop.action_mode=joint`
> - `--robot.action_mode=eef --teleop.action_mode=eef`
>
> Evaluation:
>
> - joint-space policy -> `--robot.action_mode=joint`
> - eef-space policy -> `--robot.action_mode=eef`

> [!NOTE]
> We have validated and tuned hyperparameters on this hardware stack for five policies: `ACT`, `Diffusion Policy`, `VQ-BeT`, `MultiTaskDiT`, `pi0`, and `pi05`.
> We will continue to publish tested training commands and recommended hyperparameters for additional policies in future updates.

## 🖥️ Real-World Demos

<p align="center">
  <img src="docs/teleop.gif" alt="Teleoperation demo">
</p>

<p align="center">
  <img src="docs/baselines.gif" alt="Baseline policy demo">
</p>

## 📢 Update

- **2026/06/28**, Extended the optional data replay support for validating collected demonstrations before policy training.
- **2026/06/27**, Updated the data collection pipeline, resolved environment dependency issues, and expanded the training and teleoperation examples with support for more policy configurations in single-arm and bimanual UR5e.
- **2026/04/13**, Added keyboard teleoperation support and a keyboard-based data collection branch.
- **2026/04/12**, We released the **UR5e-LeRobot** project.

# 🛠️ Installation

```bash
git clone --recursive https://github.com/yechen056/UR5e-LeRobot.git
cd UR5e-LeRobot

conda create -y -n ur5e_lerobot python=3.12
conda activate ur5e_lerobot
conda install -y -c conda-forge ffmpeg libstdcxx-ng

python -m pip install --upgrade pip
pip install -e ".[pi,peft,multi_task_dit,intelrealsense]"
pip install ur-rtde==1.6.3 pymodbus==2.5.3 spnav==0.9
pip install -e third_party/oculus_reader
```

## Verify The Installation

```bash
lerobot-info
python -c "import rtde_control, rtde_receive; from pymodbus.client.sync import ModbusTcpClient; from oculus_reader.reader import OculusReader; import pyrealsense2; from torchcodec.decoders import VideoDecoder; import transformers, peft; print('UR5e deps OK')"
python -c "from lerobot.teleoperators.spnav.teleop_spnav import _load_spnav_backend; backend = _load_spnav_backend(); print('SpaceMouse backend OK:', backend.__name__)"
```

# 🕹️ Teleoperation

This section provides reference commands for common teleoperation modes.

## Single-Arm Teleoperation

- `SpaceMouse`

```bash
lerobot-teleoperate \
  --robot.type=ur5e_pgi \
  --robot.action_mode=eef \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --teleop.type=spnav \
  --teleop.robot_ip=192.168.1.5 \
  --teleop.action_mode=eef \
  --fps=60
```

- `Keyboard`

```bash
lerobot-teleoperate \
  --robot.type=ur5e_pgi \
  --robot.action_mode=eef \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --teleop.type=keyboard_ur5e \
  --fps=30
```

Keyboard mapping: `↑/↓` move along X, `←/→` move along Y, `left shift` / `right shift` move along Z, and `ctrl_l` / `ctrl_r` close/open the gripper.

- `GELLO`

GELLO uses joint-space teleoperation. The first run calibrates the GELLO against the live UR5e joint positions; use `--recalibrate=true` whenever you want to overwrite the saved calibration.

```bash
lerobot-teleoperate \
  --robot.type=ur5e_pgi \
  --robot.action_mode=joint \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --teleop.type=gello_leader \
  --teleop.id=right_gello \
  --teleop.port=/dev/ttyUSB0 \
  --recalibrate=true \
  --fps=60
```

For normal teleoperation after calibration, run the same command without `--recalibrate=true`.

- `VR`

```bash
lerobot-teleoperate \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --teleop.type=quest \
  --teleop.robot_ip=192.168.1.5 \
  --fps=60
```

## Bimanual Teleoperation

- `GELLO`

Each GELLO arm needs its own serial port. With `--teleop.id=dual_gello`, calibration is saved separately for `dual_gello_left` and `dual_gello_right`.

```bash
lerobot-teleoperate \
  --robot.type=bi_ur5e_pgi \
  --robot.left_arm_config.action_mode=joint \
  --robot.left_arm_config.robot_ip=192.168.1.3 \
  --robot.left_arm_config.gripper_ip=192.168.1.8 \
  --robot.left_arm_config.gripper_port=8888 \
  --robot.right_arm_config.action_mode=joint \
  --robot.right_arm_config.robot_ip=192.168.1.5 \
  --robot.right_arm_config.gripper_ip=192.168.1.7 \
  --robot.right_arm_config.gripper_port=8887 \
  --teleop.type=bi_gello_leader \
  --teleop.id=dual_gello \
  --teleop.left_arm_config.port=/dev/ttyUSB0 \
  --teleop.right_arm_config.port=/dev/ttyUSB1 \
  --recalibrate=true \
  --fps=60
```

During calibration, keep the UR5e arms still, place the GELLO arms at the same six-joint poses as the robots, fully open the GELLO grippers, then press `Enter`.

- `VR`

```bash
lerobot-teleoperate \
  --robot.type=bi_ur5e_pgi \
  --robot.left_arm_config.robot_ip=192.168.1.3 \
  --robot.left_arm_config.gripper_ip=192.168.1.8 \
  --robot.left_arm_config.gripper_port=8888 \
  --robot.right_arm_config.robot_ip=192.168.1.5 \
  --robot.right_arm_config.gripper_ip=192.168.1.7 \
  --robot.right_arm_config.gripper_port=8887 \
  --teleop.type=quest \
  --teleop.bimanual=true \
  --teleop.left_robot_ip=192.168.1.3 \
  --teleop.right_robot_ip=192.168.1.5 \
  --fps=60
```

# ⚙️ Data Collection

Use `lerobot-record` to collect demonstrations using teleoperation together with front and wrist camera streams.
Before recording, check the camera names and device paths:

```bash
lerobot-find-cameras opencv
lerobot-find-cameras realsense
```

After finding the cameras, update the camera IDs, serial numbers, or device paths in the collect configs under `configs/`.

## Single-Arm Data Collection

- `Keyboard`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_keyboard_collect_data.yaml
```

- `SpaceMouse`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_spacemouse_collect_data.yaml
```

- `VR`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_vr_collect_data.yaml
```

- `GELLO`

Calibrate first with the `Single-Arm GELLO` teleoperation command, then record with the same `teleop.id`.

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_gello_collect_data.yaml
```

## Bimanual Data Collection

- `VR`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_vr_collect_data.yaml
```

- `GELLO`

Calibrate first with the `Bimanual GELLO` teleoperation command, then record with the same `teleop.id`.

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_gello_collect_data.yaml
```

`--manual_episode_control=true` keyboard controls:

- `C`: start recording current episode
- `S`: stop recording current episode
- `Backspace`: delete the previous episode
- `Esc`: exit the program

# ▶️ Dataset Replay (Optional)

```bash
bash scripts/replay_data.sh /path/to/dataset 0
```

`/path/to/dataset` is the local dataset root, and `0` is the episode index to replay.

Useful options:

- `playback-speed 0.25`: Replay at quarter speed
- `filter-static`: Remove exactly repeated action frames
- `no-gripper`: Do not connect or command PGI grippers
- `yes`: Skip the REPLAY confirmation

# 🚀 Training

Use the dataset pair that matches the data you collected:

```bash
# Single-arm dataset example
DATASET_REPO=yechen/ur5e_pgi_spnav
DATASET_ROOT=/home/yechen/UR5e-LeRobot/data/ur5e_pgi_spnav

# Bimanual dataset example
DATASET_REPO=FANYECHEN/ur5e_bimanual_vr
DATASET_ROOT=/home/yechen/UR5e-LeRobot/data/ur5e_bimanual_vr
```

- `Train ACT`

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.optimizer_lr=2e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --output_dir=./outputs/act \
  --job_name=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=60000 \
  --batch_size=32 \
  --save_freq=10000 \
  --wandb.enable=false
```

- `Train Diffusion Policy`

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=diffusion \
  --output_dir=./outputs/dp \
  --job_name=dp \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=32 \
  --save_freq=10000 \
  --wandb.enable=false
```

- `Train VQ-BeT`

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=vqbet \
  --output_dir=./outputs/vqbet \
  --job_name=vqbet \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=60000 \
  --batch_size=32 \
  --save_freq=10000 \
  --wandb.enable=false
```

- `Train MultiTaskDiT`

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=multi_task_dit \
  --policy.objective=diffusion \
  --output_dir=./outputs/multi_task_dit \
  --job_name=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=60000 \
  --batch_size=32 \
  --save_freq=10000 \
  --wandb.enable=false
```

- `Train PI0`
  Required base model: `pi_models/pi0-base` (download from [pi0_base](https://huggingface.co/lerobot/pi0_base/tree/main)).

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=pi0 \
  --policy.pretrained_path=/home/yechen/UR5e-LeRobot/pi_models/pi0-base \
  --output_dir=./outputs/pi0 \
  --job_name=pi0 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=16 \
  --save_freq=10000 \
  --peft.method_type=LORA \
  --peft.r=64 \
  --wandb.enable=false
```

- `Train PI05`
  Required base model: `pi_models/pi05-base` (download from [pi05_base](https://huggingface.co/lerobot/pi05_base/tree/main)).

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/yechen/UR5e-LeRobot/pi_models/pi05-base \
  --output_dir=./outputs/pi05 \
  --job_name=pi05 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=32 \
  --save_freq=10000 \
  --peft.method_type=LORA \
  --peft.r=64 \
  --wandb.enable=false
```

# 🤖 Evaluation

## Single-Arm Evaluation

Use this for single-arm checkpoints trained with `ur5e_pgi` observations and 7-dimensional actions.

- `Eval ACT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_act \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_act \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_act/checkpoints/last/pretrained_model
```

- `Eval Diffusion Policy`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_dp \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_dp \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_dp/checkpoints/last/pretrained_model
```

- `Eval VQ-BeT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_vqbet \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_vqbet \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_vqbet/checkpoints/last/pretrained_model
```

- `Eval MultiTaskDiT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_multi_task_dit \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_multi_task_dit \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_multi_task_dit/checkpoints/last/pretrained_model
```

- `Eval PI0`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_pi0 \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_pi0 \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_pi0/checkpoints/last/pretrained_model
```

- `Eval PI05`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/single_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_single_pi05 \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_single_pi05 \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/single_pi05/checkpoints/last/pretrained_model
```

## Bimanual Evaluation

Use this for bimanual checkpoints trained with `bi_ur5e_pgi` observations and 14-dimensional actions.

- `Eval ACT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_act \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_act \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_act/checkpoints/last/pretrained_model
```

- `Eval Diffusion Policy`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_dp \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_dp \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_dp/checkpoints/last/pretrained_model
```

- `Eval VQ-BeT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_vqbet \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_vqbet \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_vqbet/checkpoints/last/pretrained_model
```

- `Eval MultiTaskDiT`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_multi_task_dit \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_multi_task_dit \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_multi_task_dit/checkpoints/last/pretrained_model
```

- `Eval PI0`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_pi0 \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_pi0 \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_pi0/checkpoints/last/pretrained_model
```

- `Eval PI05`

```bash
lerobot-record \
  --config_path=/home/yechen/UR5e-LeRobot/configs/bimanual_eval.yaml \
  --dataset.repo_id=FANYECHEN/eval_bimanual_pi05 \
  --dataset.root=/home/yechen/UR5e-LeRobot/data/eval_bimanual_pi05 \
  --policy.path=/home/yechen/UR5e-LeRobot/outputs/bimanual_pi05/checkpoints/last/pretrained_model
```

# 📄 License

This project is released under the [MIT License](LICENSE).

# 🙏 Acknowledgements

This work builds upon excellent open-source projects including [LeRobot](https://github.com/huggingface/lerobot), [gello_software](https://github.com/wuphilipp/gello_software), [openpi](https://github.com/Physical-Intelligence/openpi), and [ACT](https://github.com/tonyzhaozh/act). We thank the authors and maintainers for their contributions.

<p align="center">
  <strong>🌟 If this project helps you, please give us a Star!</strong>
</p>
