<div align="center">
  <h1 align="center"> UR5e-Lerobot </h1>
</div>

<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

# 📖 Introduction

**UR5e-Lerobot (UR5e + PGI)** is a practical real-world robotics extension focused on lowering the barrier to data-driven robot learning with **LeRobot**.

We focus on an end-to-end workflow for real hardware, from teleoperation to deployment:

- 🦾 **Robot Setup Support**: Single-arm and bimanual UR5e setups with PGI gripper support.
- 🕹️ **Teleoperation Pipelines**: SpaceMouse (`spnav`), Quest 3 VR (`quest`), Gello(`gello`), and Keyboard (`keyboard_ur5e`) for UR5e.
- 📦 **Data-to-Policy Workflow**: Unified examples for data collection, policy training (`act`, `pi0`, `pi0.5`), and real-world evaluation.
- 💻 **LeRobot-Native Experience**: Keep the upstream CLI workflow so teams can move quickly without heavy customization.

| Category | Supported Options |
|---|---|
| Hardware | `UR5e`, `PGI gripper` |
| Robot types | `ur5e_pgi`, `bi_ur5e_pgi` |
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
> We have validated and tuned hyperparameters on this hardware stack for three policies: `ACT`, `pi0`, and `pi0.5`.
> This does **not** mean other policies (for example `diffusion` and `smolvla`) are unsupported.
> We will continue to publish tested training commands and recommended hyperparameters for additional policies in future updates.

## 📢 Update

- **2026/04/13**, Added keyboard teleoperation support and a keyboard-based data collection branch.
- **2026/04/12**, Officially released the **UR5e-Lerobot** project.

# 🛠️ Installation

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge

cd UR5e-Lerobot
git submodule update --init --recursive
pip install -e .

pip install -e ".[pi,peft]"
pip install ur-rtde==1.6.3 pymodbus==2.1.0 spnav==0.9
pip install -e third_party/oculus_reader
```

## Verify The Installation

```bash
lerobot-info
python -c "import rtde_control, pymodbus, spnav; from oculus_reader.reader import OculusReader; print('UR5e deps OK')"
```

# 🕹️ Teleoperation

This section provides reference commands for common teleoperation modes.

## Single-Arm SpaceMouse

```bash
lerobot-teleoperate \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --teleop.type=spnav \
  --teleop.robot_ip=192.168.1.5 \
  --fps=60
```

## Single-Arm Keyboard

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

## Single-Arm Quest / VR

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

## Bimanual Quest / VR

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

## Single-Arm Keyboard Recording

```bash
lerobot-record \
  --robot.type=ur5e_pgi \
  --robot.action_mode=eef \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30} }" \
  --teleop.type=keyboard_ur5e \
  --dataset.repo_id=yechen/ur5e_pgi_keyboard \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/ur5e_pgi_keyboard \
  --dataset.single_task="pick up the target object and place it in the target area" \
  --dataset.num_episodes=30 \
  --dataset.fps=20 \
  --dataset.push_to_hub=false \
  --manual_episode_control=true \
  --display_data=true
```

## Single-Arm VR Recording

```bash
lerobot-record \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30} }" \
  --teleop.type=quest \
  --teleop.robot_ip=192.168.1.5 \
  --dataset.repo_id=yechen/ur5e_pgi_vr \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/ur5e_pgi_vr \
  --dataset.single_task="pick up the target object and place it in the target area" \
  --dataset.num_episodes=30 \
  --dataset.fps=20 \
  --dataset.push_to_hub=false \
  --manual_episode_control=true \
  --display_data=true
```

`--manual_episode_control=true` keyboard controls:

- `C`: start recording current episode
- `S`: stop recording current episode
- `Backspace`: delete the previous episode
- `Esc`: exit the program

# 🚀 Training

For training, please refer to the official LeRobot training guides for more details and advanced parameters.

- `Train ACT Policy`

```bash
lerobot-train \
  --dataset.repo_id=yechen/ur5e_pgi_spnav_demo50 \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/ur5e_pgi_spnav_demo50 \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.optimizer_lr=2e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --output_dir=./outputs/act_ur5e_dualcam_lerobot2style \
  --job_name=act_ur5e_dualcam_lerobot2style \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=32 \
  --save_freq=10000
```

- `Train Pi0 Policy`
  Required base model: `pi_models/pi0-base` (download from [pi0_base](https://huggingface.co/lerobot/pi05_base/tree/main)).

```bash
lerobot-train \
  --dataset.repo_id=yechen/ur5e_pgi_spnav_demo50 \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/ur5e_pgi_spnav_demo50 \
  --policy.type=pi0 \
  --policy.pretrained_path=/home/yechen/UR5e-Lerobot/pi_models/pi0-base \
  --output_dir=./outputs/pi0_ur5e_lora_abs_v2 \
  --job_name=pi0_ur5e_lora_abs_v2 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=32 \
  --save_freq=10000 \
  --peft.method_type=LORA \
  --peft.r=64
```

- `Train Pi05 Policy`
  Required base model: `pi_models/pi0.5-base` (download from [pi05_base](https://huggingface.co/lerobot/pi05_base/tree/main)).

```bash
lerobot-train \
  --dataset.repo_id=yechen/ur5e_pgi_spnav_demo50 \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/ur5e_pgi_spnav_demo50 \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/yechen/UR5e-Lerobot/pi_models/pi0.5-base \
  --output_dir=./outputs/pi05_lora \
  --job_name=pi05_lora \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=32 \
  --save_freq=10000 \
  --peft.method_type=LORA \
  --peft.r=64
```

# 🤖 Evaluation

To evaluate trained checkpoints on the real robot, you can use `lerobot-record` with `--policy.path` for inference.

- `Run ACT`

```bash
lerobot-record \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=yechen/eval_act_ur5e \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/eval_act_ur5e \
  --dataset.single_task="pick up the target object and place it in the target area" \
  --dataset.num_episodes=10 \
  --dataset.fps=20 \
  --dataset.push_to_hub=false \
  --manual_episode_control=true \
  --display_data=true \
  --policy.path=/home/yechen/UR5e-Lerobot/outputs/act_ur5e/checkpoints/last/pretrained_model
```

- `Run Pi0`

```bash
lerobot-record \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=yechen/eval_pi0_ur5e \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/eval_pi0_ur5e \
  --dataset.single_task="pick up the target object and place it in the target area" \
  --dataset.num_episodes=10 \
  --dataset.fps=20 \
  --dataset.push_to_hub=false \
  --manual_episode_control=true \
  --display_data=true \
  --policy.path=/home/yechen/UR5e-Lerobot/outputs/pi0_ur5e/checkpoints/last/pretrained_model
```

- `Run Pi05`

```bash
lerobot-record \
  --robot.type=ur5e_pgi \
  --robot.robot_ip=192.168.1.5 \
  --robot.gripper_ip=192.168.1.7 \
  --robot.gripper_port=8887 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=yechen/eval_pi05_ur5e \
  --dataset.root=/home/yechen/UR5e-Lerobot/data/eval_pi05_ur5e \
  --dataset.single_task="pick up the target object and place it in the target area" \
  --dataset.num_episodes=10 \
  --dataset.fps=20 \
  --dataset.push_to_hub=false \
  --manual_episode_control=true \
  --display_data=true \
  --policy.path=/home/yechen/UR5e-Lerobot/outputs/pi05_ur5e/checkpoints/last/pretrained_model
```

# 📄 License

This project is released under the [MIT License](LICENSE).

# 🙏 Acknowledgements

This work builds upon excellent open-source projects including [LeRobot](https://github.com/huggingface/lerobot), [gello_software](https://github.com/wuphilipp/gello_software), [openpi](https://github.com/Physical-Intelligence/openpi), and [ACT](https://github.com/tonyzhaozh/act). We thank the authors and maintainers for their contributions.
