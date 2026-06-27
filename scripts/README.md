# Dataset replay

Always inspect an episode without connecting to the robots first:

```bash
bash scripts/replay_data.sh 0 --dry-run
```

The wrapper defaults to `data/ur5e-bimanual-vr-joint` and contains the left/right UR5e and PGI
network settings used by `configs/bimanual_vr_joint.yaml`.

Real replay automatically infers joint/EEF mode, single/bimanual layout, and gripper fields from the
dataset action names. It runs at half of the recorded speed by default and asks for an explicit `REPLAY`
confirmation before moving:

```bash
bash scripts/replay_data.sh 0
```

To replay another local dataset, the original explicit form is still supported:

```bash
bash scripts/replay_data.sh /path/to/lerobot_dataset 0 --dry-run
```

Useful options:

```text
--playback-speed 0.25  Replay at quarter speed
--filter-static        Remove exactly repeated action frames
--no-gripper           Do not connect or command PGI grippers
--yes                  Skip the REPLAY confirmation
```

Default hardware mapping:

```text
single/right UR:       192.168.1.5
single/right gripper:  192.168.1.7:8887
left UR:               192.168.1.3
left gripper:          192.168.1.8:8888
right UR:              192.168.1.5
right gripper:         192.168.1.7:8887
```

The robot first follows an interpolated trajectory to the episode's first action. Press `Ctrl-C` to stop
replay and disconnect the robot.
