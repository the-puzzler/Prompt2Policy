## P2P - Prompt To Policy

PPO-based reach task for a Franka FR3 manipulator in MuJoCo. The robot learns to move its end-effector to a randomly placed green target sphere, using joint state and a camera image as observations.

### Project structure

```
src/
  env.py          Gymnasium environment (FrankaReachEnv)
  reward.py       Reward function definition
  ppo_config.py   PPO hyperparameters
  train.py        Training entry point

mujoco_menagerie/franka_fr3/
  fr3.xml         Franka FR3 model (from MuJoCo Menagerie)
  reach_scene.xml Scene with robot, target sphere, floor, and camera
```

### Environment details

**Observations** (Dict / `MultiInputPolicy`):
- `state` (20-dim float): joint positions (7) + joint velocities (7) + end-effector position (3) + target position (3)
- `image` (64x64x3 uint8): RGB from a fixed front camera

**Actions** (7-dim continuous [-1, 1]): delta joint position targets, scaled by 0.05 rad per step.

**Reward**: `-distance(ee, target)` with a +1.0 bonus when within 5 cm. Episode terminates on success or after 200 steps.

The target is randomized each episode within the workspace volume `[0.2, -0.3, 0.2]` to `[0.6, 0.3, 0.6]`.

SB3's `MultiInputPolicy` handles feature extraction automatically via its built-in `CombinedExtractor` (NatureCNN for the image, flatten for the state vector).

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

The repo expects `mujoco_menagerie/` at the project root (already included as a subdirectory).

### Training

```bash
cd src
python train.py
```

CLI options:

| Flag | Default | Description |
|---|---|---|
| `--total-timesteps` | 1000000 | Total training steps |
| `--n-envs` | 4 | Number of vectorized environments (DummyVecEnv) |
| `--eval-freq` | 10000 | Steps between evaluations |
| `--save-path` | `./runs/fr3_reach` | Output directory |
| `--seed` | 42 | Random seed |

### Monitoring

```bash
tensorboard --logdir src/runs/fr3_reach/tb
```

Best model (by eval reward) is saved to `<save-path>/best/`, final model to `<save-path>/final_model.zip`.
