## P2P - Prompt To Policy

PPO-based robot learning for a Franka FR3 manipulator in MuJoCo. Two environments are available: an **exploration** task (learn to move the arm around the workspace) and a **reach** task (move the end-effector to a randomly placed green ball). Both environments share identical observation and action spaces to enable transfer learning.

### Project structure

```
src/
  env.py            FrankaReachEnv — reach the green ball
  explore_env.py    FrankaExploreEnv — explore the workspace
  reward.py         Reward function for the reach task
  ppo_config.py     PPO hyperparameters
  train.py          Training entry point
  make_mp4.py       Render a saved policy to MP4
  viewer.py         MuJoCo viewer helper

envs/
  reach_scene.xml   Scene with robot, target sphere, floor, cameras
  explore_scene.xml Scene with robot, floor, cameras
```

### Environments

Both environments share the same observation and action spaces so that a model trained on one can be fine-tuned on the other.

#### Observations (Dict / `MultiInputPolicy`)

| Key | Shape | Content |
|---|---|---|
| `state` | (25,) float32 | joint positions (7) + EE position (3) + EE rotation matrix (9) + EE twist (6) |
| `image` | (64, 64, 6) uint8 | front camera RGB (channels 0-2) + top-down camera RGB (channels 3-5) |

#### Actions (6-dim continuous [-1, 1])

Joystick-style end-effector control: `[dx, dy, dz, droll, dpitch, dyaw]`. Converted to joint deltas via Jacobian pseudoinverse.

- Translation scale: 5 cm/step
- Rotation scale: 0.1 rad/step

#### FrankaReachEnv

- **Goal**: move the EE to a green sphere randomized each episode within the reachable workspace
- **Reward**: `-distance(ee, target)` + 1.0 bonus on success
- **Success**: EE within 6 cm of ball center (ball radius = 6 cm)
- **Episode length**: 200 steps

#### FrankaExploreEnv

- **Goal**: visit as many 15 cm voxel cells in the workspace as possible
- **Reward**: +1.0 per newly visited cell, minus a small energy penalty on torques
- **Episode length**: 300 steps

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Training

```bash
cd src

# Train explore from scratch
python train.py --env explore --runs-dir runs --render

# Train reach from scratch
python train.py --env reach --runs-dir runs --render

# Fine-tune reach from a pretrained explore model (transfer learning)
python train.py --env reach --pretrained-model runs/explore_YYYYMMDD_HHMMSS/best/best_model.zip --runs-dir runs --render
```

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--env` | `reach` | Environment: `reach` or `explore` |
| `--total-timesteps` | 1000000 | Total training steps |
| `--n-envs` | 4 | Parallel environments |
| `--eval-freq` | 25000 | Steps between evaluations |
| `--runs-dir` | `./runs` | Base directory for auto timestamped runs |
| `--plot-every` | 5 | Save/update plots every N rollout iterations |
| `--seed` | 42 | Random seed |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--vec-env` | `subproc` | `subproc` (parallel) or `dummy` |
| `--render` | off | Open MuJoCo viewer during training |
| `--pretrained-model` | None | Path to pretrained model zip to fine-tune |

### Monitoring

Each training run auto-creates:

- `runs/<env>_YYYYMMDD_HHMMSS/config.json` with key run/training settings and PPO params
- `runs/<env>_YYYYMMDD_HHMMSS/stats.csv` with per-rollout iteration stats
- `runs/<env>_YYYYMMDD_HHMMSS/training_plots.png` with:
  - subplot 1: average reward (train + eval)
  - subplot 2: success rate (train + eval)
- `runs/<env>_YYYYMMDD_HHMMSS/best/best_model.zip` best model (by eval reward)
- `runs/<env>_YYYYMMDD_HHMMSS/final_model.zip` final model

### Rendering rollouts

```bash
cd src

# Render reach policy
python make_mp4.py --env reach --model-path runs/reach_YYYYMMDD_HHMMSS/best/best_model.zip --output runs/reach_YYYYMMDD_HHMMSS/rollout.mp4 --episodes 5

# Render explore policy
python make_mp4.py --env explore --model-path runs/explore_YYYYMMDD_HHMMSS/best/best_model.zip --output runs/explore_YYYYMMDD_HHMMSS/rollout.mp4 --episodes 3
```

The output video has three panes side by side: **human view** | **front camera (model input)** | **top-down camera (model input)**.

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--env` | `reach` | Environment: `reach` or `explore` |
| `--model-path` | (explore default) | Path to saved model zip |
| `--output` | (explore default) | Output MP4 path |
| `--episodes` | 1 | Number of episodes to render |
| `--max-steps` | 200 | Per-episode step cap |
| `--fps` | 50 | Output video FPS |
| `--width` | 640 | Width of each pane in pixels |
| `--height` | 640 | Height of each pane in pixels |
| `--device` | `auto` | Device for model inference |
| `--stochastic` | off | Use stochastic policy (default: deterministic) |
