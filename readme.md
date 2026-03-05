## Prompt2Policy (P2P)

Prompt2Policy is a modular RL training framework for MuJoCo robots that combines:
- Curriculum learning (progressively harder tasks).
- LLM-generated task/environment proposals (provider-agnostic interface, mock in v1).
- Swappable simulation backends (`native` MuJoCo or `mcp` via `mujoco_mcp`).
- Swappable RL algorithm interface (PPO implemented in v1).
- Declarative reward DSL that can be overridden per task.

The current default setup is FR3 reaching with RGB + proprio observations and PPO (`MultiInputPolicy`).

## Implementation Status

### Implemented
- Modular package layout under `src/prompt2policy` with clear domain boundaries.
- Typed config/spec system with Pydantic (`ExperimentConfig`, `RobotSpec`, `TaskSpec`, `WorldSpec`, `RewardSpec`).
- Hybrid environment architecture with backend selection (`native` MuJoCo or `mcp`).
- FR3 default robot setup and seed reaching task.
- Dict observation pipeline (`image` RGB 84x84 + `proprio` joint pos/vel).
- Declarative reward DSL and runtime reward evaluator.
- RL adapter abstraction with PPO implementation (SB3 `MultiInputPolicy`).
- Curriculum orchestration with stage loop, artifact persistence, and rolling mean-reward promotion gate.
- Provider-agnostic curriculum generation interface with a deterministic mock provider.
- Fallback task generation when LLM output is invalid.
- CLI commands for `train`, `generate-stage`, `evaluate`.
- Structured runtime logging (`--log-level`, `-v`) for stage-by-stage visibility.
- Unit/integration/smoke tests for core behavior.

### Missing / Not Yet Implemented
- Real LLM provider integrations (OpenAI/Anthropic/etc.); only mock provider exists in v1.
- Additional RL algorithms behind the registry (A2C/SAC/TD3); PPO is the only implemented adapter.
- Rich camera/vision stack features (multi-camera fusion, augmentations, temporal stacks).
- Advanced reward terms beyond current DSL primitives (e.g., orientation, force/torque shaping, learned rewards).
- Full MCP-driven world editing/reward modeling loop with robust schema-roundtrips to external LLMs.
- Production-grade experiment tracking (TensorBoard/W&B integration, dashboarding, alerting).
- Distributed or multi-process training support.
- Full dependency bootstrap for all optional runtime paths by default (`stable-baselines3`, `gymnasium`, viewer-side MCP process) is still environment-dependent.

## Quick Start

### 1) Setup with `uv + .venv` (recommended)

```bash
./scripts/bootstrap_uv_venv.sh
source .venv/bin/activate
```

### 2) Run training

```bash
uv run prompt2policy train --experiment configs/experiment.yaml
```

You can increase log detail with:

```bash
uv run prompt2policy train --experiment configs/experiment.yaml -v
```

## High-Level Flow

Each training run follows this lifecycle:

1. Load experiment config, robot spec, and seed task YAML.
2. Compose a stage scene from base scene + task world additions (objects/cameras).
3. Build reward model from declarative reward terms.
4. Create train/eval envs with selected backend (`native` or `mcp`).
5. Train PPO, evaluate, and store metrics/checkpoint/artifacts.
6. Apply promotion rule (rolling mean reward threshold).
7. If promoted and stages remain, ask curriculum generator for next task.
8. Validate generated task; fallback to conservative increment if invalid.
9. Repeat until promotion fails or `max_stages` reached.

## Repository Structure

```text
Prompt2Policy/
├── configs/                     # Experiment/robot/training/curriculum YAML
├── tasks/seed/                  # Seed task specs
├── scripts/                     # Utility scripts (uv bootstrap)
├── src/prompt2policy/
│   ├── algorithms/              # RL adapter interfaces + PPO implementation
│   ├── cli/                     # CLI entrypoint and logging controls
│   ├── config/                  # Typed config models + YAML I/O
│   ├── curriculum/              # Task specs, promotion, stage generation
│   ├── envs/                    # Gym env and backend integrations
│   ├── llm/                     # Provider interface, parser, mock provider
│   ├── observations/            # Image + proprio assembly
│   ├── rewards/                 # Reward DSL and runtime reward computation
│   ├── robots/                  # Robot specs and registry defaults (FR3)
│   ├── training/                # Curriculum runner and evaluation utilities
│   └── world/                   # Scene specs, scene composition, MCP adapter
└── tests/                       # unit / integration / smoke tests
```

## Core Modules and Responsibilities

### Config and Specs
- `config/models.py`: top-level runtime/training/curriculum typed config models.
- `config/loader.py`: YAML load/save and validation into Pydantic models.
- `robots/specs.py`: `RobotSpec` (robot id, joint list, scene path, camera defaults).
- `world/specs.py`: `WorldSpec`, `ObjectSpec`, `CameraSpec`.
- `curriculum/specs.py`: `TaskSpec` for each curriculum stage.
- `rewards/specs.py`: reward term schema (`distance`, `contact`, `time_penalty`, `action_l2`, `goal_bonus`).

### Simulation and Environment
- `world/builder.py`: composes runtime scene XML per stage.
- `envs/native_backend.py`: in-process MuJoCo stepping and observations.
- `envs/mcp_backend.py`: MuJoCo MCP adapter backend for remote/viewer-driven runs.
- `envs/hybrid_env.py`: Gymnasium env combining backend, reward model, and observations.
- `envs/factory.py`: backend selection and env creation.
- `observations/assembler.py`: standard Dict observation assembly:
  - `image`: RGB `84x84x3` uint8
  - `proprio`: joint pos/vel float32

### Rewards
- `rewards/model.py`: compiles declarative reward spec into runtime reward model with:
  - `compute(obs, action, next_obs, info) -> float`
  - `is_success(obs, info) -> bool`

### Algorithms
- `algorithms/interface.py`: algorithm adapter protocol + common config.
- `algorithms/ppo_adapter.py`: SB3 PPO implementation (`MultiInputPolicy`).
- `algorithms/registry.py`: algorithm selection by name (currently `ppo`).

### Curriculum + LLM
- `curriculum/promotion.py`: rolling mean-reward promotion gate.
- `curriculum/generator.py`: LLM-based next-stage generation + fallback on invalid output.
- `llm/interfaces.py`: provider protocol.
- `llm/mock_provider.py`: deterministic mock generator for v1.
- `llm/parser.py`: strict TaskSpec parsing/validation.

### Orchestration and CLI
- `training/curriculum_runner.py`: full run orchestrator (train/eval/promotion/artifacts).
- `training/evaluator.py`: checkpoint evaluation helper.
- `training/stage_generation.py`: generate next-stage task utility.
- `cli/__main__.py`: CLI commands + log configuration.

## CLI Reference

### Train
```bash
prompt2policy train --experiment configs/experiment.yaml
```

Optional logging flags:
```bash
prompt2policy train --experiment configs/experiment.yaml --log-level INFO
prompt2policy train --experiment configs/experiment.yaml -v
```

### Generate Stage
```bash
prompt2policy generate-stage \
  --previous-task tasks/seed/fr3_reaching.yaml \
  --output tasks/generated/next.yaml \
  --mean-reward -50.0
```

### Evaluate Checkpoint
```bash
prompt2policy evaluate \
  --checkpoint runs/fr3_seed/stage_000/policy_checkpoint \
  --robot-config configs/robot_fr3.yaml \
  --task tasks/seed/fr3_reaching.yaml \
  --backend native \
  --episodes 5
```

## Config Files

Default files:
- `configs/experiment.yaml`: full run config (backend, training hyperparameters, curriculum limits).
- `configs/robot_fr3.yaml`: FR3 robot spec and default third-person camera.
- `tasks/seed/fr3_reaching.yaml`: seed stage task/world/reward definition.

To switch backends, change:

```yaml
runtime:
  backend: native   # or mcp
```

To override reward behavior, edit `reward_spec` in task YAML.

## Artifacts and Outputs

Training run output directory (example `runs/fr3_seed`) contains:
- `stage_000/scene.xml`: composed scene used for that stage.
- `stage_000/task.yaml` + `task.json`: exact task spec used.
- `stage_000/metrics.json`: train/eval metrics for the stage.
- `stage_000/policy_checkpoint.zip`: saved SB3 checkpoint.
- `generated_tasks/*.yaml`: generated next-stage task specs (if promoted).
- `run_summary.json`: final run summary across stages.

## Testing

Run all tests:

```bash
pytest -q
```

Test layout:
- `tests/unit`: schema/reward/promotion/registry correctness.
- `tests/integration`: backend/env/algorithm/curriculum wiring.
- `tests/smoke`: end-to-end command and fallback behavior.

## Notes

- Native backend is the default for performance and determinism.
- MCP backend is useful for debugging and world overrides through `mujoco_mcp`.
- The v1 LLM provider is mock-only; interfaces are ready for real provider adapters.
- Submodules (`mujoco_mcp`, `mujoco_menagerie`) are treated as vendored dependencies and are not modified by the first-party runtime.
