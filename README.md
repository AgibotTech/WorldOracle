<h1 align="center">WorldOracle: Benchmarking Action-Conditioned World Models against Real-World Executions</h1>

<p align="center">
  <img alt="Paper — Coming Soon" src="https://img.shields.io/badge/Paper-Coming_Soon-6B7280?style=for-the-badge">
  <img alt="Dataset — Coming Soon" src="https://img.shields.io/badge/Dataset-Coming_Soon-6B7280?style=for-the-badge">
  <img alt="Leaderboard — Coming Soon" src="https://img.shields.io/badge/Leaderboard-Coming_Soon-6B7280?style=for-the-badge">
</p>

<p align="center">
  <img src="assets/framework.png" alt="WorldOracle evaluation framework" width="100%">
</p>

## Contents

- **[Overview](#overview)**
- **[Environment Setup](#environment-setup)**
   - [Requirements](#requirements)
   - [Evaluation checkpoints](#evaluation-checkpoints)
   - [Physical-state evaluator](#physical-state-evaluator)
- **[Data Preparation & Inference](#data-preparation--inference)**
- **[Evaluation](#evaluation)**
   - [Evaluation metrics](#evaluation-metrics)
   - [Human-operated distribution](#human-operated-distribution)
   - [Policy-rollout distribution](#policy-rollout-distribution)
   - [Targeted and resume-safe runs](#targeted-and-resume-safe-runs)
   - [Outputs and aggregation](#outputs-and-aggregation)
- **[LeaderBoard Submission](#leaderboard-submission)**
   - [Submission Registration](#submission-registration)
   - [Validation Sets](#validation-sets)
   - [Testset](#testset)
   - [Submission](#submission)

## Overview

Existing embodied world-model benchmarks typically evaluate text- or
task-conditioned videos, rely on simulation, or focus on short successful
clips. WorldOracle instead measures execution coherence by comparing an
action-conditioned prediction with its paired real-world execution under the
same initial observation and low-level action trajectory. Across AgiBot G2,
YAM, and human-hand embodiments, the benchmark includes successful and failed
interactions, long multi-segment episodes, and synchronized multi-view
observations.

WorldOracle evaluates two action distributions:

- **Human-operated distribution:** trajectories collected from human operation
  across the benchmark embodiments and tasks.
- **Policy-rollout distribution:** trajectories produced by a robot policy and
  then executed in the real world. World models are evaluated on completed
  predictions conditioned on these policy-induced actions.

Four complementary metrics measure physical state correctness, interaction
spatial alignment, key interaction area fidelity, and multi-view action
coherence. WorldOracle-SV averages the first three metrics, while WorldOracle-MV
averages all four for models that produce synchronized multi-view predictions.

Some package names, CLI messages, and output keys retain the historical
`acwmbench` identifier for compatibility. They refer to the WorldOracle
implementation released here.

## Environment Setup

### Requirements

- Linux
- Python 3.10
- A CUDA-capable GPU is strongly recommended for V-JEPA2, LPIPS, and DINOv3
- Enough local storage for the benchmark datasets, model predictions, and
  checkpoints

From the repository root, create the default environment:

```bash
git submodule update --init third_party/dinov3

bash setup_env.sh
conda activate acwmbench
```

The script creates an environment named `acwmbench` with Python 3.10 and
installs `requirements.txt`. To use a different environment name or Python
version:

```bash
ACWM_ENV=my_acwmbench ACWM_PYTHON=3.10 bash setup_env.sh
conda activate my_acwmbench
```

Verify the installation:

```bash
python evaluate.py --help
python -m pytest -q
```

### Evaluation checkpoints

The main entry point is `evaluate.py`. For large submissions,
`tools/evaluate_submission.py` provides bounded CPU/GPU workers, safe resume,
and optional concurrent physical-state evaluation while preserving the metric
definitions and aggregation rules.

V-JEPA2 and DINOv3 each need two different resources: a model checkpoint under
`ckpt/`, and the corresponding source repository under `third_party/` for
local `torch.hub` loading. Starting from the WorldOracle repository root, arrange
them as follows:

```text
WorldOracle/
├── ckpt/                              # checkpoint files only
│   ├── vjepa2/
│   │   └── vjepa2_1_vitb_dist_vitG_384.pt
│   └── dinov3/
│       └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── third_party/                       # cloned source repositories
    ├── vjepa2/
    │   ├── hubconf.py
    │   └── ...
    └── dinov3/
        ├── hubconf.py
        └── ...
```

The `third_party/vjepa2/` and `third_party/dinov3/` directories must be the
roots of the cloned repositories—the directories that contain `hubconf.py`—not
directories containing only the checkpoint files.

Point `configs/base.yaml` to these four locations. Relative paths below assume
that evaluation is launched from the repository root:

```yaml
dimensions:
  keyarea_visual_quality:
    vjepa:
      model_path: ./ckpt/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
      repo: ./third_party/vjepa2

  interaction_spatial_alignment:
    checkpoint: ./ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
    dinov3_repo: ./third_party/dinov3
```

`keyarea_visual_quality` loads the V-JEPA2 checkpoint and source repository.
`interaction_spatial_alignment` loads the DINOv3 pair, and
`multiview_consistency` reuses the DINOv3 settings recorded by the Interaction
run. No additional Multi-View checkpoint is required.

### Physical-state evaluator

Physical state correctness uses an OpenAI-compatible VLM endpoint. Export the
credential without committing it:

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

Copy `configs/tasks/openai_compatible_example.yaml`, replace the endpoint
placeholder with the URL supplied by your provider, and retain the released
model and decoding settings for benchmark comparability. Pass this file as
`--po-config-path` to the submission executor. Never commit credentials.

Example:

```bash
python tools/evaluate_submission.py \
  --config-path configs/tasks/my_model.yaml \
  --po-config-path configs/tasks/openai_compatible_example.yaml \
  --predictions /path/to/model-predictions \
  --gt /path/to/main-ground-truth \
  --output /path/to/evaluation-output \
  --with-po --api-workers 8 --confirm-paid-api
```

## Data Preparation & Inference

WorldOracle uses two data packages:

- the inference package: initial observations, action/state trajectories,
  calibration, and robot kinematics supplied to the world model;
- the ground-truth package: withheld future observations, prompts, masks, and physical
  outcome criteria used by the evaluator.

Extract both packages outside the repository, then point the configuration at
their local locations. The evaluator does not require a fixed directory name.

```bash
mkdir -p /path/to/inference-data /path/to/main-ground-truth
# Extract the separately distributed archives into these directories.
```

The inference package follows this layout:

```text
data/inference_dataset/
├── manifest.json
├── schema.json
├── checksums.sha256
├── shared/<embodiment>/
└── <task_id>/
    ├── task.json
    ├── schema.json
    └── <episode_id>/
        ├── metadata.json
        ├── calibration.json
        ├── trajectory.parquet
        ├── checksums.sha256
        └── observations/
            ├── top_head.png
            ├── hand_left.png
            └── hand_right.png
```

Use `manifest.json` to discover tasks and episodes. Every inference episode
contains only the initial RGB observations. Generate all future frames using
the provided trajectory while preserving the task and episode identifiers.

Write model predictions as:

```text
<prediction_root>/
└── <task_id>/
    └── <episode_id>/
        └── video/
            ├── top_head/frame_000000.png
            ├── top_head/frame_000001.png
            ├── hand_left/frame_000000.png
            └── hand_right/frame_000000.png
```

Requirements:

- `top_head` is required for the standard metrics.
- `hand_left` and `hand_right` are required only for Multi-View evaluation.
- Frame names must sort in temporal order. Zero-padded names are recommended.
- PNG, JPEG, BMP, and WebP frames are supported.
- Every submitted view must contain exactly the expected number of frames.
- Task and episode directory names must match the inference manifest and GT
  dataset exactly.

Configure the evaluation roots in `configs/base.yaml`:

```yaml
save_path: ./output
data:
  gt_path: /path/to/main-ground-truth
  val_base: /path/to/model-predictions
```

For multiple models, keep `configs/base.yaml` shared and create a small override
for each prediction root:

```yaml
# configs/tasks/my_model.yaml
data:
  val_base: /path/to/my_model_predictions
```

Check every episode before evaluation:

```bash
python evaluate.py \
  --check_data \
  --config_path configs/tasks/my_model.yaml
```

## Evaluation

General command:

```bash
python evaluate.py \
  --dimension <dimension> [<dimension> ...] \
  --config_path configs/tasks/my_model.yaml
```

### Evaluation metrics

The four paper metrics correspond to the following internal dimension names:

- `physical_judge`: Physical State Correctness
- `interaction_spatial_alignment`: Interaction Spatial Alignment
- `keyarea_visual_quality`: Key Interaction Area Fidelity
- `multiview_consistency`: Multi-View Action Coherence

All scores are reported on a 0--100 scale, and higher is better.
`multiview_consistency` requires synchronized `top_head`, `hand_left`, and
`hand_right` predictions. It reuses the saved `top_head` robot-arm alignment
from `interaction_spatial_alignment`, so keep Interaction Spatial Alignment
before Multi-View Action Coherence in the dimension list. For single-view
models, omit `multiview_consistency`.

### Human-operated distribution

Validate and evaluate AgiBot G2 human-operated predictions with:

```bash
python evaluate.py \
  --check_data \
  --distribution human_operated \
  --val-base /path/to/human_operated_predictions \
  --config_path configs/tasks/my_model.yaml

python evaluate.py \
  --dimension physical_judge keyarea_visual_quality \
    interaction_spatial_alignment multiview_consistency \
  --distribution human_operated \
  --val-base /path/to/human_operated_predictions \
  --output-dir /path/to/evaluation_output \
  --config_path configs/tasks/my_model.yaml
```

For YAM, use `--distribution yam` with
`configs/distributions/yam.yaml`. Human-hand evaluation follows the same
prediction layout and metric interface with its corresponding GT profile.

### Policy-rollout distribution

Policy-rollout distribution uses the same four metrics as the human-operated
distribution. Select its GT profile independently from the model predictions:

```bash
python evaluate.py \
  --check_data \
  --distribution policy_rollout \
  --val-base /path/to/policy_predictions \
  --config_path configs/distributions/policy_rollout.yaml

python evaluate.py \
  --dimension physical_judge keyarea_visual_quality \
    interaction_spatial_alignment multiview_consistency \
  --distribution policy_rollout \
  --val-base /path/to/policy_predictions \
  --output-dir /path/to/evaluation_output \
  --config_path configs/distributions/policy_rollout.yaml
```

The normal command inherits the accelerated pipeline settings from
`configs/base.yaml`. For the restart-safe multi-process executor, which keeps
metric definitions unchanged and writes results outside the prediction tree:

```bash
python tools/evaluate_submission.py \
  --config-path configs/distributions/policy_rollout.yaml \
  --po-config-path configs/tasks/openai_compatible_example.yaml \
  --predictions /path/to/policy_predictions \
  --gt /path/to/policy-ground-truth \
  --output /path/to/evaluation_output \
  --views three \
  --workers 6 \
  --threads 4 \
  --cpu-heavy \
  --with-po \
  --api-workers 8 \
  --confirm-paid-api
```

`--with-po` requires completed keyframe and `success_criteria` annotations.
Until those annotations are ready, omit `--with-po` to run the other three
metrics.

### Targeted and resume-safe runs

```bash
# First five episodes in every task
python evaluate.py \
  --dimension keyarea_visual_quality \
  --config_path configs/tasks/my_model.yaml \
  --top-n-per-task 5

# One task and episode
python evaluate.py \
  --dimension interaction_spatial_alignment \
  --config_path configs/tasks/my_model.yaml \
  --task hangtowel \
  --episode ep000000
```

Existing compatible per-episode JSON files are reused automatically. Pass
`--overwrite` for a true recomputation. Use `--skip-check` only when the input
has already been validated and skipping the automatic completeness check is
intentional.

### Outputs and aggregation

Per-episode results are written beside the predictions:

```text
<prediction_root>/<task_id>/<episode_id>/
├── physical_judge.json
├── keyarea_visual_quality.json
├── interaction_spatial_alignment.json
└── multiview_consistency.json
```

Aggregate one or more evaluated baselines with:

```bash
python tools/summarize_acwmbench_scores.py \
  --baseline MyModel=/path/to/my_model_predictions \
  --output-dir output/worldoracle_score_summary
```

The command writes `acwmbench_scores.csv` and `acwmbench_scores.json`. Metric
results are task-balanced: episode scores are averaged within each task first,
then task scores receive equal weight. For compatibility, the output key
`acwmbench_core_score` corresponds to WorldOracle-SV and
`acwmbench_full_score` corresponds to WorldOracle-MV.

## LeaderBoard Submission

### Submission Registration

Leaderboard submission requires an approved participant account bound to one
model. Submit the access-request form with your name, institution or team,
contact email, model description, intended track, and a project or paper link
when available. After email verification and organizer approval, sign in with
the verified registration email and the password created during registration.

See the
[account-registration guide](https://worldoracle-benchmark.github.io/submission/#access)
for the complete approval and sign-in workflow.

### Validation Sets

Download the validation release from the
[WorldOracle validation dataset on Hugging Face](https://huggingface.co/datasets/WorldOracle/worldoracle-validation).

The current AgiBot G2 validation split contains:

- **Human-operated distribution:** 20 episodes, balanced as five episodes from
  each of `hangtowel`, `pickfruit`, `pourwater`, and `pulldrawer`. The disclosed
  GT portion is approximately **17.0 GiB**.
- **Policy-rollout distribution:** 12 episodes, balanced as four episodes from
  each of `hangtowel`, `pickfruit`, and `pourwater`. The disclosed GT portion is
  approximately **1.6 GiB**.

Together, the disclosed GT is approximately **18.6 GiB**, plus the smaller
inference packages and archive overhead. Each distribution follows this logical
layout:

```text
<distribution>/
├── manifest.json
├── inference/<task>/<episode>/
│   ├── observations/             # Initial top/head and wrist observations
│   ├── trajectory.parquet        # Recorded action/state trajectory
│   ├── calibration.json
│   ├── metadata.json
│   └── checksums.sha256
└── ground_truth/<task>/<episode>/
    ├── prompt/                    # Prompts, criteria, and evaluation masks
    ├── video/                     # Paired real-world execution
    └── multiview_masks/           # Multi-view evaluation annotations
```

Run your model on `inference/`, generate predictions with the same
`task / episode_id / view` identities, and evaluate them against the disclosed
validation GT. Use these results to verify temporal alignment, view
synchronization, output quality, and packaging before submitting predictions
for the withheld leaderboard test set.

### Testset

Download the test-set inference release from the
[WorldOracle test set on Hugging Face](https://huggingface.co/datasets/WorldOracle/worldoracle-testset).
The release contains the inputs required to generate leaderboard submissions;
test-set ground truth remains private and is evaluated by the WorldOracle
evaluation service.

### Submission

Create a draft in the submission portal, select one track and platform, and
upload one ZIP using the `preview-2` manifest schema. The ZIP root must contain
`manifest.json`, `README.md`, and `predictions/` directly—do not add a wrapper
directory.

```text
submission.zip
├── manifest.json
├── README.md
└── predictions/
    └── <task>/<episode>/
        ├── top_head.mp4
        ├── hand_left.mp4          # Multi-view submissions only
        └── hand_right.mp4         # Multi-view submissions only
```

Use the official `task`, `episode_id`, and `view` values without positional
reordering. Videos must use contiguous frames from index zero; synchronized
views must have equal frame counts, frame rates, and durations. The standard
format is H.264 MP4, `yuv420p`, no audio, source resolution preserved, and
30 FPS. Single-view submissions are eligible for WorldOracle-SV;
WorldOracle-MV requires all three synchronized views.

Package limits are **3 GiB** for Main Track, **3 GiB** for Policy Track, and
**12 GiB** for Long-Horizon Track. The limit applies to both the uploaded ZIP
and its declared expanded contents. Never include GT data, credentials,
checkpoints, caches, or unrelated experiment outputs.

The portal uploads verified 16 MiB chunks over up to six resumable lanes,
validates the package, and then places a confirmed submission in the automatic
evaluation queue. See the
[package and submission guide](https://worldoracle-benchmark.github.io/submission/#prepare)
for the manifest example, required FFmpeg profile, ZIP command, local checks,
and the complete four-step submission workflow.
