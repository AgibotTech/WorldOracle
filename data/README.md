# ACWMBench datasets

ACWMBench uses two separately distributed data packages:

- the ground-truth package, used only by the evaluator;
- the inference package, containing initial observations, action/state
  trajectories, calibration, and robot kinematics provided to a model.

## Install datasets

Extract the separately distributed archives into locations of your choice. The
repository does not require or publish their original directory names.

```bash
mkdir -p /path/to/main-ground-truth /path/to/inference-data
# Extract each archive into the corresponding directory.
```

The extracted layout should be:

```text
/path/to/
├── main-ground-truth/   # evaluation-only data
└── inference-data/      # model inputs
```
