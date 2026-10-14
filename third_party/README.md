# Source dependencies (no model weights)

- `dinov3` is the existing upstream submodule pinned to
  `6876159a11b4df116f30f667f8c9888617df0751`. Initialize it with
  `git submodule update --init third_party/dinov3` (upstream network access required).
- `vjepa2` is the local source snapshot used in the speedup experiments, from
  https://github.com/facebookresearch/vjepa2. Its original `LICENSE` and
  `APACHE-LICENSE` are preserved. This local snapshot did not retain Git metadata;
  no unverified upstream commit is claimed. The enclosing release commit pins
  the exact source content used here.
  The upstream notebook example trajectory (`*.npz`) is omitted; it is not
  required for benchmark inference.

Weights are **not** distributed in this branch. Obtain them according to their
upstream licenses and configure `configs/base.yaml` accordingly. Both source
directories must contain `hubconf.py` before the local metrics can run.
