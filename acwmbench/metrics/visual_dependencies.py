"""Lazy loaders shared by visual metrics.

KeyArea calls ``ensure_lpips`` and ``ensure_jepa``. Multi-view additionally
calls ``ensure_torchmetrics``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_torchmetrics = None
_lpips = None
_jepa_model = None
_jepa_transform = None


def ensure_torchmetrics():
    global _torchmetrics
    if _torchmetrics is None:
        try:
            import torchmetrics
        except ImportError as exc:
            raise ImportError(
                "torchmetrics is required for visual metrics. "
                "Please `pip install torchmetrics`."
            ) from exc
        _torchmetrics = torchmetrics
    return _torchmetrics


def ensure_lpips():
    global _lpips
    if _lpips is None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "lpips is required for visual metrics. Please `pip install lpips`."
            ) from exc
        _lpips = lpips
    return _lpips


_VJEPA_CHECKPOINT_KEY = {
    "vjepa2_1_vit_base_384": "ema_encoder",
    "vjepa2_1_vit_large_384": "ema_encoder",
    "vjepa2_1_vit_giant_384": "target_encoder",
    "vjepa2_1_vit_gigantic_384": "target_encoder",
    "vjepa2_vit_large": "encoder",
    "vjepa2_vit_huge": "encoder",
    "vjepa2_vit_giant": "encoder",
    "vjepa2_vit_giant_384": "encoder",
}


def _clean_backbone_state_dict(state_dict):
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def ensure_jepa(
    device: str = "cuda",
    model_path: str | None = None,
    model_name: str = "vjepa2_1_vit_base_384",
    repo_or_dir: str | None = None,
):
    """Load the V-JEPA encoder used by Multi-View Consistency."""
    global _jepa_model, _jepa_transform

    if _jepa_model is not None or _jepa_model is False:
        return (
            _jepa_model if _jepa_model is not False else None,
            _jepa_transform,
        )

    try:
        out = torch.hub.load(
            repo_or_dir or "facebookresearch/vjepa2",
            model_name,
            pretrained=model_path is None,
            source="local" if repo_or_dir else "github",
        )
        encoder = out[0] if isinstance(out, tuple) else out

        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            checkpoint_key = _VJEPA_CHECKPOINT_KEY.get(model_name, "ema_encoder")
            if isinstance(ckpt, dict) and checkpoint_key in ckpt:
                state_dict = ckpt[checkpoint_key]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            state_dict = _clean_backbone_state_dict(state_dict)
            missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(
                    "V-JEPA encoder: %d missing keys (first 3: %s)",
                    len(missing),
                    missing[:3],
                )
            if unexpected:
                logger.warning(
                    "V-JEPA encoder: %d unexpected keys (first 3: %s)",
                    len(unexpected),
                    unexpected[:3],
                )
            logger.info(
                "V-JEPA encoder loaded from %s (key=%s)",
                model_path,
                checkpoint_key,
            )

        encoder.eval().to(device)
        _jepa_model = encoder

        import torchvision.transforms as transforms

        img_size = 384 if "384" in model_name else 256
        _jepa_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return _jepa_model, _jepa_transform
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load V-JEPA (%s): %s. V-JEPA similarity will be NaN.",
            model_name,
            exc,
        )
        _jepa_model = False
        _jepa_transform = None
        return None, None
