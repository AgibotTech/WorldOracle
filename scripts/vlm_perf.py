"""Detailed perf diagnostics for the loaded VLM.

What this prints:
  1. Model device map (any layer on CPU?)
  2. Parameter dtype + total param count
  3. attn_implementation actually selected
  4. Text-only generation (no vision tower) -- isolates the LLM
  5. Prefill vs decode timing (cuda events) -- isolates per-token cost
  6. nvidia-smi snapshots before / during / after generate
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from acwmbench.models.model_zoo import (  # noqa: E402
    auto_pick_default,
    resolve_model_path,
)

_CKPT_ROOT = ROOT / "ckpt"


def _resolve_model_path() -> str:
    env_val = os.environ.get("ACWMBENCH_VLM_PATH") or os.environ.get(
        "ACWMBENCH_VLM_MODEL"
    )
    if env_val:
        return resolve_model_path(env_val, ckpt_root=_CKPT_ROOT)
    picked = auto_pick_default(ckpt_root=_CKPT_ROOT)
    if picked:
        return picked
    return resolve_model_path("qwen3_vl_8b", ckpt_root=_CKPT_ROOT)


MODEL_PATH = _resolve_model_path()


def nvidia_smi() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip()
    except Exception as e:
        return f"<nvidia-smi failed: {e}>"


def main() -> None:
    print("=" * 70)
    print("Loading model from:", MODEL_PATH)
    print("=" * 70)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  load time: {time.time() - t0:.1f}s")

    print()
    print("---- Model layout --------------------------------------------------")
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        cpu_layers = [k for k, v in hf_map.items() if str(v) == "cpu"]
        disk_layers = [k for k, v in hf_map.items() if str(v) == "disk"]
        print(f"  total entries:    {len(hf_map)}")
        print(f"  on CPU:           {len(cpu_layers)}   {cpu_layers[:5]}")
        print(f"  on disk:          {len(disk_layers)} {disk_layers[:5]}")
        gpus = sorted({str(v) for v in hf_map.values() if str(v).startswith("cuda")})
        print(f"  on GPU(s):        {gpus}")
    else:
        print("  hf_device_map = None (likely loaded entirely on one device)")

    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"  param dtypes:     {dtypes}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params:     {n_params/1e9:.2f} B")

    cfg = model.config
    attn_impl = getattr(cfg, "_attn_implementation", None) or getattr(
        cfg, "attn_implementation", None
    )
    print(f"  attn_impl (cfg):  {attn_impl}")
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        text_impl = getattr(text_cfg, "_attn_implementation", None) or getattr(
            text_cfg, "attn_implementation", None
        )
        print(f"  attn_impl (text): {text_impl}")

    print(f"  nvidia-smi:       {nvidia_smi()}")

    # ------------------------------------------------------------------
    print()
    print("---- Text-only generate (no image) ---------------------------------")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count from 1 to 30 separated by commas."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"  prompt tokens: {prompt_len}")

    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    # Time prefill only (max_new_tokens=1)
    torch.cuda.synchronize()
    t_pre = time.time()
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    prefill_s = time.time() - t_pre
    print(f"  prefill (1 tok):  {prefill_s*1000:.1f} ms   (gpu {nvidia_smi()})")

    # Time decode (max_new_tokens=64)
    n_decode = 64
    torch.cuda.synchronize()
    t_dec = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=n_decode,
            do_sample=False,
            min_new_tokens=n_decode,
        )
    torch.cuda.synchronize()
    total_s = time.time() - t_dec
    new_tokens = out.shape[-1] - prompt_len
    decode_s = total_s - prefill_s
    print(
        f"  decode {new_tokens} toks: total={total_s:.2f}s  "
        f"decode-only={decode_s:.2f}s  "
        f"per-token={decode_s/max(new_tokens-1,1)*1000:.1f} ms  "
        f"tok/s={new_tokens/total_s:.2f}"
    )

    sample = processor.batch_decode(
        out[:, prompt_len:], skip_special_tokens=True
    )[0]
    print(f"  output: {sample[:120]!r}")

    # ------------------------------------------------------------------
    print()
    print("---- Decode-only sustained (256 tokens) ----------------------------")
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode():
        out2 = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            min_new_tokens=256,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t
    new2 = out2.shape[-1] - prompt_len
    print(
        f"  256 tokens: {elapsed:.2f}s  "
        f"per-token={elapsed/new2*1000:.1f} ms  "
        f"tok/s={new2/elapsed:.2f}"
    )
    print(f"  gpu after: {nvidia_smi()}")


if __name__ == "__main__":
    main()
