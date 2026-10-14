"""VLM backends used by ACWMBench metrics.

The default backend is a local-transformers wrapper. It auto-detects the
model family from `config.json` and works for both supported checkpoints:

  * `Qwen/Qwen3-VL-8B-Instruct`  - dense transformer, fast on RTX 4090,
                                   no extra kernels required.
  * `Qwen/Qwen3.5-9B`            - hybrid Gated-DeltaNet, requires
                                   `flash-linear-attention` + `causal-conv1d`
                                   (otherwise warns and falls back to a slow
                                   pure-torch path).

The public interface is intentionally narrow (`chat(messages) -> str`) so we
can swap in an OpenAI-compatible server (vLLM / SGLang / DashScope) later
without touching metric code.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
def strip_thinking(text: str) -> str:
    """Remove `<think>...</think>` blocks emitted by reasoning-tuned models."""
    if not text:
        return text
    return THINK_PATTERN.sub("", text).strip()


def load_api_key_env_file(path: str | os.PathLike | None = None) -> None:
    """Load an explicitly configured KEY=VALUE file into ``os.environ``."""
    configured_path = path or os.environ.get("ACWMBENCH_API_KEYS_FILE")
    if not configured_path:
        return
    env_path = Path(configured_path).expanduser()
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    import base64

    data = _pil_to_png_bytes(img)
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_system_and_user_parts(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    user_parts: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue
        if isinstance(content, str):
            user_parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            user_parts.extend(item for item in content if isinstance(item, dict))
    return "\n\n".join(system_parts), user_parts


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


@dataclass
class VLMConfig:
    """Backend-agnostic VLM configuration."""

    backend: str = "transformers"
    model_path: str = ""
    device_map: str = "auto"
    dtype: str = "auto"
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    disable_thinking: bool = True
    trust_remote_code: bool = True
    max_image_long_side: Optional[int] = 768
    extra: Dict[str, Any] = field(default_factory=dict)


class VLMBackend(ABC):
    def __init__(self, config: VLMConfig):
        self.config = config

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """Run a single chat turn and return the (post-processed) reply."""

    def close(self) -> None:
        pass

    def get_usage_summary(self) -> Optional[Dict[str, Any]]:
        """Return backend-reported token usage stats, if available."""
        return None


# ---------------------------------------------------------------------------
# Local transformers backend (default)
# ---------------------------------------------------------------------------


_QWEN3_5_FAMILIES = {"qwen3_5", "qwen3_5_moe"}


def _detect_model_family(model_path: str) -> str:
    """Best-effort family name from `config.json` (e.g. 'qwen3_vl', 'qwen3_5')."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return "unknown"
    try:
        import json
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except Exception:
        return "unknown"
    return cfg.get("model_type") or "unknown"


def _check_qwen3_5_kernels() -> None:
    """Warn loudly if Qwen3.5 was selected but fla / causal_conv1d are missing."""
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )
    except Exception:
        return
    missing = []
    if not is_flash_linear_attention_available():
        missing.append("flash-linear-attention")
    if not is_causal_conv1d_available():
        missing.append("causal-conv1d")
    if missing:
        logger.warning(
            "Qwen3.5 selected but missing fused kernels: %s. "
            "Decode will fall back to a slow pure-torch path (~0.1 tok/s). "
            "See requirements.txt for install instructions.",
            ", ".join(missing),
        )


class TransformersVLMBackend(VLMBackend):
    """Loads a Qwen3-VL / Qwen3.5 style model via 🤗 transformers."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("VLMConfig.model_path must be set for transformers backend")
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"Model path {config.model_path!r} does not exist. "
                "Did you run scripts/download_models.sh?"
            )

        self.model_family = _detect_model_family(config.model_path)
        logger.info(
            "Loading VLM: path=%s  family=%s", config.model_path, self.model_family
        )
        if self.model_family in _QWEN3_5_FAMILIES:
            _check_qwen3_5_kernels()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = self._resolve_dtype(config.dtype)
        model_kwargs: Dict[str, Any] = {
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_path, **model_kwargs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )

    @staticmethod
    def _resolve_dtype(name: str):
        if name in (None, "", "auto"):
            return "auto"
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if name in mapping:
            return mapping[name]
        raise ValueError(f"Unsupported dtype: {name!r}")

    def _resize_images(self, messages: List[Dict[str, Any]]) -> None:
        """In-place clamp of overly large PIL images to keep VRAM under control."""
        long_side = self.config.max_image_long_side
        if not long_side:
            return
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                img = item.get("image") if isinstance(item, dict) else None
                if isinstance(img, Image.Image) and max(img.size) > long_side:
                    scale = long_side / float(max(img.size))
                    new_size = (
                        int(round(img.size[0] * scale)),
                        int(round(img.size[1] * scale)),
                    )
                    item["image"] = img.resize(new_size, Image.BICUBIC)

    def _apply_chat_template(self, messages):
        kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if self.config.disable_thinking:
            try:
                return self.processor.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                logger.debug(
                    "Processor.apply_chat_template does not accept "
                    "`enable_thinking`; will strip <think> blocks post-hoc."
                )
        return self.processor.apply_chat_template(messages, **kwargs)

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, Any]]) -> str:
        self._resize_images(messages)
        inputs = self._apply_chat_template(messages)
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
        }
        if self.config.temperature and self.config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(**inputs, **gen_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            trimmed = outputs[:, input_ids.shape[-1] :]
        else:
            trimmed = outputs
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if self.config.disable_thinking:
            text = strip_thinking(text)
        return text


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (vLLM / SGLang / DashScope)
# ---------------------------------------------------------------------------


class OpenAIVLMBackend(VLMBackend):
    """Hit any OpenAI-compatible `/v1/chat/completions` endpoint.

    Tested against a local vLLM 0.20.x server started with::

        vllm serve <model_path> --port 8000 --dtype bfloat16 \\
            --served-model-name qwen3_vl_8b --gpu-memory-utilization 0.85

    Required `config.extra` (or top-level VLMConfig.extra) keys:
        base_url            : default "http://localhost:8000/v1"
        served_model_name   : the `model` field sent in each request
        api_key             : default "EMPTY" (vLLM ignores it)
        request_timeout     : seconds, default 120

    Images in `messages` may be one of:
        - `{"type": "image", "image": <PIL.Image.Image>}`     (our convention)
        - `{"type": "image_url", "image_url": {"url": "..."}}` (raw OpenAI form,
                                                                 forwarded as-is)
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAIVLMBackend requires the `openai` package. "
                "Run `pip install openai>=1.40` in the acwmbench env."
            ) from exc

        extras = config.extra or {}
        load_api_key_env_file(extras.get("api_key_file"))
        self.base_url = extras.get("base_url", "http://localhost:8000/v1")
        self.served_model_name = (
            extras.get("served_model_name") or config.model_path or "default"
        )
        api_key_env = extras.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = extras.get("api_key") or os.environ.get(api_key_env, "EMPTY")
        self.request_timeout = float(extras.get("request_timeout", 120))
        self.max_retries = int(extras.get("max_retries", 0))

        logger.info(
            "OpenAIVLMBackend: base_url=%s model=%s",
            self.base_url, self.served_model_name,
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )
        self._usage_summary: Dict[str, Any] = {
            "backend": "openai",
            "model": self.served_model_name,
            "n_requests": 0,
            "n_requests_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _resize_images(self, messages: List[Dict[str, Any]]) -> None:
        long_side = self.config.max_image_long_side
        if not long_side:
            return
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                img = item.get("image") if isinstance(item, dict) else None
                if isinstance(img, Image.Image) and max(img.size) > long_side:
                    scale = long_side / float(max(img.size))
                    new_size = (
                        int(round(img.size[0] * scale)),
                        int(round(img.size[1] * scale)),
                    )
                    item["image"] = img.resize(new_size, Image.BICUBIC)

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Translate our internal chat format into strict OpenAI chat format."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            role = msg.get("role", "user")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            new_parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text":
                    new_parts.append({"type": "text", "text": item["text"]})
                elif t == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        url = _pil_to_data_url(img)
                    elif isinstance(img, str):
                        url = img
                    else:
                        raise ValueError(
                            f"Unsupported image payload type {type(img)!r}"
                        )
                    new_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                elif t == "image_url":
                    new_parts.append(item)
                else:
                    logger.warning("Dropping unknown content item type=%r", t)
            out.append({"role": role, "content": new_parts})
        return out

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        self._resize_images(messages)
        oai_messages = self._convert_messages(messages)

        params: Dict[str, Any] = {
            "model": self.served_model_name,
            "messages": oai_messages,
        }
        extras = self.config.extra or {}
        max_tokens_param = str(extras.get("max_tokens_param", "auto"))
        if max_tokens_param == "auto":
            model_lower = self.served_model_name.lower()
            max_tokens_param = (
                "max_completion_tokens"
                if model_lower.startswith(("gpt-5", "o1", "o3", "o4"))
                else "max_tokens"
            )
        params[max_tokens_param] = self.config.max_new_tokens
        if self.config.temperature and self.config.temperature > 0:
            params["temperature"] = self.config.temperature
            params["top_p"] = self.config.top_p
        else:
            params["temperature"] = 0.0

        # Pass chat-template kwargs (e.g. enable_thinking=False for Qwen3.5)
        # via vLLM/SGLang's `chat_template_kwargs` extra body. Do not send this
        # to hosted OpenAI APIs because they reject unknown request fields.
        allow_extra_body = bool(extras.get("allow_extra_body")) or self.base_url.startswith(
            "http://localhost"
        ) or self.base_url.startswith("http://127.0.0.1")
        if self.config.disable_thinking and allow_extra_body:
            params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        try:
            resp = self.client.chat.completions.create(**params)
        except Exception as exc:
            # Some newer OpenAI models reject the legacy `max_tokens` parameter.
            # Retry once with `max_completion_tokens` so model swaps do not break
            # long benchmark runs near the end.
            if "max_tokens" in params and "max_completion_tokens" in str(exc):
                params["max_completion_tokens"] = params.pop("max_tokens")
                resp = self.client.chat.completions.create(**params)
            else:
                raise
        self._record_usage(resp)
        text = resp.choices[0].message.content or ""
        if self.config.disable_thinking:
            text = strip_thinking(text)
        return text

    def _record_usage(self, resp: Any) -> None:
        self._usage_summary["n_requests"] += 1
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        self._usage_summary["n_requests_with_usage"] += 1
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        self._usage_summary["prompt_tokens"] += prompt_tokens
        self._usage_summary["completion_tokens"] += completion_tokens
        self._usage_summary["total_tokens"] += total_tokens

        # Some models expose reasoning token breakdown under completion_tokens_details.
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        self._usage_summary["reasoning_tokens"] += reasoning_tokens

    def get_usage_summary(self) -> Optional[Dict[str, Any]]:
        return dict(self._usage_summary)


# ---------------------------------------------------------------------------
# Anthropic Claude backend
# ---------------------------------------------------------------------------


class AnthropicVLMBackend(VLMBackend):
    """Use Anthropic Claude models via the native Messages API."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "AnthropicVLMBackend requires the `anthropic` package. "
                "Run `pip install anthropic>=0.40` in the acwmbench env."
            ) from exc

        extras = config.extra or {}
        load_api_key_env_file(extras.get("api_key_file"))
        api_key_env = extras.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_key = extras.get("api_key") or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing Anthropic API key. Set {api_key_env} in the environment.")
        self.model = extras.get("served_model_name") or config.model_path or "claude-sonnet-4-5"
        self.client = Anthropic(api_key=self.api_key)
        logger.info("AnthropicVLMBackend: model=%s", self.model)

    @staticmethod
    def _convert_parts(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import base64

        out: List[Dict[str, Any]] = []
        for item in parts:
            t = item.get("type")
            if t == "text":
                out.append({"type": "text", "text": str(item.get("text", ""))})
            elif t == "image":
                img = item.get("image")
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Anthropic backend expects PIL images, got {type(img)!r}")
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(_pil_to_png_bytes(img)).decode("ascii"),
                        },
                    }
                )
        return out

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        system_text, user_parts = _extract_system_and_user_parts(messages)
        content = self._convert_parts(user_parts)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature or 0.0,
            system=system_text or None,
            messages=[{"role": "user", "content": content}],
        )
        texts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return strip_thinking("\n".join(texts).strip()) if self.config.disable_thinking else "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Google Gemini backend
# ---------------------------------------------------------------------------


class GeminiVLMBackend(VLMBackend):
    """Use Google Gemini models via the google-genai SDK."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "GeminiVLMBackend requires the `google-genai` package. "
                "Run `pip install google-genai>=1.0` in the acwmbench env."
            ) from exc

        extras = config.extra or {}
        load_api_key_env_file(extras.get("api_key_file"))
        api_key_env = extras.get("api_key_env", "GEMINI_API_KEY")
        self.api_key = extras.get("api_key") or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing Gemini API key. Set {api_key_env} in the environment.")
        self.model = extras.get("served_model_name") or config.model_path or "gemini-2.5-pro"
        self.client = genai.Client(api_key=self.api_key)
        logger.info("GeminiVLMBackend: model=%s", self.model)

    @staticmethod
    def _convert_parts(parts: List[Dict[str, Any]]) -> List[Any]:
        from google.genai import types

        out: List[Any] = []
        for item in parts:
            t = item.get("type")
            if t == "text":
                out.append(types.Part.from_text(text=str(item.get("text", ""))))
            elif t == "image":
                img = item.get("image")
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Gemini backend expects PIL images, got {type(img)!r}")
                out.append(types.Part.from_bytes(data=_pil_to_png_bytes(img), mime_type="image/png"))
        return out

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        from google.genai import types

        system_text, user_parts = _extract_system_and_user_parts(messages)
        contents = [
            types.Content(role="user", parts=self._convert_parts(user_parts)),
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_text or None,
            max_output_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature or 0.0,
            top_p=self.config.top_p,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        text = resp.text or ""
        return strip_thinking(text) if self.config.disable_thinking else text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_vlm_backend(config: Dict[str, Any] | VLMConfig) -> VLMBackend:
    if isinstance(config, dict):
        # Filter dict to VLMConfig fields, route the rest to `extra`.
        known = {f for f in VLMConfig.__dataclass_fields__}
        extra = {k: v for k, v in config.items() if k not in known}
        kwargs = {k: v for k, v in config.items() if k in known}
        kwargs["extra"] = extra
        config = VLMConfig(**kwargs)

    if config.backend == "transformers":
        return TransformersVLMBackend(config)
    if config.backend in ("openai", "vllm", "sglang"):
        return OpenAIVLMBackend(config)
    if config.backend in ("anthropic", "claude"):
        return AnthropicVLMBackend(config)
    if config.backend in ("gemini", "google"):
        return GeminiVLMBackend(config)
    raise ValueError(
        f"Unknown VLM backend: {config.backend!r}. "
        "Supported: 'transformers', 'openai' (alias 'vllm' / 'sglang'), "
        "'anthropic'/'claude', 'gemini'/'google'."
    )
