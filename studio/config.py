"""Settings: model paths, backend location, LM Studio endpoint and sampler defaults.

All settings live in ``config.json`` next to the app so the tool stays portable
between Linux and Windows. Model paths may be absolute or relative to
``models_dir``.
"""

import copy
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.json"
CACHE_DIR = APP_DIR / "cache"

IS_WINDOWS = sys.platform.startswith("win")


def _default_models_dir() -> str:
    for candidate in (
        Path.home() / "software" / "ComfyUI" / "models",
        Path("C:/ComfyUI/models"),
    ):
        if candidate.is_dir():
            return str(candidate)
    return ""


def _default_musubi_dir() -> str:
    for candidate in (
        Path.home() / "software" / "musubi-tuner",
        APP_DIR.parent / "musubi-tuner",
        Path("C:/musubi-tuner"),
    ):
        if candidate.is_dir():
            return str(candidate)
    return ""


DEFAULTS = {
    "models_dir": _default_models_dir(),
    "dit": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
    "lora_dir": "loras",
    "turbo_lora": "minimax_h3_turbo_v4_step600_ema.safetensors",  # relative to lora_dir
    "musubi_dir": _default_musubi_dir(),
    "musubi_python": "",  # empty = auto-detect the musubi venv
    "output_dir": str(APP_DIR / "outputs"),
    "lmstudio_url": "http://127.0.0.1:1234/v1",
    "llm_model": "",
    "llm_temperature": 0.7,
    "llm_max_tokens": 8192,
    "send_images_to_llm": True,
    "unload_llm_before_generate": True,  # LLM + H3 together do not fit in 32 GB VRAM
    "blocks_to_swap": 16,
    "text_encoder_blocks_to_swap": 0,
    "attn_mode": "sdpa",
    "extra_args": "",
    "last": {},  # last used generation parameters, restored on start
}


def resolve_model_path(cfg: dict, value: str) -> str:
    """Absolute paths stay as-is; relative ones are resolved against models_dir."""
    if not value:
        return ""
    p = Path(os.path.expanduser(value))
    if not p.is_absolute() and cfg.get("models_dir"):
        p = Path(cfg["models_dir"]) / p
    return str(p)


def find_musubi_python(cfg: dict) -> str:
    if cfg.get("musubi_python"):
        return cfg["musubi_python"]
    root = Path(cfg.get("musubi_dir") or "")
    rel = ("Scripts", "python.exe") if IS_WINDOWS else ("bin", "python")
    for venv in (".venv", "venv"):
        candidate = root.joinpath(venv, *rel)
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
