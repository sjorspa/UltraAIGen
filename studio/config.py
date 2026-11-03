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


ENGINE_VENV = APP_DIR / ".engine-venv"
_VENV_PYTHON = ("Scripts", "python.exe") if IS_WINDOWS else ("bin", "python")


DEFAULTS = {
    "models_dir": _default_models_dir(),
    "dit": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
    "lora_dir": "loras",
    "turbo_lora": "minimax_h3_turbo_v4_step600_ema.safetensors",  # relative to lora_dir
    "engine_python": "",  # empty = auto-detect (see find_engine_python)
    "keep_loaded": True,  # keep the transformer loaded between generations
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


def find_engine_python(cfg: dict) -> str:
    """The Python that runs the engine: it needs PyTorch, transformers, av and friends.

    Order: the configured path, the app's own .engine-venv, then the venv of an existing
    musubi-tuner checkout (only its Python environment is used, not its code).
    """
    if cfg.get("engine_python"):
        return cfg["engine_python"]
    candidates = [ENGINE_VENV.joinpath(*_VENV_PYTHON)]
    if cfg.get("musubi_python"):  # setting from older versions
        candidates.append(Path(cfg["musubi_python"]))
    for root in (cfg.get("musubi_dir"), Path.home() / "software" / "musubi-tuner", APP_DIR.parent / "musubi-tuner"):
        if root:
            candidates += [Path(root).joinpath(v, *_VENV_PYTHON) for v in (".venv", "venv")]
    return next((str(c) for c in candidates if c.is_file()), "")


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
