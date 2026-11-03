"""Generation jobs and engine settings for the native MiniMax-H3 engine (studio/engine)."""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import IS_WINDOWS, find_engine_python, resolve_model_path

FPS = 24
RELEASED_SECONDS = (5.0, 15.0)  # the released model is trained on this range; outside it is experimental

# H3 canvas rule: short edge at the tier size, area capped at short*long_cap, sides multiples of 32
RESOLUTION_TIERS = [
    ("Native · 768p (recommended)", 768, 1344),
    ("Fast · 544p", 544, 960),
    ("Draft · 384p", 384, 672),
]
ASPECTS = [("16:9", 16, 9), ("9:16", 9, 16), ("4:3", 4, 3), ("3:4", 3, 4), ("3:2", 3, 2),
           ("2:3", 2, 3), ("1:1", 1, 1), ("21:9", 21, 9)]


def frames_for_seconds(seconds: float) -> int:
    """Nearest frame count on H3's 17n+5 grid (n >= 1) for a duration at 24 fps."""
    n = max(1, round((seconds * FPS - 5) / 17))
    return 17 * n + 5


def is_released_duration(frame_count: int) -> bool:
    return RELEASED_SECONDS[0] <= frame_count / FPS <= RELEASED_SECONDS[1]


def native_canvas(ratio: float, short_edge: int = 768, long_cap: int = 1344) -> tuple[int, int]:
    w, h = (short_edge * ratio, short_edge) if ratio >= 1 else (short_edge, short_edge / ratio)
    cap = short_edge * long_cap
    if w * h > cap:
        s = (cap / (w * h)) ** 0.5
        w, h = w * s, h * s
    return max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)


# attention backends the engine supports: mode -> (package, description); sdpa needs nothing extra
ATTENTION_BACKENDS = {
    "sdpa": (None, "PyTorch SDPA (default, exact)"),
    "ck_int8": ("comfy_kitchen", "comfy-kitchen INT8 (fastest; same seed gives different details)"),
    "sageattn": ("sageattention", "SageAttention (faster; same seed gives different details)"),
    "flash": ("flash_attn", "FlashAttention 2"),
    "flash3": ("flash_attn_interface", "FlashAttention 3 (Hopper)"),
    "xformers": ("xformers", "xFormers"),
}


def available_attention(cfg: dict) -> dict[str, bool] | None:
    """Which attention backends are installed in the engine environment (None if unknown).

    Only checks that the packages exist, so it needs no PyTorch import and takes a moment.
    """
    python = find_engine_python(cfg)
    if not python or not Path(python).is_file():
        return None
    packages = {mode: pkg for mode, (pkg, _) in ATTENTION_BACKENDS.items() if pkg}
    code = ("import importlib.util as u, json; "
            f"print(json.dumps({{m: u.find_spec(p) is not None for m, p in {packages!r}.items()}}))")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}
    try:
        out = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=30, **kwargs)
        found = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"sdpa": True, **found}


@dataclass
class Job:
    task: str  # t2va | i2va | fl2va | l2va
    prompt: str
    width: int
    height: int
    frame_count: int
    steps: int
    seed: int
    output_path: str
    first_frame: str = ""
    last_frame: str = ""
    loras: list[tuple[str, float]] = field(default_factory=list)  # (path, multiplier)
    shift_video: float | None = None
    shift_audio: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def engine_config(cfg: dict) -> dict:
    """The model paths and memory settings the engine needs, with paths resolved."""
    return {
        "dit": resolve_model_path(cfg, cfg["dit"]),
        "text_encoder": resolve_model_path(cfg, cfg["text_encoder"]),
        "video_vae": resolve_model_path(cfg, cfg["video_vae"]),
        "audio_vae": resolve_model_path(cfg, cfg["audio_vae"]),
        "blocks_to_swap": int(cfg.get("blocks_to_swap") or 0),
        "text_encoder_blocks_to_swap": int(cfg.get("text_encoder_blocks_to_swap") or 0),
        "attn_mode": cfg.get("attn_mode") or "sdpa",
        "keep_loaded": bool(cfg.get("keep_loaded", True)),
    }


def check_setup(cfg: dict) -> list[str]:
    """Return a list of human-readable problems (empty = ready)."""
    problems = []
    python = find_engine_python(cfg)
    if not python or not Path(python).is_file():
        problems.append("Engine Python not found: set 'Engine Python' in Settings "
                        "(a Python environment with PyTorch, see README).")
    for key, label in (("dit", "Transformer (DiT)"), ("text_encoder", "Text encoder"),
                       ("video_vae", "Video VAE"), ("audio_vae", "Audio VAE")):
        p = resolve_model_path(cfg, cfg.get(key, ""))
        if not p or not Path(p).is_file():
            problems.append(f"{label} missing: {p or '(not set)'}")
    return problems
