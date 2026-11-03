"""Builds the native MiniMax-H3 generation command (musubi-tuner, pure PyTorch).

The GUI runs the returned command in a QProcess, so a generation never blocks
the interface and all VRAM is released when it ends.
"""

import hashlib
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import APP_DIR, CACHE_DIR, IS_WINDOWS, find_musubi_python, resolve_model_path

CONVERT_SCRIPT = APP_DIR / "studio" / "tools" / "convert_lora.py"
LAUNCHER = APP_DIR / "studio" / "tools" / "run_h3.py"
FPS = 24
RELEASED_SECONDS = (5.0, 15.0)  # musubi rejects other durations without --allow_experimental_duration


def frames_for_seconds(seconds: float) -> int:
    """Nearest frame count on H3's 17n+5 grid (n >= 1) for a duration at 24 fps."""
    n = max(1, round((seconds * FPS - 5) / 17))
    return 17 * n + 5


def is_released_duration(frame_count: int) -> bool:
    return RELEASED_SECONDS[0] <= frame_count / FPS <= RELEASED_SECONDS[1]

# H3 canvas rule: short edge at the tier size, area capped at short*long_cap, sides multiples of 32
RESOLUTION_TIERS = [
    ("Native · 768p (recommended)", 768, 1344),
    ("Fast · 544p", 544, 960),
    ("Draft · 384p", 384, 672),
]
ASPECTS = [("16:9", 16, 9), ("9:16", 9, 16), ("4:3", 4, 3), ("3:4", 3, 4), ("3:2", 3, 2),
           ("2:3", 2, 3), ("1:1", 1, 1), ("21:9", 21, 9)]


def native_canvas(ratio: float, short_edge: int = 768, long_cap: int = 1344) -> tuple[int, int]:
    w, h = (short_edge * ratio, short_edge) if ratio >= 1 else (short_edge, short_edge / ratio)
    cap = short_edge * long_cap
    if w * h > cap:
        s = (cap / (w * h)) ** 0.5
        w, h = w * s, h * s
    return max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)


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


class BackendError(RuntimeError):
    pass


def generate_script(cfg: dict) -> Path:
    return Path(cfg.get("musubi_dir") or "") / "src" / "musubi_tuner" / "minimax_h3_generate_video.py"


def check_setup(cfg: dict) -> list[str]:
    """Return a list of human-readable problems (empty = ready)."""
    problems = []
    if not generate_script(cfg).is_file():
        problems.append(f"musubi-tuner not found: {generate_script(cfg)}")
    if not Path(find_musubi_python(cfg)).is_file():
        problems.append(f"musubi Python not found: {find_musubi_python(cfg)}")
    for key, label in (("dit", "Transformer (DiT)"), ("text_encoder", "Text encoder"),
                       ("video_vae", "Video VAE"), ("audio_vae", "Audio VAE")):
        p = resolve_model_path(cfg, cfg.get(key, ""))
        if not p or not Path(p).is_file():
            problems.append(f"{label} missing: {p or '(not set)'}")
    return problems


def prepare_lora(cfg: dict, path: str) -> str:
    """Return a musubi-compatible LoRA path, converting PEFT-style files once into the cache."""
    src = Path(path)
    dit = resolve_model_path(cfg, cfg["dit"])
    stat = src.stat()
    # the converted file depends on the DiT too (pruned DiTs drop full-width AdaLN deltas)
    tag = hashlib.sha1(f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{Path(dit).name}".encode()).hexdigest()[:12]
    dst = CACHE_DIR / "loras" / f"{src.stem}.{tag}.musubi.safetensors"
    keep = CACHE_DIR / "loras" / f"{src.stem}.{tag}.unchanged"
    if dst.is_file():
        return str(dst)
    if keep.is_file():
        return str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}
    res = subprocess.run([find_musubi_python(cfg), "-I", str(CONVERT_SCRIPT), str(src), str(dst), dit],
                         capture_output=True, text=True, **kwargs)
    if res.returncode == 3:
        keep.touch()
        return str(src)  # already musubi layout and compatible
    if res.returncode != 0:
        raise BackendError(f"LoRA conversion failed ({src.name}):\n{res.stderr[-1500:]}")
    return str(dst)


def _batch_line(job: Job) -> str:
    # musubi splits prompt lines on " --" and turns a literal "\n" into a newline
    prompt = job.prompt.replace(" --", " —").replace("\r", "").replace("\n", "\\n")
    return f"{prompt} --d {job.seed} --o {Path(job.output_path).name}"


def build_command(cfg: dict, jobs: list[Job]) -> tuple[str, list[str], str]:
    """Return (program, arguments, working directory).

    One job runs as a single generation; several jobs (same task, size and
    keyframes, different seeds) run through ``--from_file`` so every model is
    loaded only once for the whole batch.
    """
    job = jobs[0]
    task = "t2va" if job.task == "t2va" else "fl2va"
    args = [
        str(LAUNCHER),
        str(generate_script(cfg)),
        "--task", task,
        "--dit", resolve_model_path(cfg, cfg["dit"]),
        "--text_encoder", resolve_model_path(cfg, cfg["text_encoder"]),
        "--video_vae", resolve_model_path(cfg, cfg["video_vae"]),
        "--audio_vae", resolve_model_path(cfg, cfg["audio_vae"]),
        "--width", str(job.width),
        "--height", str(job.height),
        "--frame_count", str(job.frame_count),
        "--steps", str(job.steps),
        "--attn_mode", cfg.get("attn_mode") or "sdpa",
    ]
    if len(jobs) == 1:
        args += ["--prompt", job.prompt, "--seed", str(job.seed), "--output", job.output_path]
    else:
        batch = CACHE_DIR / "batch_prompts.txt"
        batch.parent.mkdir(parents=True, exist_ok=True)
        batch.write_text("\n".join(_batch_line(j) for j in jobs) + "\n", encoding="utf-8")
        args += ["--from_file", str(batch), "--output", str(Path(job.output_path).parent)]
    if job.task in ("i2va", "fl2va"):
        if not job.first_frame:
            raise BackendError("This mode needs a first frame.")
        args += ["--first_frame", job.first_frame]
    if job.task in ("fl2va", "l2va"):
        if not job.last_frame:
            raise BackendError("This mode needs a last frame.")
        args += ["--last_frame", job.last_frame]
    if not is_released_duration(job.frame_count):
        args.append("--allow_experimental_duration")
    if int(cfg.get("blocks_to_swap") or 0) > 0:
        args += ["--blocks_to_swap", str(int(cfg["blocks_to_swap"]))]
    if int(cfg.get("text_encoder_blocks_to_swap") or 0) > 0:
        args += ["--text_encoder_blocks_to_swap", str(int(cfg["text_encoder_blocks_to_swap"]))]
    if job.shift_video is not None:
        args += ["--h3_shift_video", str(job.shift_video)]
    if job.shift_audio is not None:
        args += ["--h3_shift_audio", str(job.shift_audio)]
    if job.loras:
        args += ["--lora_weight", *[prepare_lora(cfg, p) for p, _ in job.loras]]
        args += ["--lora_multiplier", *[str(m) for _, m in job.loras]]
    if cfg.get("extra_args"):
        args += shlex.split(cfg["extra_args"], posix=not IS_WINDOWS)
    return find_musubi_python(cfg), args, str(Path(cfg["musubi_dir"]))
