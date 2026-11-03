"""Ultra AI Generator engine: native MiniMax-H3 generation in a long-lived process.

The GUI starts this script once and sends it JSON commands, one per line, on
stdin. Protocol messages go to stdout as ``@@H3 {json}`` lines; everything else
(logging, library output) is plain log text.

Commands
  {"cmd": "generate", "id": str, "config": {...}, "job": {...}}
  {"cmd": "cancel"}    stop the running generation after the current step
  {"cmd": "release"}   free every loaded model
  {"cmd": "quit"}

Events
  ready · status{msg} · progress{step,total} · done{id,path,seconds}
  · error{id,msg} · cancelled{id} · released

With ``keep_loaded`` the transformer stays loaded between jobs (the text
encoder idles in RAM and streams its layers, the VAEs idle in RAM), and the
text conditioning of each prompt is cached, so a new seed or a batch skips
both the model load and the text encoder.

The orchestration follows musubi-tuner's minimax_h3_generate_video.py
(Apache-2.0); the model code itself is the vendored ``h3core`` package.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from h3core.minimax_h3.audio_vae import load_audio_vae  # noqa: E402
from h3core.minimax_h3.generation_inputs import (  # noqa: E402
    VIDEO_VAE_SPATIAL_RATIO,
    H3GenerationRequest,
    build_generation_layout,
    decode_generation_visuals,
    encode_visual_conditions,
    fl_condition_entries,
    load_generation_record,
    validate_generation_request,
)
from h3core.minimax_h3.media import PyAVH3MediaDecoder, fingerprint_file  # noqa: E402
from h3core.minimax_h3.model import load_h3_transformer  # noqa: E402
from h3core.minimax_h3.sampling import sample_joint_av_latents, synchronize_decoded_av, write_joint_av  # noqa: E402
from h3core.minimax_h3.text_encoder import (  # noqa: E402
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    presentation_fingerprint,
)
from h3core.minimax_h3.video_vae import VIDEO_VAE_DECODE_DTYPE, VIDEO_VAE_ENCODE_DTYPE, load_video_vae  # noqa: E402
from h3core.modules.custom_offloading_utils import BlockSwapConfig  # noqa: E402
from h3core.networks import lora_minimax_h3  # noqa: E402
from h3core.utils.device_utils import clean_memory_on_device  # noqa: E402

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("engine")

TEXT_CACHE_ENTRIES = 16
# in keep-loaded mode the text encoder shares the GPU with the resident transformer,
# so it streams its layers from RAM (the H3 presentation uses the first 50 Qwen layers)
RESIDENT_TEXT_ENCODER_SWAP = 50
_PEFT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "base_model.model.")

_out = sys.stdout
sys.stdout = sys.stderr  # stray prints from libraries must not corrupt the protocol


def emit(event: str, **data) -> None:
    _out.write("@@H3 " + json.dumps({"event": event, **data}) + "\n")
    _out.flush()


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- LoRA

def _musubi_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """PEFT/diffusers keys (``blocks.0.attn.qkv_proj.lora_A.weight``) → musubi
    ``lora_unet_*`` layout. PEFT applies W + B @ A, so alpha = rank (scale 1)."""
    if any(k.startswith("lora_unet_") for k in state):
        return state
    out = {}
    for key, value in state.items():
        for prefix in _PEFT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        for tag, new in ((".lora_A.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_up.weight"),
                         (".lora_down.weight", ".lora_down.weight"), (".lora_up.weight", ".lora_up.weight"),
                         (".alpha", ".alpha")):
            if key.endswith(tag):
                name = "lora_unet_" + key[: -len(tag)].replace(".", "_")
                out[name + new] = value
                if new == ".lora_down.weight" and name + ".alpha" not in out:
                    out[name + ".alpha"] = torch.tensor(float(value.shape[0]))
                break
    return out


def _fit_lora_to_model(state: dict[str, torch.Tensor], transformer) -> tuple[dict[str, torch.Tensor], int]:
    """Keep only LoRA modules whose shapes match the loaded transformer.

    The pruned-AdaLN transformer feeds its AdaLN projections 8 curve coefficients
    instead of the 2688-wide time embedding; full-width AdaLN deltas cannot be
    mapped onto it (the projection is not stored in the pruned file) and are dropped.
    """
    linears = {"lora_unet_" + name.replace(".", "_"): m for name, m in transformer.named_modules()
               if hasattr(m, "in_features") and hasattr(m, "out_features")}
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        groups.setdefault(key.split(".", 1)[0], {})[key] = value
    kept, dropped = {}, 0
    for name, tensors in groups.items():
        module = linears.get(name)
        down, up = tensors.get(name + ".lora_down.weight"), tensors.get(name + ".lora_up.weight")
        if module is None or down is None or up is None or down.shape[1] != module.in_features \
                or up.shape[0] != module.out_features:
            dropped += 1
            continue
        kept.update(tensors)
    return kept, dropped


def attach_loras(transformer, loras: list[list], device: torch.device) -> list[torch.nn.Module]:
    networks = []
    for path, multiplier in loras:
        state, dropped = _fit_lora_to_model(_musubi_lora_state(load_file(path)), transformer)
        network = lora_minimax_h3.create_arch_network_from_weights(
            float(multiplier), state, unet=transformer, for_inference=True)
        names = {lora.lora_name for lora in network.unet_loras}
        state = {k: v for k, v in state.items() if k.split(".", 1)[0] in names}
        if not names:
            raise ValueError(f"LoRA {Path(path).name} has no modules that fit this transformer")
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(state, strict=True)
        network.eval().requires_grad_(False).to(device)
        networks.append(network)
        logger.info("LoRA %s: %d modules attached%s", Path(path).name, len(names),
                    f", {dropped} skipped (no matching layer in this transformer)" if dropped else "")
    return networks


# --------------------------------------------------------------------------- engine

class Engine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = PyAVH3MediaDecoder()
        self.cancel = threading.Event()
        self.transformer = None
        self.transformer_key = None
        self.lora_networks: list = []
        self.processor = None
        self.text_encoder = None
        self.text_encoder_key = None
        self.video_vaes: dict[torch.dtype, torch.nn.Module] = {}
        self.audio_vae = None
        self.vae_paths = None
        self.text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    # ---- memory
    def release_transformer(self) -> None:
        if self.transformer is None:
            return
        if self.transformer.offloader is not None:
            self.transformer.offloader.set_forward_only(True)  # wait for pending block transfers
        self.transformer = None
        self.transformer_key = None
        self.lora_networks = []
        gc.collect()
        clean_memory_on_device(self.device)

    def release_text_encoder(self) -> None:
        self.processor = self.text_encoder = self.text_encoder_key = None
        gc.collect()
        clean_memory_on_device(self.device)

    def release(self) -> None:
        self.release_transformer()
        self.release_text_encoder()
        self.video_vaes.clear()
        self.audio_vae = None
        self.vae_paths = None
        self.text_cache.clear()
        gc.collect()
        clean_memory_on_device(self.device)

    # ---- models
    @contextmanager
    def video_vae(self, cfg: dict, dtype: torch.dtype, keep: bool):
        if not keep:
            vae = load_video_vae(cfg["video_vae"], device=self.device, dtype=dtype)
            try:
                yield vae
            finally:
                del vae
                gc.collect()
                clean_memory_on_device(self.device)
            return
        vae = self.video_vaes.get(dtype)
        if vae is None:
            vae = self.video_vaes[dtype] = load_video_vae(cfg["video_vae"], device="cpu", dtype=dtype)
        vae.to(self.device)
        try:
            yield vae
        finally:
            vae.to("cpu")
            clean_memory_on_device(self.device)

    @contextmanager
    def audio_vae_ctx(self, cfg: dict, keep: bool):
        if not keep:
            vae = load_audio_vae(cfg["audio_vae"], device=self.device, dtype=torch.float32)
            try:
                yield vae
            finally:
                del vae
                gc.collect()
                clean_memory_on_device(self.device)
            return
        if self.audio_vae is None:
            self.audio_vae = load_audio_vae(cfg["audio_vae"], device="cpu", dtype=torch.float32)
        self.audio_vae.to(self.device)
        try:
            yield self.audio_vae
        finally:
            self.audio_vae.to("cpu")
            clean_memory_on_device(self.device)

    def acquire_transformer(self, cfg: dict, loras: list[list]):
        key = (cfg["dit"], int(cfg["blocks_to_swap"]), cfg["attn_mode"],
               tuple((str(p), float(m)) for p, m in loras))
        if self.transformer is not None and self.transformer_key == key:
            # finish pending block transfers from the previous run, then reset placement
            self.transformer.switch_block_swap_for_inference()
            return self.transformer
        self.release_transformer()
        emit("status", msg="Loading transformer…")
        swap = int(cfg["blocks_to_swap"])
        attn = cfg["attn_mode"]
        transformer = load_h3_transformer(
            cfg["dit"], device="cpu" if swap else self.device, dtype=torch.bfloat16,
            attn_mode="torch" if attn == "sdpa" else attn, quant_device=self.device)
        self.lora_networks = attach_loras(transformer, loras, self.device) if loras else []
        if swap:
            transformer.enable_block_swap(swap, BlockSwapConfig(device=self.device, supports_backward=False))
            transformer.move_to_device_except_swap_blocks(self.device)
            transformer.prepare_block_swap_before_forward()
            transformer.switch_block_swap_for_inference()
        else:
            transformer.to(self.device)
        transformer.eval().requires_grad_(False)
        self.transformer, self.transformer_key = transformer, key
        return transformer

    def encode_text(self, cfg: dict, request: H3GenerationRequest, record, text_visuals, keep: bool):
        presentation = build_presentation(record, request.task, text_visuals)
        media = {Path(path): fingerprint_file(path) for _, path in fl_condition_entries(request)} \
            if request.task == "fl2va" else {}
        cache_key = presentation_fingerprint(presentation, media, frame_count=request.frame_count)
        cached = self.text_cache.get(cache_key)
        if cached is not None:
            self.text_cache.move_to_end(cache_key)
            emit("status", msg="Reusing cached text conditioning…")
            return cached
        swap = max(int(cfg["text_encoder_blocks_to_swap"]), RESIDENT_TEXT_ENCODER_SWAP if keep else 0)
        te_key = (cfg["text_encoder"], swap)
        if self.text_encoder is None or self.text_encoder_key != te_key:
            self.release_text_encoder()
            emit("status", msg="Loading text encoder…")
            self.processor = load_h3_processor()
            self.text_encoder = load_h3_text_encoder(
                cfg["text_encoder"], device=self.device, dtype=torch.bfloat16, blocks_to_swap=swap)
            self.text_encoder_key = te_key
        emit("status", msg="Encoding prompt…")
        hidden, tags = encode_h3_presentation(self.processor, self.text_encoder, presentation)
        if not keep:
            self.release_text_encoder()
        clean_memory_on_device(self.device)
        result = (hidden.to(torch.bfloat16).unsqueeze(0).cpu(), tags.unsqueeze(0).cpu())
        self.text_cache[cache_key] = result
        while len(self.text_cache) > TEXT_CACHE_ENTRIES:
            self.text_cache.popitem(last=False)
        return result

    # ---- one generation
    def generate(self, cfg: dict, job: dict) -> Path:
        keep = bool(cfg.get("keep_loaded", True))
        if self.vae_paths != (cfg["video_vae"], cfg["audio_vae"]):
            self.video_vaes.clear()
            self.audio_vae = None
            self.vae_paths = (cfg["video_vae"], cfg["audio_vae"])
        task = job["task"]
        request = H3GenerationRequest(
            task="t2va" if task == "t2va" else "fl2va",
            prompt=job["prompt"],
            first_frame=job.get("first_frame") or None if task in ("i2va", "fl2va") else None,
            last_frame=job.get("last_frame") or None if task in ("fl2va", "l2va") else None,
            width=int(job["width"]), height=int(job["height"]), frame_count=int(job["frame_count"]),
            steps=int(job["steps"]), seed=int(job["seed"]),
            allow_experimental_duration=True,  # the GUI decides which durations to offer
            **({"h3_shift_video": float(job["shift_video"])} if job.get("shift_video") else {}),
            **({"h3_shift_audio": float(job["shift_audio"])} if job.get("shift_audio") else {}),
        )
        validate_generation_request(request)
        if not keep:
            self.release()

        record = load_generation_record(request)
        raw_visuals, text_visuals = decode_generation_visuals(request, record, self.decoder)
        hidden, tags = self.encode_text(cfg, request, record, text_visuals, keep)

        visual_conditions, visual_geometries = (), ()
        if request.task != "t2va":
            emit("status", msg="Encoding keyframes…")
            with self.video_vae(cfg, VIDEO_VAE_ENCODE_DTYPE, keep) as vae:
                if vae.vae_ratio != VIDEO_VAE_SPATIAL_RATIO:
                    raise ValueError(f"video VAE spatial ratio must be {VIDEO_VAE_SPATIAL_RATIO}")
                visual_conditions, visual_geometries, _ = encode_visual_conditions(request, record, raw_visuals, vae)
        del raw_visuals, text_visuals
        layout = build_generation_layout(request, text_length=hidden.shape[1], visual_geometries=visual_geometries)

        transformer = self.acquire_transformer(cfg, job.get("loras") or [])
        emit("status", msg="Sampling…")

        def on_step(completed: int, total: int) -> None:
            emit("progress", step=completed, total=total)
            if self.cancel.is_set():
                raise Cancelled()

        try:
            sample = sample_joint_av_latents(
                transformer, layout=layout, seed=request.seed, text_hidden_states=hidden, text_token_tags=tags,
                visual_conditions=visual_conditions, steps=request.steps, video_shift=request.h3_shift_video,
                audio_shift=request.h3_shift_audio, visual_condition_clean=request.h3_visual_cond_clean,
                audio_condition_clean=request.h3_audio_cond_clean, device=self.device, step_callback=on_step)
        finally:
            if transformer.offloader is not None:
                transformer.offloader.set_forward_only(True)  # never leave block transfers pending
            if not keep:
                self.release_transformer()
            del transformer
            gc.collect()
            clean_memory_on_device(self.device)

        emit("status", msg="Decoding video…")
        with self.video_vae(cfg, VIDEO_VAE_DECODE_DTYPE, keep) as vae, torch.no_grad():
            video = vae.decode(sample.video.to(device=self.device, dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
        emit("status", msg="Decoding audio…")
        with self.audio_vae_ctx(cfg, keep) as vae, torch.no_grad():
            audio = vae.decode(sample.audio.to(device=self.device, dtype=torch.float32)).cpu()
        decoded = synchronize_decoded_av(video, audio, frame_count=request.frame_count)
        out = Path(job["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        write_joint_av(decoded, out)
        if not keep:
            self.release()
        return out


def stdin_reader(commands: queue.Queue, engine: Engine) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("ignoring malformed command: %s", line[:200])
            continue
        if cmd.get("cmd") == "cancel":
            engine.cancel.set()  # handled immediately, even during a generation
        else:
            commands.put(cmd)
    commands.put({"cmd": "quit"})


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    engine = Engine()
    commands: queue.Queue = queue.Queue()
    threading.Thread(target=stdin_reader, args=(commands, engine), daemon=True).start()
    emit("ready", device=str(engine.device),
         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    while True:
        cmd = commands.get()
        kind = cmd.get("cmd")
        if kind == "quit":
            break
        if kind == "release":
            engine.release()
            emit("released")
        elif kind == "generate":
            job_id = cmd.get("id")
            engine.cancel.clear()
            start = time.time()
            try:
                path = engine.generate(cmd["config"], cmd["job"])
                emit("done", id=job_id, path=str(path), seconds=round(time.time() - start))
            except Cancelled:
                emit("cancelled", id=job_id)
            except torch.OutOfMemoryError as e:
                engine.release()
                emit("error", id=job_id, msg=f"Out of GPU memory: {e}. All models were unloaded; "
                                             "raise 'DiT blocks to swap' or close other GPU programs.")
            except Exception as e:  # reported to the GUI; the engine keeps running
                traceback.print_exc()
                emit("error", id=job_id, msg=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
