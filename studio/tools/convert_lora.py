"""Convert a PEFT/diffusers-style MiniMax-H3 LoRA to the musubi ``lora_unet_*`` layout.

Runs inside the musubi venv (needs torch + safetensors):
    python convert_lora.py INPUT OUTPUT [DIT]

``blocks.0.attn.qkv_proj.lora_A.weight`` becomes
``lora_unet_blocks_0_attn_qkv_proj.lora_down.weight``. PEFT LoRAs apply
``W + B @ A`` (scale 1), so each module gets ``alpha = rank``.

When DIT is a pruned-AdaLN checkpoint (it carries ``adaln_t_table``), its AdaLN
projections take the 8-wide curve coefficients instead of the 2688-wide time
embedding. The projection between the two is not stored in the pruned file, so
full-width AdaLN LoRA deltas cannot be mapped and are dropped; attention and
MLP deltas are kept.

Exit code 3 means nothing needed to change and no file was written.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "base_model.model.")


def is_pruned_dit(path: str) -> bool:
    with safe_open(path, "pt") as f:
        return "adaln_t_table" in f.keys()


def main(src: str, dst: str, dit: str | None = None) -> int:
    sd = load_file(src)
    drop_adaln = bool(dit) and is_pruned_dit(dit) and any(
        "adaln_proj" in k and ("lora_A" in k or "lora_down" in k) and v.shape[-1] != 8 for k, v in sd.items())
    if any(k.startswith("lora_unet_") for k in sd) and not drop_adaln:
        return 3
    out = {}
    dropped = 0
    for key, value in sd.items():
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key.startswith("lora_unet_"):  # already musubi layout: only filtering applies
            if drop_adaln and "adaln_proj" in key:
                dropped += 1
            else:
                out[key] = value.contiguous()
            continue
        if drop_adaln and "adaln_proj" in key:
            dropped += 1
            continue
        for tag, new in ((".lora_A.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_up.weight"),
                         (".lora_down.weight", ".lora_down.weight"), (".lora_up.weight", ".lora_up.weight"),
                         (".alpha", ".alpha")):
            if key.endswith(tag):
                module = key[: -len(tag)]
                name = "lora_unet_" + module.replace(".", "_")
                out[name + new] = value.contiguous()
                if new == ".lora_down.weight" and name + ".alpha" not in out:
                    out[name + ".alpha"] = torch.tensor(float(value.shape[0]))
                break
        else:
            print(f"skipping unknown key: {key}", file=sys.stderr)
    if not out:
        print("no LoRA tensors found", file=sys.stderr)
        return 2
    save_file(out, dst, metadata={"converted_from": src, "dropped_adaln_tensors": str(dropped)})
    print(f"converted {len(out)} tensors -> {dst}" + (f" (dropped {dropped} AdaLN tensors for pruned DiT)" if dropped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
