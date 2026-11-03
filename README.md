# Ultra AI Generator

A desktop app for generating **video with synchronized audio** locally with **MiniMax-H3**. It runs natively in PyTorch, without ComfyUI. A local LLM in **LM Studio** turns a short idea into a prompt written in the structured format H3 expects.

Works on **Linux and Windows**.

![Studio tab with a finished video](docs/screenshots/05_result.png)

---

## Features

- **Native generation.** The MiniMax-H3 implementation of [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) does the generation in pure PyTorch. It loads your existing model files directly, including the ConvRot INT8 transformer and the NVFP4 text encoder.
- **Four modes:**
  - text → video
  - first frame → video
  - first + last frame → video
  - last frame → video
- **Prompt enhancement with LM Studio:**
  - Pick any LM Studio model from a dropdown. Vision models are marked 👁 and can look at your keyframes.
  - The system prompts are editable presets, one per mode, based on the official H3 prompt-writing guide.
- **Native H3 resolutions.** Pick an aspect ratio and a resolution tier; the app computes the exact canvas H3 is trained on.
- **Duration from 1 to 15 seconds.** The app snaps it to H3's 17n+5 frame grid.
- **Turbo LoRA.** One switch, with 4, 6 or 8 steps.
- **Seed control and batches:**
  - Seed modes: fixed, random or increment.
  - Generate several videos per click. They run one after another, each with its own seed.
- **System monitor.** Live CPU, RAM, VRAM and GPU usage, with a **Clear memory** button.
- **Built-in player and history.** Every video is saved with a `.json` sidecar that holds the prompt and settings, and **Reuse prompt** restores them.
- **Configurable paths.** You set every model path, the LoRA folder, the engine location and the output folder in the app.

---

## How it works

```
 your idea ──► LM Studio (system prompt preset) ──► H3 prompt ──► musubi-tuner (native PyTorch) ──► .mp4 (video + audio)
```

1. You write a short idea, in any language.
2. The LLM rewrites it into the three fields H3 is trained on: `integrated_multimodal_description`, `overall_soundscape` and `non_diegetic_music`.
3. For keyframe modes, the app adds the official alignment instruction line itself, with the correct duration and shot number. The LLM never has to get this right.
4. The app unloads the LLM from VRAM, because the LLM and H3 do not fit together on a 32 GB GPU.
5. The app runs musubi-tuner's H3 generator as a separate process. The interface stays responsive, and all VRAM is released when the run ends.

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA with CUDA. Tested on an RTX 5090 (32 GB). Smaller GPUs work with more block swapping (see [Settings](#settings)). |
| RAM | 64 GB recommended. |
| Python | 3.10 or newer for the GUI. musubi-tuner has its own venv. |
| LM Studio | Optional. Needed only for prompt enhancement. |

### Model files

You need these MiniMax-H3 files, for example from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | Default file (relative to the models folder) |
|---|---|
| Transformer (DiT) | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |
| Turbo LoRA (optional) | `loras/minimax_h3_turbo_v4_step600_ema.safetensors` |

Other variants work too: the full or pruned BF16 transformer, and the BF16 or INT8 text encoder. You select them in **Settings**. Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before you use the weights.

---

## Installation

### 1. Generation engine (musubi-tuner)

**Linux**

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e .
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e .
```

On the first run, musubi downloads a few small tokenizer and config files from Hugging Face. It downloads no weights.

### 2. Ultra AI Generator

- **Linux:** `./run.sh`
- **Windows:** double-click `run.bat`

Both scripts create the GUI's virtual environment on the first start (PySide6 + requests) and then open the app.

### 3. First start

1. Open the **Settings** tab.
2. Check the **Models folder** and the **musubi-tuner folder**. The app looks for `~/software/ComfyUI/models` and `~/software/musubi-tuner` automatically.
3. Click **Check setup**. The app reports anything it cannot find.
4. Click **Save**.

---

## Using the app

### Studio

![Studio tab](docs/screenshots/01_studio.png)

1. **Mode.** Choose text → video or one of the keyframe modes. In a keyframe mode, drop images onto the first and last frame slots, or click a slot to browse.
2. **Idea.** Write what you want to see and hear. Dialogue in quotes stays verbatim and keeps its language, for example `'Help me even, jongen!'` becomes `<d>[Dutch] Help me even, jongen!</d>`.
3. **LLM and System prompt.**
   - Pick an LM Studio model. 👁 means it is a vision model, ● means it is loaded right now. Click ⟳ to reload the list.
   - The preset that matches the current mode is selected automatically.
4. **Enhance prompt.** The enhanced prompt streams into *Prompt for MiniMax-H3*. You can edit it before you generate. If you leave the field empty, the app uses the idea as-is.
5. **Generate video.** You see progress, the current step, the time left and the log. When the run ends, the video plays in the preview.

![Generating](docs/screenshots/02_generating.png)

### System monitor

The bar at the top right shows live **CPU**, **RAM**, **VRAM** and **GPU** usage. A bar turns orange at 75 % and red at 90 %.

**🧹 Clear memory** frees GPU and system memory:

- It unloads every model loaded in LM Studio.
- It ends leftover generator processes, for example after a crash. It asks before ending anything.
- It frees memory held by the app itself.
- If a generation is running, it first asks whether to stop it.

### Video settings

![Video settings](docs/screenshots/06_video_settings.png)

| Setting | Meaning |
|---|---|
| **Frame** | Aspect ratio (16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 1:1, 21:9, *From keyframe* or *Custom*) plus a resolution tier. **Native · 768p** is what H3 is trained on: a 768 px short edge, at most 768×1344 pixels, sides multiples of 32. **Fast · 544p** and **Draft · 384p** are quicker previews. |
| **Duration** | 1–15 s. Snapped to the nearest 17n+5 frame count at 24 fps, and the actual length is shown. The released model is trained on 5–15 s; outside that range the run is marked *experimental*. |
| **Turbo** | Uses the turbo LoRA with **4, 6 or 8 steps**, instead of the normal step count of about 20–30. |
| **Steps** | Number of model evaluations without turbo. |
| **Seed** | **Fixed** uses this seed; several videos get seed, seed+1, and so on. **Random** gives each video a new seed. **Increment** works like fixed, then moves the seed on for the next click. |
| **Count** | Number of videos per click. Each video runs as its own process, one after another. |
| **LoRA 1/2** | Extra LoRAs from the LoRA folder, each with a strength. |

### History

Every finished video appears in **History**. Click an item to play it. **Reuse prompt** restores the mode, prompt, size, duration, steps, seed and keyframes. **Open folder** opens the output folder.

### System prompts

![System prompts](docs/screenshots/03_system_prompts.png)

Each preset is a system prompt bound to one mode. You can create, duplicate, edit and delete presets. **Restore default presets** brings back the originals.

These placeholders are filled in before a prompt is sent to the LLM:

| Placeholder | Example |
|---|---|
| `{duration}` | `5.17` |
| `{frame_count}` | `124` |
| `{width}`, `{height}` | `1344`, `768` |
| `{orientation}` | `landscape`, `portrait` or `square` |

The LLM writes only the three H3 fields. The app adds the keyframe instruction line itself.

### Settings

![Settings](docs/screenshots/04_settings.png)

| Setting | Notes |
|---|---|
| Models folder and model paths | Paths can be absolute or relative to the models folder. |
| LoRA folder / Turbo LoRA | The turbo LoRA path is relative to the LoRA folder. |
| musubi-tuner folder / Python | Leave Python empty to use `.venv` or `venv` inside the musubi folder. |
| DiT blocks to swap | The number of the 50 transformer blocks that are streamed from RAM. `16` is good for 32 GB. Use more for smaller GPUs, up to 48. |
| Text encoder blocks to swap | `0` on 32 GB. `50` for GPUs with 24 GB or less. |
| Attention | `sdpa` (default), or `flash`, `sageattn` and so on if they are installed. |
| Extra arguments | Passed straight to the generator, for example `--compile`. |
| LM Studio | API URL, temperature, max tokens, *send keyframes to the LLM*, *unload the LLM before generating*. |

All settings are stored in `config.json`, and the presets in `system_prompts.json`.

---

## Performance

These figures were measured on an RTX 5090 with 16 DiT blocks swapped. Model loading is included.

| Run | Time |
|---|---|
| 960×544, 5 s, 20 steps | ~3 min 25 s |
| 960×544, 5 s, turbo LoRA, 8 steps | ~2 min |

In a batch, every video loads the models again (about 1 minute). musubi's `--from_file` batch mode would load them only once, but with block swap enabled it fails from the second prompt onward, so the app does not use it.

---

## Notes and limitations

- **Turbo LoRA on a pruned transformer.** The turbo LoRA was trained on the full transformer. The *pruned* transformer replaces the time embedding with an 8-wide lookup table, and the file does not store the projection needed to convert the LoRA's AdaLN deltas. When the selected DiT is pruned, the app therefore drops those deltas and applies the attention, MLP and token-refiner deltas. In testing this gave sharp, coherent results in 8 steps. To apply the LoRA completely, select a full (non-pruned) transformer.
- **PEFT-style LoRAs** (`lora_A`/`lora_B`) are converted once to musubi's format and cached in `cache/loras/`.
- **No negative prompt and no CFG.** The released H3 weights are guidance-distilled.
- **Reference-to-video (Ref2VA)** needs the separate Ref2VA transformer and is not in the GUI yet.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` while loading | Another program holds VRAM, often an LLM in LM Studio. Keep *Unload the LLM before generating* enabled, or raise *DiT blocks to swap*. |
| "LM Studio not reachable" | Start LM Studio's local server (Developer tab) and check the API URL in Settings. |
| Enhancing seems to hang, or no prompt comes back | Reasoning models (such as Qwen3.5-based ones) think first. The status shows *💭 thinking… N chars* and then *✍ writing*. If a model spends all its tokens on thinking, you get a message; raise *Max tokens* in Settings (default 8192). After a generation the LLM was unloaded, so the first enhance also waits for LM Studio to load it again. |
| The LLM ignores keyframes | Pick a model marked 👁 and enable *Send keyframes to the LLM*. |
| A run fails | The **Log** panel shows the generator's full output. |

## Project layout

```
minimax_studio.py        entry point
run.sh / run.bat         launchers (create the GUI venv on first run)
studio/
  config.py              settings and path resolution
  presets.py             default system prompts and H3 prompt assembly
  enhancer.py            LM Studio client (model list, streaming, unload)
  backend.py             builds the native generation command
  tools/convert_lora.py  PEFT → musubi LoRA conversion (runs in the musubi venv)
  tools/run_h3.py        launches the generator with token-refiner LoRA support
  ui/                    PySide6 interface
docs/screenshots/        images used in this README
```
