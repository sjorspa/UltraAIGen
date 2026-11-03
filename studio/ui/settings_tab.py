"""Settings tab: configurable model paths, backend, LM Studio and memory options."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .. import backend
from ..config import find_engine_python, resolve_model_path, save_config
from .widgets import PathField, card

SAFETENSORS = "Safetensors (*.safetensors)"


class SettingsTab(QWidget):
    saved = Signal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(14)

        models_dir = lambda: self.models_dir.text()  # noqa: E731

        # --- models
        c, cl = card("Models")
        hint = QLabel("Paths can be absolute or relative to the models folder. "
                      "Supported: BF16 or ConvRot-INT8 DiT, BF16/INT8/NVFP4 Qwen3-VL text encoder.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        cl.addWidget(hint)
        f = QFormLayout()
        f.setSpacing(8)
        self.models_dir = PathField("dir")
        self.dit = PathField("file", SAFETENSORS, models_dir)
        self.text_encoder = PathField("file", SAFETENSORS, models_dir)
        self.video_vae = PathField("file", SAFETENSORS, models_dir)
        self.audio_vae = PathField("file", SAFETENSORS, models_dir)
        self.lora_dir = PathField("dir")
        self.turbo_lora = PathField("file", SAFETENSORS, lambda: self.lora_dir_abs())
        f.addRow("Models folder", self.models_dir)
        f.addRow("Transformer (DiT)", self.dit)
        f.addRow("Text encoder", self.text_encoder)
        f.addRow("Video VAE", self.video_vae)
        f.addRow("Audio VAE", self.audio_vae)
        f.addRow("LoRA folder", self.lora_dir)
        f.addRow("Turbo-LoRA", self.turbo_lora)
        cl.addLayout(f)
        lay.addWidget(c)

        # --- backend
        c, cl = card("Generation engine (native PyTorch)")
        hint = QLabel("Videos are generated locally by the built-in MiniMax-H3 engine (PyTorch, no ComfyUI). "
                      "Leave 'Engine Python' empty to detect a Python environment with PyTorch automatically.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        cl.addWidget(hint)
        f = QFormLayout()
        self.engine_python = PathField("file")
        self.engine_python.edit.setPlaceholderText("auto")
        self.keep_loaded = QCheckBox("Keep models loaded between generations (much faster repeats; uses ~30 GB RAM)")
        self.output_dir = PathField("dir")
        self.blocks_to_swap = QSpinBox(minimum=0, maximum=48)
        self.blocks_to_swap.setToolTip("How many of the 50 DiT blocks stay in RAM and are streamed to the GPU each step. "
                                       "Higher = less VRAM, slightly slower. 16 works well on 32 GB.")
        self.te_blocks_to_swap = QSpinBox(minimum=0, maximum=64)
        self.te_blocks_to_swap.setToolTip("Stream text-encoder layers from RAM (50 for GPUs with ≤24 GB).")
        self.attn_mode = QComboBox()
        self.attn_hint = QLabel()
        self.attn_hint.setObjectName("hint")
        self.attn_hint.setWordWrap(True)
        f.addRow("Engine Python", self.engine_python)
        f.addRow("", self.keep_loaded)
        f.addRow("Output folder", self.output_dir)
        f.addRow("DiT blocks to swap", self.blocks_to_swap)
        f.addRow("Text encoder blocks to swap", self.te_blocks_to_swap)
        f.addRow("Attention", self.attn_mode)
        f.addRow("", self.attn_hint)
        cl.addLayout(f)
        lay.addWidget(c)

        # --- LM Studio
        c, cl = card("LM Studio (prompt enhancement)")
        f = QFormLayout()
        self.lm_url = QLineEdit()
        self.temperature = QDoubleSpinBox(minimum=0.0, maximum=2.0, singleStep=0.05, decimals=2)
        self.max_tokens = QSpinBox(minimum=256, maximum=32768, singleStep=256)
        self.send_images = QCheckBox("Send keyframes to the LLM (needs a vision model 👁)")
        self.unload_llm = QCheckBox("Unload the LLM from VRAM before generating (recommended: both do not fit together)")
        f.addRow("API URL", self.lm_url)
        f.addRow("Temperature", self.temperature)
        f.addRow("Max tokens", self.max_tokens)
        f.addRow("", self.send_images)
        f.addRow("", self.unload_llm)
        cl.addLayout(f)
        lay.addWidget(c)

        self.attn_mode.setMaximumWidth(520)
        for w in (self.blocks_to_swap, self.te_blocks_to_swap, self.temperature, self.max_tokens):
            w.setMaximumWidth(220)

        row = QHBoxLayout()
        check = QPushButton("Check setup")
        check.clicked.connect(self.check)
        save = QPushButton("Save")
        save.setObjectName("accent")
        save.clicked.connect(self.save)
        row.addStretch()
        row.addWidget(check)
        row.addWidget(save)
        lay.addLayout(row)
        lay.addStretch()
        self.load()

    def lora_dir_abs(self) -> str:
        return resolve_model_path({"models_dir": self.models_dir.text()}, self.lora_dir.text())

    def load(self):
        c = self.cfg
        self.turbo_lora.setText(c["turbo_lora"])
        self.models_dir.setText(c["models_dir"])
        self.dit.setText(c["dit"])
        self.text_encoder.setText(c["text_encoder"])
        self.video_vae.setText(c["video_vae"])
        self.audio_vae.setText(c["audio_vae"])
        self.lora_dir.setText(c["lora_dir"])
        self.engine_python.setText(c["engine_python"])
        self.keep_loaded.setChecked(bool(c["keep_loaded"]))
        self.output_dir.setText(c["output_dir"])
        self.blocks_to_swap.setValue(int(c["blocks_to_swap"]))
        self.te_blocks_to_swap.setValue(int(c["text_encoder_blocks_to_swap"]))
        self.refresh_attention()
        self.lm_url.setText(c["lmstudio_url"])
        self.temperature.setValue(float(c["llm_temperature"]))
        self.max_tokens.setValue(int(c["llm_max_tokens"]))
        self.send_images.setChecked(bool(c["send_images_to_llm"]))
        self.unload_llm.setChecked(bool(c["unload_llm_before_generate"]))

    def apply(self):
        self.cfg.update({
            "models_dir": self.models_dir.text(),
            "dit": self.dit.text(),
            "text_encoder": self.text_encoder.text(),
            "video_vae": self.video_vae.text(),
            "audio_vae": self.audio_vae.text(),
            "lora_dir": self.lora_dir.text(),
            "turbo_lora": self.turbo_lora.text(),
            "engine_python": self.engine_python.text(),
            "keep_loaded": self.keep_loaded.isChecked(),
            "output_dir": self.output_dir.text(),
            "blocks_to_swap": self.blocks_to_swap.value(),
            "text_encoder_blocks_to_swap": self.te_blocks_to_swap.value(),
            "attn_mode": self.attn_mode.currentData() or "sdpa",
            "lmstudio_url": self.lm_url.text().strip(),
            "llm_temperature": self.temperature.value(),
            "llm_max_tokens": self.max_tokens.value(),
            "send_images_to_llm": self.send_images.isChecked(),
            "unload_llm_before_generate": self.unload_llm.isChecked(),
        })

    def refresh_attention(self):
        """List the attention backends with what is installed in the engine environment."""
        installed = backend.available_attention(self.cfg) or {"sdpa": True}
        current = self.attn_mode.currentData() or self.cfg.get("attn_mode") or "sdpa"
        if current == "torch":  # older name for sdpa
            current = "sdpa"
        self.attn_mode.clear()
        for mode, (_pkg, description) in backend.ATTENTION_BACKENDS.items():
            ok = installed.get(mode, False)
            self.attn_mode.addItem(f"{mode}  ·  {description}" + ("" if ok else "  (not installed)"), mode)
            item = self.attn_mode.model().item(self.attn_mode.count() - 1)
            item.setEnabled(ok or mode == current)
        self.attn_mode.setCurrentIndex(max(0, self.attn_mode.findData(current)))
        missing = [] if installed.get("sageattn") else ["sageattn"]
        self.attn_hint.setText(
            "SageAttention is not installed. Add it with  ./setup_engine.sh --sage  (Windows: setup_engine.bat sage)."
            if missing else "If a chosen backend is missing, the engine falls back to sdpa.")

    def save(self):
        self.apply()
        save_config(self.cfg)
        self.saved.emit()

    def check(self):
        self.apply()
        self.refresh_attention()
        problems = backend.check_setup(self.cfg)
        if problems:
            QMessageBox.warning(self, "Setup", "\n".join(problems))
        else:
            QMessageBox.information(self, "Setup", f"All models and the engine Python found:\n{find_engine_python(self.cfg)}\n\nReady to generate.")
