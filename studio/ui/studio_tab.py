"""Main Studio tab: idea → LM Studio enhancement → native H3 generation → preview."""

import json
import os
import random
import re
import time
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QProcess, QProcessEnvironment, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QImageReader, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from .. import backend, enhancer
from ..config import resolve_model_path, save_config
from ..presets import TASKS, build_final_prompt, duration_str, fill_placeholders, load_presets
from .widgets import ImageDrop, card

ASPECT_FROM_IMAGE = "From keyframe"
ASPECT_CUSTOM = "Custom"
SEED_MODES = [("fixed", "Fixed"), ("random", "Random"), ("increment", "Increment")]
_TQDM_RE = re.compile(r"MiniMax-H3:\s*(\d+)%\|.*?(\d+)/(\d+)\s*\[([^\]]*)\]")


def image_ratio(path: str) -> float | None:
    size = QImageReader(path).size()
    if not size.isValid() or size.height() == 0:
        return None
    return size.width() / size.height()


class FuncThread(QThread):
    done = Signal(object)
    failed = Signal(str)
    token = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.stop_requested = False

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except Exception as e:  # surfaced in the UI
            self.failed.emit(str(e))


class StudioTab(QWidget):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.presets = load_presets()
        self.proc: QProcess | None = None
        self.jobs: list[backend.Job] = []
        self.job_meta: dict = {}
        self.out_buffer = ""
        self.done_paths: list[str] = []
        self.enhance_thread: FuncThread | None = None
        self.models_thread: FuncThread | None = None
        self.timer = QElapsedTimer()
        self.tick = QTimer(self, interval=1000, timeout=self._update_elapsed)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setSizes([620, 860])
        split.setStretchFactor(1, 1)

        self._restore_last()
        self.refresh_presets()
        self.refresh_loras()
        self.refresh_history()
        QTimer.singleShot(200, self.refresh_models)

    # ------------------------------------------------------------------ layout
    def _build_left(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(600)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(12)

        # mode
        c, cl = card("Mode")
        grid = QGridLayout()
        grid.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        for i, (key, label) in enumerate(TASKS.items()):
            b = QPushButton(label)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.setProperty("task", key)
            self.mode_group.addButton(b, i)
            grid.addWidget(b, i // 2, i % 2)
        self.mode_group.buttons()[0].setChecked(True)
        self.mode_group.buttonToggled.connect(lambda *_: self._mode_changed())
        cl.addLayout(grid)
        lay.addWidget(c)

        # keyframes
        self.kf_card, kl = card("Keyframes")
        row = QHBoxLayout()
        self.first_drop = ImageDrop("First frame")
        self.last_drop = ImageDrop("Last frame")
        self.first_drop.changed.connect(lambda _: self._update_resolution())
        row.addWidget(self.first_drop)
        row.addWidget(self.last_drop)
        kl.addLayout(row)
        lay.addWidget(self.kf_card)

        # idea + LLM
        c, cl = card("Idea")
        self.idea = QPlainTextEdit()
        self.idea.setPlaceholderText("Briefly describe what you want to see and hear. For example: a fox runs through a snowy "
                                     "forest at sunrise, the camera follows it, calm piano music.")
        self.idea.setFixedHeight(110)
        cl.addWidget(self.idea)
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        self.llm_model = QComboBox()
        self.llm_model.setMinimumWidth(180)
        self.llm_model.currentTextChanged.connect(self._llm_model_changed)
        refresh = QToolButton(text="⟳")
        refresh.setToolTip("Reload models from LM Studio")
        refresh.clicked.connect(self.refresh_models)
        self.preset = QComboBox()
        g.addWidget(QLabel("LLM"), 0, 0)
        g.addWidget(self.llm_model, 0, 1)
        g.addWidget(refresh, 0, 2)
        g.addWidget(QLabel("System prompt"), 1, 0)
        g.addWidget(self.preset, 1, 1, 1, 2)
        g.setColumnStretch(1, 1)
        cl.addLayout(g)
        row = QHBoxLayout()
        self.llm_status = QLabel("")
        self.llm_status.setObjectName("hint")
        self.enhance_btn = QPushButton("✨  Enhance prompt")
        self.enhance_btn.setObjectName("accent")
        self.enhance_btn.clicked.connect(self.enhance)
        row.addWidget(self.llm_status, 1)
        row.addWidget(self.enhance_btn)
        cl.addLayout(row)
        lay.addWidget(c)

        # final prompt
        c, cl = card("Prompt for MiniMax-H3")
        self.final = QPlainTextEdit()
        self.final.setPlaceholderText("The enhanced prompt appears here and can be edited. "
                                      "If this stays empty, the idea is used as-is.")
        self.final.setMinimumHeight(220)
        f = QFont(self.final.font())
        f.setPointSize(max(9, f.pointSize() - 1))
        self.final.setFont(f)
        cl.addWidget(self.final)
        lay.addWidget(c)

        # video settings
        c, cl = card("Video")
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)
        self.aspect = QComboBox()
        for label, a, b in backend.ASPECTS:
            self.aspect.addItem(label, a / b)
        self.aspect.addItem(ASPECT_FROM_IMAGE, 0.0)
        self.aspect.addItem(ASPECT_CUSTOM, -1.0)
        self.aspect.setToolTip("'From keyframe' uses the aspect ratio of the first (or last) frame.")
        self.tier = QComboBox()
        for label, short, cap in backend.RESOLUTION_TIERS:
            self.tier.addItem(label, (short, cap))
        self.tier.setToolTip("H3 is trained at a 768 px short edge (max. 768×1344 pixels). "
                             "Lower tiers are faster but less sharp.")
        self.aspect.currentIndexChanged.connect(lambda _: self._update_resolution())
        self.tier.currentIndexChanged.connect(lambda _: self._update_resolution())
        self.width_spin = QSpinBox(minimum=256, maximum=2048, singleStep=32)
        self.height_spin = QSpinBox(minimum=256, maximum=2048, singleStep=32)
        res_row = QHBoxLayout()
        res_row.addWidget(self.aspect, 1)
        res_row.addWidget(self.tier, 2)
        wh = QHBoxLayout()
        wh.addWidget(self.width_spin)
        wh.addWidget(QLabel("×"))
        wh.addWidget(self.height_spin)
        self.res_info = QLabel()
        self.res_info.setObjectName("hint")
        wh.addWidget(self.res_info, 1)
        self.width_spin.valueChanged.connect(self._update_res_info)
        self.height_spin.valueChanged.connect(self._update_res_info)
        self.duration = QSpinBox(minimum=1, maximum=15, suffix=" s")
        self.duration.setToolTip("H3 generates on a 17n+5 frame grid at 24 fps, so the nearest valid frame count is used.\n"
                                 "The released model is trained on 5-15 s; shorter clips are experimental.")
        self.duration_info = QLabel()
        self.duration_info.setObjectName("hint")
        self.duration.valueChanged.connect(self._update_duration_info)
        duration_row = QHBoxLayout()
        duration_row.addWidget(self.duration)
        duration_row.addWidget(self.duration_info, 1)
        self.steps = QSpinBox(minimum=1, maximum=100)
        self.steps.setToolTip("Number of model evaluations. ±20-30 without turbo LoRA.")
        self.seed = QSpinBox(minimum=0, maximum=2**31 - 1)
        dice = QToolButton(text="🎲")
        dice.setToolTip("New random seed")
        dice.clicked.connect(lambda: self.seed.setValue(random.randint(0, 2**31 - 1)))
        self.seed_mode = QComboBox()
        for key, label in SEED_MODES:
            self.seed_mode.addItem(label, key)
        self.seed_mode.setToolTip("Fixed: this seed (for several videos: seed, seed+1, …).\n"
                                  "Random: a new seed for every video.\n"
                                  "Increment: like fixed, then the seed advances for the next click.")
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(dice)
        seed_row.addWidget(self.seed_mode)
        self.turbo = QCheckBox("Turbo LoRA")
        self.turbo.setToolTip("Uses the turbo LoRA from the LoRA folder (configurable in Settings) "
                              "for fast generation in 4, 6 or 8 steps.")
        self.turbo_steps = QComboBox()
        for n in (4, 6, 8):
            self.turbo_steps.addItem(f"{n} steps", n)
        self.turbo_info = QLabel()
        self.turbo_info.setObjectName("hint")
        self.turbo.toggled.connect(self._turbo_toggled)
        turbo_row = QHBoxLayout()
        turbo_row.addWidget(self.turbo)
        turbo_row.addWidget(self.turbo_steps)
        turbo_row.addWidget(self.turbo_info, 1)
        self.count = QSpinBox(minimum=1, maximum=99)
        self.count.setToolTip("Videos per click. They run one after another, each with its own seed.")
        g.addWidget(QLabel("Frame"), 0, 0)
        g.addLayout(res_row, 0, 1)
        g.addLayout(wh, 1, 1)
        g.addWidget(QLabel("Duration"), 2, 0)
        g.addLayout(duration_row, 2, 1)
        g.addWidget(QLabel("Turbo"), 3, 0)
        g.addLayout(turbo_row, 3, 1)
        g.addWidget(QLabel("Steps"), 4, 0)
        g.addWidget(self.steps, 4, 1)
        g.addWidget(QLabel("Seed"), 5, 0)
        g.addLayout(seed_row, 5, 1)
        g.addWidget(QLabel("Count"), 6, 0)
        g.addWidget(self.count, 6, 1)
        self.lora_rows = []
        for i in range(2):
            combo = QComboBox()
            strength = QDoubleSpinBox(minimum=-2.0, maximum=3.0, singleStep=0.05, decimals=2, value=1.0)
            r = QHBoxLayout()
            r.addWidget(combo, 1)
            r.addWidget(strength)
            g.addWidget(QLabel(f"LoRA {i + 1}"), 7 + i, 0)
            g.addLayout(r, 7 + i, 1)
            self.lora_rows.append((combo, strength))
        g.setColumnStretch(1, 1)
        cl.addLayout(g)
        lay.addWidget(c)

        row = QHBoxLayout()
        self.gen_btn = QPushButton("▶  Generate video")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.clicked.connect(self.generate)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.gen_btn, 1)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        lay.addStretch()
        return scroll

    def _build_right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(12)

        c, cl = card("Preview")
        # frames are drawn into a plain label (via QVideoSink) instead of a native video
        # surface: identical on Windows/Linux/Wayland and visible in screenshots
        self.video = QLabel("No video yet")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 270)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background: #000; border-radius: 8px; color: #555;")
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self._show_frame)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoSink(self.sink)
        self.player.setLoops(QMediaPlayer.Infinite)
        cl.addWidget(self.video, 1)
        ctl = QHBoxLayout()
        self.play_btn = QToolButton(text="⏯")
        self.play_btn.clicked.connect(self._toggle_play)
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda d: self.pos_slider.setRange(0, d))
        self.player.positionChanged.connect(self._position_changed)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("hint")
        mute = QToolButton(text="🔊")
        mute.setCheckable(True)
        mute.toggled.connect(lambda m: (self.audio.setMuted(m), mute.setText("🔇" if m else "🔊")))
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.pos_slider, 1)
        ctl.addWidget(self.time_label)
        ctl.addWidget(mute)
        cl.addLayout(ctl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready.")
        self.status.setObjectName("status")
        cl.addWidget(self.progress)
        cl.addWidget(self.status)
        lay.addWidget(c, 3)

        bottom = QSplitter(Qt.Horizontal)
        c, cl = card("History")
        self.history = QListWidget()
        self.history.setWordWrap(True)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history.currentItemChanged.connect(self._history_selected)
        cl.addWidget(self.history, 1)
        row = QHBoxLayout()
        reuse = QPushButton("Reuse prompt")
        reuse.clicked.connect(self._reuse_history)
        open_dir = QPushButton("Open folder")
        open_dir.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir())))
        row.addWidget(reuse)
        row.addWidget(open_dir)
        cl.addLayout(row)
        bottom.addWidget(c)
        c, cl = card("Log")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        f = QFont("Monospace")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(9)
        self.log.setFont(f)
        cl.addWidget(self.log, 1)
        bottom.addWidget(c)
        bottom.setSizes([380, 420])
        lay.addWidget(bottom, 2)
        return w

    # ------------------------------------------------------------------ state
    def task(self) -> str:
        return self.mode_group.checkedButton().property("task")

    def _mode_changed(self):
        t = self.task()
        self.kf_card.setVisible(t != "t2va")
        self.first_drop.setVisible(t in ("i2va", "fl2va"))
        self.last_drop.setVisible(t in ("fl2va", "l2va"))
        self.refresh_presets(select_for_task=True)
        self._update_resolution()

    def _update_resolution(self):
        ratio = self.aspect.currentData()
        custom = ratio < 0
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)
        self.tier.setEnabled(not custom)
        if custom:
            self._update_res_info()
            return
        if ratio == 0:
            img = self.first_drop.path if self.task() in ("i2va", "fl2va") else self.last_drop.path
            ratio = (image_ratio(img) if img else None) or 16 / 9
        short, cap = self.tier.currentData()
        w, h = backend.native_canvas(ratio, short, cap)
        self.width_spin.setValue(w)
        self.height_spin.setValue(h)
        self._update_res_info()

    def _update_res_info(self):
        w, h = self.width_spin.value(), self.height_spin.value()
        warn = "  ⚠ not a multiple of 32" if w % 32 or h % 32 else ""
        if not warn and w * h > 768 * 1344:
            warn = "  ⚠ above native (768×1344)"
        self.res_info.setText(f"{w * h / 1e6:.2f} MP{warn}")

    def refresh_presets(self, select_for_task: bool = False):
        self.presets = load_presets()
        current = self.preset.currentText()
        self.preset.blockSignals(True)
        self.preset.clear()
        for p in self.presets:
            self.preset.addItem(p["name"], p)
        self.preset.blockSignals(False)
        idx = self.preset.findText(current)
        if select_for_task or idx < 0:
            idx = next((i for i, p in enumerate(self.presets) if p.get("task") == self.task()), 0)
        self.preset.setCurrentIndex(idx)

    def refresh_loras(self):
        lora_dir = Path(resolve_model_path(self.cfg, self.cfg.get("lora_dir", "")))
        files = sorted(p.name for p in lora_dir.glob("*.safetensors")) if lora_dir.is_dir() else []
        for combo, _ in self.lora_rows:
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— none —")
            combo.addItems(files)
            combo.setCurrentIndex(max(0, combo.findText(cur)))
            combo.blockSignals(False)

    def frames(self) -> int:
        return backend.frames_for_seconds(self.duration.value())

    def _update_duration_info(self):
        n = self.frames()
        warn = "  ·  experimental (outside 5-15 s)" if not backend.is_released_duration(n) else ""
        self.duration_info.setText(f"{n} frames = {duration_str(n)} s{warn}")

    def turbo_path(self) -> Path:
        name = self.cfg.get("turbo_lora", "")
        p = Path(name)
        if not p.is_absolute():
            p = Path(resolve_model_path(self.cfg, self.cfg.get("lora_dir", ""))) / name
        return p

    def _turbo_toggled(self, on: bool):
        self.turbo_steps.setEnabled(on)
        self.steps.setEnabled(not on)
        exists = self.turbo_path().is_file()
        self.turbo_info.setText(("" if exists else "⚠ file not found") if on else "")
        self.turbo_info.setToolTip(str(self.turbo_path()))

    def refresh_models(self):
        if self.models_thread and self.models_thread.isRunning():
            return
        self.llm_status.setText("Connecting to LM Studio…")
        url = self.cfg["lmstudio_url"]
        self.models_thread = FuncThread(lambda _t: enhancer.list_models(url), self)
        self.models_thread.done.connect(self._models_loaded)
        self.models_thread.failed.connect(lambda e: self.llm_status.setText("⚠ LM Studio not reachable"))
        self.models_thread.start()

    def _models_loaded(self, models: list):
        self.llm_model.blockSignals(True)
        self.llm_model.clear()
        for m in models:
            label = m["id"] + ("  👁" if m["vision"] else "") + ("  ●" if m["loaded"] else "")
            self.llm_model.addItem(label, m["id"])
            self.llm_model.setItemData(self.llm_model.count() - 1,
                                       ("Vision model: can look at keyframes. " if m["vision"] else "Text only. ")
                                       + ("Loaded now." if m["loaded"] else "Loads on first use."), Qt.ToolTipRole)
        idx = self.llm_model.findData(self.cfg.get("llm_model") or "")
        if idx >= 0:
            self.llm_model.setCurrentIndex(idx)
        self.llm_model.blockSignals(False)
        self.llm_status.setText(f"● {len(models)} models available" if models else "No models in LM Studio")

    def _llm_model_changed(self, _label: str):
        if self.llm_model.currentData():
            self.cfg["llm_model"] = self.llm_model.currentData()

    def _output_dir(self) -> str:
        d = Path(self.cfg["output_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def _restore_last(self):
        last = self.cfg.get("last", {})
        for b in self.mode_group.buttons():
            if b.property("task") == last.get("task", "t2va"):
                b.setChecked(True)
        self.idea.setPlainText(last.get("idea", ""))
        self.final.setPlainText(last.get("final", ""))
        self.aspect.setCurrentIndex(max(0, self.aspect.findText(last.get("aspect", "16:9"))))
        self.tier.setCurrentIndex(max(0, min(self.tier.count() - 1, last.get("tier", 0))))
        self.width_spin.setValue(last.get("width", 1344))
        self.height_spin.setValue(last.get("height", 768))
        self.duration.setValue(last.get("duration_s", 5))
        self._update_duration_info()
        self.steps.setValue(last.get("steps", 20))
        self.seed.setValue(last.get("seed", 42))
        self.seed_mode.setCurrentIndex(max(0, self.seed_mode.findData(last.get("seed_mode", "random"))))
        self.count.setValue(last.get("count", 1))
        self.turbo_steps.setCurrentIndex(max(0, self.turbo_steps.findData(last.get("turbo_steps", 8))))
        self.turbo.setChecked(last.get("turbo", False))
        self._turbo_toggled(self.turbo.isChecked())
        self.first_drop.set_path(last.get("first_frame", ""))
        self.last_drop.set_path(last.get("last_frame", ""))
        self._pending_loras = last.get("loras", [])
        self._mode_changed()

    def save_last(self):
        self.cfg["last"] = {
            "task": self.task(),
            "idea": self.idea.toPlainText(),
            "final": self.final.toPlainText(),
            "aspect": self.aspect.currentText(),
            "tier": self.tier.currentIndex(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "duration_s": self.duration.value(),
            "steps": self.steps.value(),
            "seed": self.seed.value(),
            "seed_mode": self.seed_mode.currentData(),
            "count": self.count.value(),
            "turbo": self.turbo.isChecked(),
            "turbo_steps": self.turbo_steps.currentData(),
            "first_frame": self.first_drop.path,
            "last_frame": self.last_drop.path,
            "loras": [[c.currentText(), s.value()] for c, s in self.lora_rows],
        }
        save_config(self.cfg)

    def apply_pending_loras(self):
        for (combo, strength), (name, value) in zip(self.lora_rows, getattr(self, "_pending_loras", [])):
            combo.setCurrentIndex(max(0, combo.findText(name)))
            strength.setValue(value)
        self._pending_loras = []

    # ------------------------------------------------------------------ enhance
    def _keyframe_images(self) -> list[str]:
        t = self.task()
        imgs = []
        if t in ("i2va", "fl2va") and self.first_drop.path:
            imgs.append(self.first_drop.path)
        if t in ("fl2va", "l2va") and self.last_drop.path:
            imgs.append(self.last_drop.path)
        return imgs

    def enhance(self):
        if self.enhance_thread and self.enhance_thread.isRunning():
            self.enhance_thread.stop_requested = True
            return
        idea = self.idea.toPlainText().strip()
        images = self._keyframe_images() if self.cfg.get("send_images_to_llm") else []
        if not idea and not images:
            QMessageBox.information(self, "Idea", "Write a short idea first.")
            return
        model = self.llm_model.currentData()
        if not model:
            QMessageBox.warning(self, "LM Studio", "No LLM selected. Start LM Studio and click ⟳.")
            return
        preset = self.preset.currentData() or {}
        frames = self.frames()
        system = fill_placeholders(preset.get("text", ""), frame_count=frames,
                                   width=self.width_spin.value(), height=self.height_spin.value())
        if not idea:
            idea = "(No text idea given: build the video from the attached image(s).)"
        cfg, task = self.cfg, self.task()

        def run(th: FuncThread):
            return enhancer.enhance(cfg["lmstudio_url"], model, system, idea, images,
                                    temperature=float(cfg["llm_temperature"]), max_tokens=int(cfg["llm_max_tokens"]),
                                    on_token=th.token.emit, should_stop=lambda: th.stop_requested)

        self.final.clear()
        self.enhance_btn.setText("■  Stop")
        self.llm_status.setText(f"{model} is writing…")
        self.enhance_thread = FuncThread(run, self)
        self.enhance_thread.token.connect(self._on_token)
        self.enhance_thread.done.connect(lambda text: self._enhance_done(task, frames, text))
        self.enhance_thread.failed.connect(self._enhance_failed)
        self.enhance_thread.start()

    def _on_token(self, piece: str):
        cur = self.final.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        cur.insertText(piece)
        self.final.ensureCursorVisible()

    def _enhance_done(self, task: str, frames: int, text: str):
        self.enhance_btn.setText("✨  Enhance prompt")
        if text:
            self.final.setPlainText(build_final_prompt(task, frames, text))
            missing = [f for f in ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
                       if f not in text]
            self.llm_status.setText("⚠ Missing fields: " + ", ".join(missing) if missing else "✓ Prompt enhanced")
        else:
            self.llm_status.setText("No answer received")
        self.save_last()

    def _enhance_failed(self, err: str):
        self.enhance_btn.setText("✨  Enhance prompt")
        self.llm_status.setText("⚠ Failed")
        QMessageBox.warning(self, "LM Studio", err)

    # ------------------------------------------------------------------ generate
    def generate(self):
        if self.proc:
            return
        problems = backend.check_setup(self.cfg)
        if problems:
            QMessageBox.warning(self, "Settings", "Fix this first (Settings tab):\n\n" + "\n".join(problems))
            return
        task = self.task()
        frames = self.frames()
        body = self.final.toPlainText().strip() or self.idea.toPlainText().strip()
        if not body:
            QMessageBox.information(self, "Prompt", "Enter an idea or a prompt.")
            return
        prompt = build_final_prompt(task, frames, body)
        n = self.count.value()
        mode = self.seed_mode.currentData()
        if mode == "random":
            seeds = [random.randint(0, 2**31 - 1) for _ in range(n)]
            self.seed.setValue(seeds[-1])
        else:
            seeds = [(self.seed.value() + i) % 2**31 for i in range(n)]
        lora_dir = Path(resolve_model_path(self.cfg, self.cfg.get("lora_dir", "")))
        loras = [(str(lora_dir / c.currentText()), s.value()) for c, s in self.lora_rows if c.currentIndex() > 0]
        steps = self.steps.value()
        if self.turbo.isChecked():
            if not self.turbo_path().is_file():
                QMessageBox.warning(self, "Turbo LoRA", f"Turbo LoRA not found:\n{self.turbo_path()}")
                return
            loras = [(str(self.turbo_path()), 1.0)] + [lo for lo in loras if Path(lo[0]) != self.turbo_path()]
            steps = self.turbo_steps.currentData()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self._output_dir())
        self.jobs = []
        for i, seed in enumerate(seeds):
            name = f"{stamp}_{seed}.mp4" if n == 1 else f"{stamp}_{i + 1:02d}_{seed}.mp4"
            self.jobs.append(backend.Job(
                task=task, prompt=prompt, width=self.width_spin.value(), height=self.height_spin.value(),
                frame_count=frames, steps=steps, seed=seed, output_path=str(out_dir / name),
                first_frame=self.first_drop.path, last_frame=self.last_drop.path, loras=loras))
        self.job_meta = {"idea": self.idea.toPlainText(), "prompt": prompt, "task": task,
                         "width": self.width_spin.value(), "height": self.height_spin.value(), "frame_count": frames,
                         "steps": steps, "loras": loras, "first_frame": self.first_drop.path,
                         "last_frame": self.last_drop.path, "dit": self.cfg["dit"]}
        try:
            if loras:
                self.status.setText("Preparing LoRA…")
                self.repaint()
            backend.build_command(self.cfg, self.jobs[0])
        except (backend.BackendError, OSError) as e:
            QMessageBox.warning(self, "Generate", str(e))
            return
        if mode == "increment":
            self.seed.setValue((seeds[-1] + 1) % 2**31)
        self.save_last()
        if self.cfg.get("unload_llm_before_generate", True):
            self.status.setText("Unloading the LLM from VRAM (LM Studio)…")
            self.repaint()
            unloaded = enhancer.unload_all(self.cfg["lmstudio_url"])
            if unloaded:
                self.llm_status.setText("LLM unloaded for generation (reloads on use)")

        self.player.stop()
        self.player.setSource(QUrl())
        self.log.clear()
        self.job_index = 0
        self.stopped = False
        self.done_paths = []
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start()
        self.tick.start()
        self._start_job()

    def _start_job(self):
        """Run the current job in its own process; _finished starts the next one."""
        job = self.jobs[self.job_index]
        try:
            program, args, cwd = backend.build_command(self.cfg, job)
        except (backend.BackendError, OSError) as e:
            self.log.appendPlainText(str(e))
            self._end_queue()
            return
        n = len(self.jobs)
        header = f"=== Video {self.job_index + 1}/{n} · seed {job.seed} ===\n" if n > 1 else ""
        self.log.appendPlainText(header + "$ " + " ".join(
            [program] + [a if len(a) < 120 else a[:117] + "…" for a in args]) + "\n")
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(cwd)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self.out_buffer = ""
        self.phase = self._video_prefix() + "Loading models…"
        self.progress.setRange(0, 0)
        self._update_elapsed()
        self.proc.start(program, args)

    def _video_prefix(self) -> str:
        n = len(self.jobs)
        return f"Video {self.job_index + 1}/{n} · " if n > 1 else ""

    def _read_output(self):
        self.out_buffer += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", self.out_buffer)
        self.out_buffer = parts.pop()
        for line in parts:
            self._handle_line(line)

    def _handle_line(self, line: str):
        line = line.rstrip()
        if not line:
            return
        m = _TQDM_RE.search(line)
        if m:
            pct, step, total, timing = m.groups()
            step, total = int(step), int(total)
            n = len(self.jobs)
            self.progress.setRange(0, 100)
            self.progress.setValue(int((self.job_index * total + step) * 100 / (n * total)))
            eta = timing.split("<")[-1].split(",")[0] if "<" in timing else ""
            self.phase = f"{self._video_prefix()}Sampling step {step}/{total}" + (f" · {eta} left" if eta else "")
            self._update_elapsed()
            return
        if "INFO" in line:
            msg = line.split(":", 2)[-1].strip()
            low = msg.lower()
            if "loading" in low or "encoding" in low or "decoding" in low:
                self.phase = self._video_prefix() + msg.replace("MiniMax-H3 ", "") + "…"
                if "decoding" in low:
                    self.progress.setRange(0, 0)
                self._update_elapsed()
        # gcc/triton header noise is not useful in the log
        if not re.search(r"pyconfig\.h|_POSIX_C_SOURCE|features\.h|__triton_launcher|^\s*\||^\s*\d+ \|", line):
            self.log.appendPlainText(line)

    def _update_elapsed(self):
        if self.proc:
            s = self.timer.elapsed() // 1000
            self.status.setText(f"{self.phase}   ·   {s // 60}:{s % 60:02d} elapsed")

    def stop(self):
        if self.proc:
            self.stopped = True
            self.phase = "Stopping…"
            self.proc.kill()

    def _proc_error(self, err):
        if err == QProcess.FailedToStart:
            self.log.appendPlainText(f"Could not start process: {self.proc.errorString()}")

    def _finished(self, code: int, _status):
        if self.out_buffer:
            self._handle_line(self.out_buffer)
        self.proc.deleteLater()
        self.proc = None
        job = self.jobs[self.job_index]
        out = Path(job.output_path)
        if code == 0 and out.is_file():
            meta = dict(self.job_meta, seed=job.seed, seconds=self.timer.elapsed() // 1000)
            out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.done_paths.append(str(out))
            self.refresh_history(select=str(out))
            self.play(str(out))
        self.job_index += 1
        if not self.stopped and self.job_index < len(self.jobs):
            self._start_job()
        else:
            self._end_queue(code)

    def _end_queue(self, code: int = 1):
        self.tick.stop()
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setRange(0, 100)
        secs = self.timer.elapsed() // 1000
        t = f"{secs // 60}:{secs % 60:02d}"
        done = self.done_paths
        if done:
            self.progress.setValue(100)
            failed = len(self.jobs) - len(done)
            extra = f"  ·  {failed} failed or stopped" if failed else ""
            label = Path(done[0]).name if len(done) == 1 else f"{len(done)} videos"
            self.status.setText(f"✓ Done in {t}  ·  {label}{extra}")
        else:
            self.progress.setValue(0)
            self.status.setText(f"✕ Stopped or failed (exit code {code}) after {t}. See the log.")

    # ------------------------------------------------------------------ preview / history
    def play(self, path: str):
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    def _show_frame(self, frame):
        img = frame.toImage()
        if not img.isNull():
            self.video.setPixmap(QPixmap.fromImage(img).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _position_changed(self, pos: int):
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos)
        fmt = lambda ms: f"{ms // 60000}:{(ms // 1000) % 60:02d}"  # noqa: E731
        self.time_label.setText(f"{fmt(pos)} / {fmt(self.player.duration())}")

    def refresh_history(self, select: str = ""):
        self.history.blockSignals(True)
        self.history.clear()
        d = Path(self.cfg["output_dir"])
        files = sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:300] if d.is_dir() else []
        sel = None
        for f in files:
            meta = {}
            side = f.with_suffix(".json")
            if side.is_file():
                try:
                    meta = json.loads(side.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            when = time.strftime("%d-%m %H:%M", time.localtime(f.stat().st_mtime))
            snippet = (meta.get("idea") or meta.get("prompt") or f.name).strip().replace("\n", " ")
            item = QListWidgetItem(f"{when}  ·  {TASKS.get(meta.get('task'), '')}\n{snippet[:70]}")
            item.setData(Qt.UserRole, (str(f), meta))
            item.setToolTip(meta.get("prompt", str(f)))
            self.history.addItem(item)
            if select and os.path.samefile(str(f), select):
                sel = item
        self.history.blockSignals(False)
        if sel:
            self.history.setCurrentItem(sel)

    def _history_selected(self, item, _prev=None):
        if item and not self.proc:
            self.play(item.data(Qt.UserRole)[0])

    def _reuse_history(self):
        item = self.history.currentItem()
        if not item:
            return
        _, meta = item.data(Qt.UserRole)
        if not meta:
            return
        for b in self.mode_group.buttons():
            if b.property("task") == meta.get("task"):
                b.setChecked(True)
        self.idea.setPlainText(meta.get("idea", ""))
        self.final.setPlainText(meta.get("prompt", ""))
        self.aspect.setCurrentIndex(self.aspect.findText(ASPECT_CUSTOM))
        self.width_spin.setValue(meta.get("width", 1344))
        self.height_spin.setValue(meta.get("height", 768))
        self.duration.setValue(max(1, min(15, round(meta.get("frame_count", 124) / 24))))
        self.steps.setValue(meta.get("steps", 20))
        self.seed.setValue(meta.get("seed", 42))
        self.seed_mode.setCurrentIndex(self.seed_mode.findData("fixed"))
        self.count.setValue(1)
        self.first_drop.set_path(meta.get("first_frame", ""))
        self.last_drop.set_path(meta.get("last_frame", ""))

    def is_busy(self) -> bool:
        return self.proc is not None
