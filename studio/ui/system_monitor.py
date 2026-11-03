"""Header status bar: live CPU / RAM / VRAM / GPU usage and a Clear memory button."""

import gc
import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from . import style

try:
    import psutil
except ImportError:  # the bar degrades gracefully when the dependency is missing
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _GPU = None

GENERATOR_MARKER = "run_h3.py"  # every generation process is started through this launcher


def _gb(n: int) -> str:
    return f"{n / 1024**3:.1f}"


class Meter(QWidget):
    """Label + value text above a thin usage bar."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(6)
        name = QLabel(label)
        name.setStyleSheet(f"color: {style.MUTED}; font-size: 11px; font-weight: 600;")
        self.value = QLabel("–")
        self.value.setStyleSheet("font-size: 11px;")
        top.addWidget(name)
        top.addStretch()
        top.addWidget(self.value)
        lay.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        lay.addWidget(self.bar)
        self.setFixedWidth(135)

    def set(self, percent: float, text: str):
        percent = max(0, min(100, int(percent)))
        self.bar.setValue(percent)
        self.value.setText(text)
        color = style.DANGER if percent >= 90 else "#f59e0b" if percent >= 75 else None
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}" if color else "")

    def unavailable(self, reason: str):
        self.value.setText("n/a")
        self.setToolTip(reason)


class SystemMonitor(QFrame):
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        lay.setSpacing(16)
        self.cpu = Meter("CPU")
        self.ram = Meter("RAM")
        self.vram = Meter("VRAM")
        self.gpu = Meter("GPU")
        for m in (self.cpu, self.ram, self.vram, self.gpu):
            lay.addWidget(m)
        self.clear_btn = QPushButton("🧹  Clear memory")
        self.clear_btn.setToolTip("Unload all LM Studio models, end leftover generator processes\n"
                                  "and free memory held by this app.")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        lay.addWidget(self.clear_btn)

        if psutil:
            psutil.cpu_percent(None)  # prime: the first call always returns 0
        else:
            for m in (self.cpu, self.ram):
                m.unavailable("Install psutil for CPU/RAM usage")
        if not _GPU:
            for m in (self.vram, self.gpu):
                m.unavailable("No NVIDIA GPU found (or nvidia-ml-py missing)")
        self.timer = QTimer(self, interval=1500, timeout=self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        if psutil:
            cpu = psutil.cpu_percent(None)
            self.cpu.set(cpu, f"{cpu:.0f}%")
            vm = psutil.virtual_memory()
            self.ram.set(vm.percent, f"{_gb(vm.total - vm.available)} / {_gb(vm.total)} GB")
        if _GPU:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
                self.vram.set(mem.used * 100 / mem.total, f"{_gb(mem.used)} / {_gb(mem.total)} GB")
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU).gpu
                self.gpu.set(util, f"{util}%")
            except pynvml.NVMLError:
                pass


def find_orphan_generators(exclude_pid: int | None = None) -> list:
    """Generator processes that are not the app's current run (e.g. left by a crash)."""
    if not psutil:
        return []
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if proc.pid not in (exclude_pid, os.getpid()) and any(GENERATOR_MARKER in part for part in cmd):
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def kill_processes(procs: list) -> int:
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5) if procs else ([], [])
    return len(procs) - len(alive)


def collect_app_garbage() -> None:
    gc.collect()
