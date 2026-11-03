"""Top-level window: header + Studio / System prompts / Settings tabs."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from .. import enhancer
from . import system_monitor
from .prompts_tab import PromptsTab
from .settings_tab import SettingsTab
from .studio_tab import FuncThread, StudioTab


class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Ultra AI Generator")
        self.resize(1440, 920)

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 4)
        title = QLabel("◆ Ultra AI Generator", objectName="appTitle")
        subtitle = QLabel("MiniMax-H3 video + audio  ·  LM Studio", objectName="appSubtitle")
        hl.addWidget(title)
        hl.addSpacing(14)
        hl.addWidget(subtitle)
        hl.addStretch()
        self.monitor = system_monitor.SystemMonitor()
        self.monitor.clear_requested.connect(self.clear_memory)
        hl.addWidget(self.monitor)
        lay.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.studio = StudioTab(cfg)
        self.prompts = PromptsTab()
        self.settings = SettingsTab(cfg)
        self.tabs.addTab(self.studio, "Studio")
        self.tabs.addTab(self.prompts, "System prompts")
        self.tabs.addTab(self.settings, "Settings")
        lay.addWidget(self.tabs, 1)
        self.studio.apply_pending_loras()

        self.prompts.changed.connect(self.studio.refresh_presets)
        self.settings.saved.connect(self._settings_saved)

    def _settings_saved(self):
        self.studio.refresh_loras()
        self.studio.refresh_history()
        self.studio.refresh_models()
        self.statusBar().showMessage("Settings saved", 3000)

    def clear_memory(self):
        """Free VRAM/RAM: stop a running generation (after asking), end leftover
        generator processes, unload LM Studio models and collect the GUI's garbage."""
        studio = self.studio
        if studio.is_busy():
            if QMessageBox.question(self, "Clear memory",
                                    "A generation is running. Stop it and clear memory?") != QMessageBox.Yes:
                return
            studio.stopped = True
            studio.engine.kill()
        # quitting the engine frees every model it keeps loaded; it restarts on the next generation
        engine_was_running = studio.engine.is_running()
        studio.shutdown_engine()
        orphans = system_monitor.find_orphan_generators()
        ended = 0
        if orphans:
            pids = ", ".join(str(p.pid) for p in orphans)
            if QMessageBox.question(self, "Clear memory",
                                    f"Found {len(orphans)} leftover generator process(es) (PID {pids}) "
                                    "that may still hold GPU memory. End them?") == QMessageBox.Yes:
                ended = system_monitor.kill_processes(orphans)
        studio.player.stop()
        system_monitor.collect_app_garbage()

        self.monitor.clear_btn.setEnabled(False)
        self.monitor.clear_btn.setText("Clearing…")
        url = self.cfg["lmstudio_url"]
        self._clear_thread = FuncThread(lambda _t: enhancer.unload_all(url), self)
        self._clear_thread.done.connect(lambda unloaded: self._memory_cleared(unloaded, ended, engine_was_running))
        self._clear_thread.failed.connect(lambda _e: self._memory_cleared([], ended, engine_was_running))
        self._clear_thread.start()

    def _memory_cleared(self, unloaded: list, ended: int, engine_stopped: bool):
        self.monitor.clear_btn.setEnabled(True)
        self.monitor.clear_btn.setText("🧹  Clear memory")
        parts = ["unloaded the H3 models"] if engine_stopped else []
        if unloaded:
            parts.append(f"unloaded {len(unloaded)} LM Studio model(s): {', '.join(unloaded)}")
        if ended:
            parts.append(f"ended {ended} generator process(es)")
        self.statusBar().showMessage("Memory cleared — " + ("; ".join(parts) if parts else "nothing else was holding memory"), 8000)
        if unloaded:
            self.studio.refresh_models()  # the ● loaded markers changed
        QTimer.singleShot(500, self.monitor.refresh)

    def closeEvent(self, e):
        if self.studio.is_busy():
            if QMessageBox.question(self, "Busy", "A generation is running. Stop it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.studio.stopped = True
        self.studio.shutdown_engine()
        self.studio.save_last()
        e.accept()
