"""Top-level window: header + Studio / System prompts / Settings tabs."""

from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from .prompts_tab import PromptsTab
from .settings_tab import SettingsTab
from .studio_tab import StudioTab


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
        subtitle = QLabel("Native MiniMax-H3 video + audio  ·  prompt enhancement via LM Studio", objectName="appSubtitle")
        hl.addWidget(title)
        hl.addSpacing(14)
        hl.addWidget(subtitle)
        hl.addStretch()
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

    def closeEvent(self, e):
        if self.studio.is_busy():
            if QMessageBox.question(self, "Busy", "A generation is running. Stop it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.studio.stop()
        self.studio.save_last()
        e.accept()
