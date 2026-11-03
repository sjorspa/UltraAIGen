"""Small reusable widgets."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QToolButton, QVBoxLayout, QWidget,
)

from . import style

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(8)
    if title:
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    return frame, lay


class ImageDrop(QFrame):
    """Click-or-drop image slot with a thumbnail and a clear button."""

    changed = Signal(str)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.path = ""
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(150)
        self._idle_style = f"QFrame {{ border: 2px dashed {style.BORDER}; border-radius: 10px; background: {style.PANEL_2}; }}"
        self._hover_style = f"QFrame {{ border: 2px dashed {style.ACCENT}; border-radius: 10px; background: {style.PANEL_2}; }}"
        self.setStyleSheet(self._idle_style)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setStyleSheet(f"border: none; color: {style.MUTED}; font-weight: 600;")
        self.clear_btn = QToolButton(text="✕")
        self.clear_btn.setToolTip("Remove")
        self.clear_btn.setStyleSheet("border: none;")
        self.clear_btn.clicked.connect(lambda: self.set_path(""))
        self.clear_btn.hide()
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.clear_btn)
        lay.addLayout(top)
        self.preview = QLabel("Drop an image here\nor click to browse")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(f"border: none; color: {style.MUTED};")
        lay.addWidget(self.preview, 1)

    def set_path(self, path: str):
        self.path = path if path and Path(path).is_file() else ""
        if self.path:
            pix = QPixmap(self.path)
            self.preview.setPixmap(pix.scaled(self.preview.width() or 240, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview.setToolTip(self.path)
            self.clear_btn.show()
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Drop an image here\nor click to browse")
            self.clear_btn.hide()
        self.changed.emit(self.path)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(self, "Choose image", str(Path(self.path).parent) if self.path else "", IMAGE_FILTER)
            if path:
                self.set_path(path)

    def _drop_path(self, e) -> str:
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in IMAGE_EXTS:
                return p
        return ""

    def dragEnterEvent(self, e):
        if self._drop_path(e):
            self.setStyleSheet(self._hover_style)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self._idle_style)

    def dropEvent(self, e):
        self.setStyleSheet(self._idle_style)
        p = self._drop_path(e)
        if p:
            self.set_path(p)


class PathField(QWidget):
    """Line edit + browse button for a file or directory."""

    def __init__(self, mode: str = "file", file_filter: str = "", base_dir_fn=None, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter
        self.base_dir_fn = base_dir_fn
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        btn = QToolButton(text="…")
        btn.setToolTip("Browse")
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str):
        self.edit.setText(value)

    def _browse(self):
        start = self.text()
        base = self.base_dir_fn() if self.base_dir_fn else ""
        if start and not Path(start).is_absolute() and base:
            start = str(Path(base) / start)
        start = start or base
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", start, self.file_filter)
        if not path:
            return
        # store paths inside models_dir relative to it, so the config moves with the folder
        if base and self.mode == "file":
            try:
                path = str(Path(path).relative_to(base))
            except ValueError:
                pass
        self.edit.setText(path)
