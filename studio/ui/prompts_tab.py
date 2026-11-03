"""System-prompt presets editor."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QWidget,
)

from ..presets import DEFAULT_PRESETS, TASKS, load_presets, save_presets
from .widgets import card


class PromptsTab(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.presets = load_presets()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 20)
        split = QSplitter()
        lay.addWidget(split)

        left, ll = card("Presets")
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        ll.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("New", self._new), ("Duplicate", self._duplicate), ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        ll.addLayout(row)
        reset = QPushButton("Restore default presets")
        reset.clicked.connect(self._reset)
        ll.addWidget(reset)
        split.addWidget(left)

        right, rl = card("System prompt")
        f = QFormLayout()
        self.name = QLineEdit()
        self.task = QComboBox()
        for key, label in TASKS.items():
            self.task.addItem(label, key)
        f.addRow("Name", self.name)
        f.addRow("Mode", self.task)
        rl.addLayout(f)
        hint = QLabel("Variables: {duration} {frame_count} {width} {height} {orientation}. "
                      "The LLM writes the three fields; the app prepends the correct keyframe instruction line itself.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        rl.addWidget(hint)
        self.text = QPlainTextEdit()
        mono = QFont("Consolas" if QFont("Consolas").exactMatch() else "Monospace")
        mono.setStyleHint(QFont.Monospace)
        self.text.setFont(mono)
        rl.addWidget(self.text, 1)
        save = QPushButton("Save")
        save.setObjectName("accent")
        save.clicked.connect(self.save)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(save)
        rl.addLayout(row)
        split.addWidget(right)
        split.setSizes([280, 900])
        self._refresh(0)

    def _refresh(self, row: int):
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.presets:
            task = TASKS.get(p.get("task"), "?")
            self.list.addItem(p["name"] if task in p["name"] else f"{p['name']}  ·  {task}")
        self.list.blockSignals(False)
        self.list.setCurrentRow(max(0, min(row, len(self.presets) - 1)))
        self._select(self.list.currentRow())

    def _select(self, row: int):
        if 0 <= row < len(self.presets):
            p = self.presets[row]
            self.name.setText(p["name"])
            self.task.setCurrentIndex(max(0, self.task.findData(p.get("task", "t2va"))))
            self.text.setPlainText(p["text"])

    def _store_current(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.presets):
            self.presets[row] = {"name": self.name.text().strip() or "Untitled",
                                 "task": self.task.currentData(), "text": self.text.toPlainText()}

    def _new(self):
        self._store_current()
        self.presets.append({"name": "New preset", "task": "t2va", "text": DEFAULT_PRESETS[0]["text"]})
        self._refresh(len(self.presets) - 1)

    def _duplicate(self):
        self._store_current()
        row = self.list.currentRow()
        if row >= 0:
            p = dict(self.presets[row])
            p["name"] += " (copy)"
            self.presets.insert(row + 1, p)
            self._refresh(row + 1)

    def _delete(self):
        row = self.list.currentRow()
        if row >= 0 and len(self.presets) > 1:
            del self.presets[row]
            self._refresh(row)

    def _reset(self):
        if QMessageBox.question(self, "Restore", "Replace all presets with the default presets?") == QMessageBox.Yes:
            self.presets = [dict(p) for p in DEFAULT_PRESETS]
            self._refresh(0)
            self.save()

    def save(self):
        self._store_current()
        save_presets(self.presets)
        self._refresh(self.list.currentRow())
        self.changed.emit()
