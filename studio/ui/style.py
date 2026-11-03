"""Dark theme for the whole app."""

ACCENT = "#8b5cf6"
ACCENT_HOVER = "#a78bfa"
BG = "#0f1117"
PANEL = "#171a23"
PANEL_2 = "#1f2330"
BORDER = "#2a2f3d"
TEXT = "#e6e8ef"
MUTED = "#8a90a2"
DANGER = "#ef4444"
OK = "#22c55e"

STYLESHEET = f"""
* {{ font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

QFrame#card {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
}}
QLabel#cardTitle {{ font-size: 12px; font-weight: 600; color: {MUTED}; letter-spacing: 1px; }}
QLabel#appTitle {{ font-size: 20px; font-weight: 700; }}
QLabel#appSubtitle {{ color: {MUTED}; }}
QLabel#hint {{ color: {MUTED}; font-size: 12px; }}
QLabel#status {{ color: {MUTED}; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 10px 18px; margin-right: 4px;
    border: none; border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {MUTED}; background: {PANEL}; border: 1px dashed {BORDER};
}}
QComboBox QAbstractItemView {{
    background: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 7px 14px; font-weight: 600;
}}
QPushButton:hover {{ border: 1px solid {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {BORDER}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #db2777);
    border: none; padding: 11px 18px; font-size: 14px;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #ec4899);
}}
QPushButton#primary:disabled {{ background: {PANEL_2}; color: {MUTED}; }}
QPushButton#accent {{ background: {ACCENT}; border: none; }}
QPushButton#accent:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#danger {{ background: transparent; border: 1px solid {DANGER}; color: {DANGER}; }}
QPushButton#mode {{ padding: 9px 10px; border-radius: 8px; }}
QPushButton#mode:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}
QToolButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 5px 8px; }}
QToolButton:hover {{ border: 1px solid {ACCENT}; }}

QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {BORDER}; background: {PANEL_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}

QProgressBar {{
    background: {PANEL_2}; border: none; border-radius: 6px; height: 12px; text-align: center; font-size: 11px;
}}
QProgressBar::chunk {{
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #db2777);
}}
QListWidget {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; outline: none; }}
QListWidget::item {{ padding: 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {ACCENT}; }}
QListWidget::item:hover:!selected {{ background: {BORDER}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: transparent; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
"""
