"""MiniMax H3 Studio — start the GUI."""

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from studio.config import load_config
from studio.ui.main_window import MainWindow
from studio.ui.style import STYLESHEET


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Ultra AI Generator")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(QIcon.fromTheme("video-x-generic"))
    window = MainWindow(load_config())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
