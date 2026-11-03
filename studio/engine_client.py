"""GUI side of the engine: starts studio/engine/worker.py once and talks JSON lines to it."""

import json

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .config import APP_DIR

WORKER = APP_DIR / "studio" / "engine" / "worker.py"
_PREFIX = "@@H3 "


class EngineClient(QObject):
    ready = Signal(dict)
    status = Signal(str)
    progress = Signal(int, int)
    done = Signal(str, str, int)  # job id, output path, seconds
    failed = Signal(str, str)  # job id, message
    cancelled = Signal(str)
    released = Signal()
    log = Signal(str)
    exited = Signal(int)  # the engine process ended (crash, quit or kill)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: QProcess | None = None
        self._buffer = ""

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def pid(self) -> int | None:
        return self.proc.processId() if self.is_running() else None

    def start(self, python: str) -> None:
        if self.is_running():
            return
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(APP_DIR))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)
        self._buffer = ""
        # -I: ignore PYTHONPATH and user site-packages, so only the engine environment is used
        self.proc.start(python, ["-I", str(WORKER)])

    def send(self, message: dict) -> None:
        if self.is_running():
            self.proc.write((json.dumps(message) + "\n").encode("utf-8"))

    def generate(self, job_id: str, config: dict, job: dict) -> None:
        self.send({"cmd": "generate", "id": job_id, "config": config, "job": job})

    def cancel(self) -> None:
        self.send({"cmd": "cancel"})

    def release(self) -> None:
        self.send({"cmd": "release"})

    def shutdown(self, wait_ms: int = 5000) -> None:
        """Ask the engine to quit (frees all GPU memory); kill it if it does not."""
        if not self.is_running():
            return
        self.send({"cmd": "quit"})
        self.proc.closeWriteChannel()
        if not self.proc.waitForFinished(wait_ms):
            self.proc.kill()
            self.proc.waitForFinished(2000)

    def kill(self) -> None:
        if self.is_running():
            self.proc.kill()

    def _read(self) -> None:
        self._buffer += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._handle(line.rstrip("\r"))

    def _handle(self, line: str) -> None:
        if not line.startswith(_PREFIX):
            if line.strip():
                self.log.emit(line)
            return
        try:
            ev = json.loads(line[len(_PREFIX):])
        except json.JSONDecodeError:
            self.log.emit(line)
            return
        kind = ev.get("event")
        if kind == "ready":
            self.ready.emit(ev)
        elif kind == "status":
            self.status.emit(ev.get("msg", ""))
        elif kind == "progress":
            self.progress.emit(int(ev["step"]), int(ev["total"]))
        elif kind == "done":
            self.done.emit(ev.get("id", ""), ev["path"], int(ev.get("seconds", 0)))
        elif kind == "error":
            self.failed.emit(ev.get("id", ""), ev.get("msg", "unknown error"))
        elif kind == "cancelled":
            self.cancelled.emit(ev.get("id", ""))
        elif kind == "released":
            self.released.emit()

    def _error(self, err) -> None:
        if err == QProcess.FailedToStart:
            self.log.emit(f"Could not start the engine: {self.proc.errorString()}")
            self.exited.emit(-1)

    def _finished(self, code: int, _status) -> None:
        if self._buffer:
            self._handle(self._buffer)
            self._buffer = ""
        self.proc.deleteLater()
        self.proc = None
        self.exited.emit(code)
