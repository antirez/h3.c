from __future__ import annotations

import sys
from pathlib import Path
from subprocess import SubprocessError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from gui.app_paths import locate_engine_dir
from gui.hardware import MacInfo, detect_mac_info
from gui.runner import H3Runner
from gui.window import MainWindow


def _mac_info() -> MacInfo:
    try:
        return detect_mac_info()
    except (OSError, ValueError, SubprocessError):
        return MacInfo(
            chip="Mac not detected",
            memory_gib=0.0,
            architecture="—",
            metal_support="Not detected",
        )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    smoke_test = "--smoke-test" in arguments
    qt_arguments = [sys.argv[0], *(item for item in arguments if item != "--smoke-test")]
    source_root = Path(__file__).resolve().parents[1]
    engine_root = locate_engine_dir(source_root=source_root)
    packaged = engine_root != source_root.resolve()
    application = QApplication(qt_arguments)
    application.setApplicationName("H3 Studio")
    application.setOrganizationName("h3-metal")
    window = MainWindow(
        repo_root=engine_root,
        mac_info=_mac_info(),
        runner=H3Runner(engine_root / "h3"),
        default_output_dir=(
            Path.home() / "Movies" / "H3 Studio" if packaged else None
        ),
    )
    window.show()
    if smoke_test:
        QTimer.singleShot(100, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
