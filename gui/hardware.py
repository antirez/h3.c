from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from collections.abc import Callable, Sequence


CommandRunner = Callable[[Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class MacInfo:
    chip: str
    memory_gib: float
    architecture: str
    metal_support: str

    @property
    def summary(self) -> str:
        return f"{self.chip} · {self.memory_gib:.0f} GB memoria unificata"


def _run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.stdout


def detect_mac_info(run: CommandRunner = _run_command) -> MacInfo:
    chip = run(("sysctl", "-n", "machdep.cpu.brand_string")).strip()
    memory_bytes = int(run(("sysctl", "-n", "hw.memsize")).strip())
    architecture = run(("uname", "-m")).strip()
    display_info = run(("system_profiler", "SPDisplaysDataType"))
    metal_match = re.search(r"Metal Support:\s*(.+)", display_info)
    metal_support = metal_match.group(1).strip() if metal_match else "Non rilevato"
    return MacInfo(
        chip=chip,
        memory_gib=memory_bytes / (1024**3),
        architecture=architecture,
        metal_support=metal_support,
    )
