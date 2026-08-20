import subprocess
import unittest
from pathlib import Path


class CliContractTests(unittest.TestCase):
    def test_cli_exposes_machine_readable_preview_directory(self) -> None:
        executable = Path(__file__).resolve().parents[2] / "h3"
        if not executable.exists():
            self.skipTest("h3 executable has not been built")

        completed = subprocess.run(
            (str(executable), "--help"),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--preview-dir PATH", completed.stderr)


if __name__ == "__main__":
    unittest.main()
