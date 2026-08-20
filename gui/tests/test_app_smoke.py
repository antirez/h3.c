import os
import subprocess
import sys
import unittest


class AppSmokeTests(unittest.TestCase):
    def test_module_can_open_and_close_without_user_input(self) -> None:
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"

        completed = subprocess.run(
            (sys.executable, "-m", "gui.main", "--smoke-test"),
            capture_output=True,
            text=True,
            env=environment,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
