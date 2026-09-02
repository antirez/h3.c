import subprocess
import sys
import unittest
from pathlib import Path

from tools.search_reuse_schedule import candidate_schedules, exact_schedule


class ReuseScheduleTests(unittest.TestCase):
    def test_exact_schedule_has_requested_count_and_endpoints(self):
        for steps in (4, 20, 50):
            for count in (2, min(steps, 4), steps):
                schedule = exact_schedule(steps, count, lambda value: value ** 1.8)
                self.assertEqual(len(schedule), count)
                self.assertEqual(schedule[0], 0)
                self.assertEqual(schedule[-1], steps - 1)
                self.assertEqual(tuple(sorted(set(schedule))), schedule)

    def test_candidates_are_unique_and_valid(self):
        candidates = candidate_schedules(20, [8, 9, 11], [0.55, 1.0, 1.8])
        schedules = [schedule for _, schedule in candidates]
        self.assertEqual(len(schedules), len(set(schedules)))
        for _, schedule in candidates:
            self.assertEqual(schedule[0], 0)
            self.assertEqual(schedule[-1], 19)

    def test_dry_run_needs_no_model(self):
        root = Path(__file__).parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tools/search_reuse_schedule.py"),
                "--model-dir", str(root / "missing-model"),
                "--prompt", "a person walks",
                "--steps", "20",
                "--counts", "8",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reference: full", result.stdout)
        self.assertIn("n8:", result.stdout)


if __name__ == "__main__":
    unittest.main()
