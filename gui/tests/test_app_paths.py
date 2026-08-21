import tempfile
import unittest
from pathlib import Path

from gui.app_paths import locate_engine_dir


class AppPathTests(unittest.TestCase):
    def test_locates_engine_inside_a_macos_app_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "H3Studio.app" / "Contents"
            executable = bundle / "MacOS" / "H3Studio"
            resources = bundle / "Resources"
            resources.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            (resources / "h3").touch()
            (resources / "h3_shaders.metal").touch()

            result = locate_engine_dir(
                source_root=Path(directory) / "source",
                executable=executable,
                environ={},
            )

            self.assertEqual(result, resources.resolve())


if __name__ == "__main__":
    unittest.main()
