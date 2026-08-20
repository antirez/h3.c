import unittest

from gui.hardware import MacInfo, detect_mac_info
from gui.presets import preset_for, recommended_preset_name


class PresetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.m4_pro = MacInfo(
            chip="Apple M4 Pro",
            memory_gib=48.0,
            architecture="arm64",
            metal_support="Metal supported",
        )

    def test_m4_pro_48_gib_defaults_to_fast_identity_preview(self) -> None:
        self.assertEqual(recommended_preset_name(self.m4_pro), "fast")

        preset = preset_for("fast", self.m4_pro)

        self.assertEqual((preset.width, preset.height), (512, 512))
        self.assertEqual((preset.render_width, preset.render_height), (320, 320))
        self.assertEqual(preset.seconds, 2)
        self.assertEqual(preset.steps, 6)
        self.assertEqual(preset.layers, 40)
        self.assertEqual(preset.reuse, 1)
        self.assertTrue(preset.ssd_streaming)
        self.assertFalse(preset.live_preview)

    def test_live_preview_is_never_enabled_automatically(self) -> None:
        large_mac = MacInfo(
            chip="Apple M5 Max",
            memory_gib=128.0,
            architecture="arm64",
            metal_support="Metal 4",
        )

        for name in ("fast", "balanced", "quality"):
            self.assertFalse(preset_for(name, large_mac).live_preview)

        self.assertEqual(recommended_preset_name(large_mac), "balanced")

    def test_low_memory_mac_gets_a_smaller_fast_canvas(self) -> None:
        small_mac = MacInfo(
            chip="Apple M3",
            memory_gib=24.0,
            architecture="arm64",
            metal_support="Metal supported",
        )

        preset = preset_for("fast", small_mac)

        self.assertEqual((preset.width, preset.height), (256, 256))
        self.assertEqual((preset.render_width, preset.render_height), (256, 256))
        self.assertEqual(preset.steps, 4)

    def test_detects_mac_characteristics_from_system_commands(self) -> None:
        outputs = {
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4 Pro\n",
            ("sysctl", "-n", "hw.memsize"): "51539607552\n",
            ("uname", "-m"): "arm64\n",
            ("system_profiler", "SPDisplaysDataType"): (
                "Chipset Model: Apple M4 Pro\nMetal: Supported\n"
            ),
        }

        info = detect_mac_info(lambda command: outputs[tuple(command)])

        self.assertEqual(info, self.m4_pro)


if __name__ == "__main__":
    unittest.main()
