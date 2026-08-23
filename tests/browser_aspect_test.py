#!/usr/bin/env python3
"""Browser regression for source-image aspect matching."""

import binascii
import os
from pathlib import Path
import struct
import zlib

from playwright.sync_api import sync_playwright


def png_bytes(width, height):
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes((80, 140, 210)) * width
    pixels = zlib.compress(row * height)
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


with sync_playwright() as playwright:
    chrome = os.environ.get("PLAYWRIGHT_CHROME")
    if not chrome:
        system_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        chrome = str(system_chrome) if system_chrome.is_file() else None
    browser = playwright.chromium.launch(headless=True, executable_path=chrome)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text)
            if message.type == "error" else None)
    page.goto("http://127.0.0.1:18080", wait_until="networkidle")
    page.locator('[data-mode="frame-anchor"]').click()
    page.locator("#sourceAspectSync").wait_for(state="visible")
    assert page.locator("#matchInputAspect").is_checked()
    with page.expect_file_chooser() as chooser:
        page.locator('.file-add-button[data-key="first_frame"]').click()
    chooser.value.set_files({
        "name": "portrait-3x4.png",
        "mimeType": "image/png",
        "buffer": png_bytes(300, 400),
    })
    page.wait_for_function(
        "document.querySelector('#sourceAspectStatus').textContent.includes('3:4')"
    )
    width = int(page.locator("#width").input_value())
    height = int(page.locator("#height").input_value())
    assert width % 32 == 0 and height % 32 == 0
    assert abs(width / height - 0.75) < 0.001
    assert f"--width {width} --height {height}" in page.locator("#commandPreview").inner_text()
    assert "人物比例不拉伸" in page.locator("#sourceAspectStatus").inner_text()
    page.locator('.aspect-option[data-ratio="16:9"]').click()
    assert not page.locator("#matchInputAspect").is_checked()
    assert int(page.locator("#width").input_value()) * 9 == \
        int(page.locator("#height").input_value()) * 16
    page.screenshot(path="/tmp/h3-aspect-ui.png", full_page=True)
    browser.close()
    assert not console_errors, console_errors
    print(f"ok: source 3:4 -> legal {width}x{height}; manual override works")
