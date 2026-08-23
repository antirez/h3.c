#!/usr/bin/env python3
"""Browser regression for runtime LoRA catalog and ordered selection."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright


with sync_playwright() as playwright:
    chrome = os.environ.get("PLAYWRIGHT_CHROME")
    if not chrome:
        system_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        chrome = str(system_chrome) if system_chrome.is_file() else None
    browser = playwright.chromium.launch(headless=True, executable_path=chrome)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text)
            if message.type == "error" else None)
    page.goto("http://127.0.0.1:18081", wait_until="networkidle")
    page.wait_for_function(
        "document.querySelectorAll('#loraCatalogSelect option').length >= 4"
    )
    options = page.locator("#loraCatalogSelect option").all_text_contents()
    assert any("HMNSFW_AIO_V2" in option for option in options)
    assert any("riding_pose_H3_i2v_v1.0" in option for option in options)

    select = page.locator("#loraCatalogSelect")
    first_id = select.locator("option").nth(1).get_attribute("value")
    second_id = select.locator("option").nth(2).get_attribute("value")
    select.select_option(first_id)
    page.locator("#loraAdd").click()
    select.select_option(second_id)
    page.locator("#loraAdd").click()
    assert page.locator(".lora-item").count() == 2

    page.locator(".lora-item").nth(0).locator(".lora-scale").fill("0.65")
    page.locator(".lora-item").nth(0).locator(".lora-scale").press("Tab")
    second_name = page.locator(".lora-item").nth(1).locator("strong").inner_text()
    page.locator(".lora-item").nth(1).locator('[data-action="up"]').click()
    assert page.locator(".lora-item").nth(0).locator("strong").inner_text() == second_name
    command = page.locator("#commandPreview").inner_text()
    assert command.count("--lora ") == 2
    assert "--lora-scale 0.65" in command
    assert page.locator("#ssdStreaming").is_disabled()

    page.locator('[data-mode="ref-image"]').click()
    assert page.locator(".lora-scope").nth(0).input_value() == "ref2va"
    page.screenshot(path="/tmp/h3-lora-ui.png", full_page=True)
    browser.close()
    assert not console_errors, console_errors
    print("ok: runtime LoRA catalog, order, scale, scope and SSD conflict UI")
