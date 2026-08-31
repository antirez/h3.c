#!/usr/bin/env node
// snapshot_ui.mjs — screenshot multi-viewport + cattura errori console.
// Uso: node scripts/snapshot_ui.mjs [url] [nome]
// Prerequisito: npm i -D playwright && npx playwright install chromium
// Esito: ultima riga "UI: PASS" oppure "UI: FAIL" + exit code.
//
// Lo screenshot cattura rotture invisibili nei log (layout, overflow, contrasto).
// Gli errori console catturano rotture invisibili nello screenshot.
// Servono entrambi: nessuno dei due da solo è una verifica sufficiente.

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const [url = "http://localhost:5173", name = "snapshot"] = process.argv.slice(2);
const OUT = "logs/agent";
mkdirSync(OUT, { recursive: true });

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  mobile: { width: 390, height: 844 },
};

const problems = [];
const browser = await chromium.launch();

for (const [label, viewport] of Object.entries(VIEWPORTS)) {
  const page = await browser.newPage({ viewport });
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`[${label}] CONSOLE ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`[${label}] PAGEERROR ${e.message}`));
  page.on("response", (r) => {
    if (r.status() >= 400) problems.push(`[${label}] HTTP ${r.status()} ${r.url()}`);
  });

  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30_000 });
    await page.screenshot({ path: `${OUT}/${name}_${label}.png`, fullPage: true });
    console.log(`  screenshot: ${OUT}/${name}_${label}.png`);
  } catch (e) {
    problems.push(`[${label}] NAVIGATION ${e.message}`);
  }
  await page.close();
}

await browser.close();

if (problems.length) {
  console.log(problems.slice(0, 15).join("\n"));
  console.log("UI: FAIL");
  process.exit(1);
}
console.log("UI: PASS");
