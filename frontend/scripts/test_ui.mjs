import { chromium } from 'playwright-core';
import fs from 'fs';
import path from 'path';

async function run() {
  console.log('Launching browser with playwright-core using msedge channel...');
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });

  const page = await context.newPage();
  console.log('Navigating to http://localhost:5173/ ...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });

  // Wait for previous results or status to settle
  await page.waitForTimeout(2000);

  // 1. Verify Diagnostic Cockpit metrics in sidebar
  const metricsText = await page.textContent('#metrics');
  console.log('--- DIAGNOSTIC COCKPIT METRICS ---');
  console.log(metricsText);

  const cards = await page.$$eval('#metrics .card', (cards) =>
    cards.map((c) => ({
      key: c.querySelector('.k')?.textContent?.trim(),
      val: c.querySelector('.v')?.textContent?.trim(),
      sub: c.querySelector('.b')?.textContent?.trim(),
    }))
  );
  console.log('Extracted metric cards:', JSON.stringify(cards, null, 2));

  // 2. Test Virtual Ground Monitor Probe: Click on right map
  console.log('Clicking on #mapRight canvas to trigger Virtual Ground Monitor Probe...');
  const mapRight = await page.$('#mapRight');
  const box = await mapRight.boundingBox();
  if (box) {
    // Click near the center of the right map
    const clickX = box.x + box.width / 2;
    const clickY = box.y + box.height / 2;
    console.log(`Clicking at coordinates (${clickX}, ${clickY})...`);
    await page.mouse.click(clickX, clickY);
  } else {
    throw new Error('#mapRight bounding box not found');
  }

  // Wait for probe panel to open and load data
  await page.waitForSelector('#probePanel:not(.hidden)', { timeout: 8000 });
  await page.waitForTimeout(1500);

  const probeCoords = await page.textContent('#probeCoords');
  const probeCurrent = await page.textContent('#probeCurrent');
  const probeMean = await page.textContent('#probeMean');
  const probeVcd = await page.textContent('#probeVcd');
  const probeStation = await page.textContent('#probeStation');

  console.log('--- VIRTUAL GROUND MONITOR PROBE RESULT ---');
  console.log('Coordinates:', probeCoords);
  console.log('Current Surface Concentration:', probeCurrent);
  console.log('Period Mean:', probeMean);
  console.log('Inversion VCD:', probeVcd);
  console.log('Nearest Station:', probeStation);

  // 3. Capture high-res screenshot artifact
  const outPath = 'C:\\Users\\rachi\\.gemini\\antigravity\\brain\\73878cd4-ecc3-4c77-b6ea-7743cafd6864\\working_ui.png';
  await page.screenshot({ path: outPath, fullPage: false });
  console.log(`Screenshot saved to: ${outPath}`);

  await browser.close();
  console.log('SUCCESS: All checks passed!');
}

run().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
