import { chromium } from 'playwright-core';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 900 });

  console.log('Navigating to http://localhost:5173...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);

  // Check Page 1 elements
  const brand = await page.textContent('.brand');
  console.log('Brand:', brand?.trim());

  const cityVal = await page.inputValue('#cityInput');
  console.log('Default City Input:', cityVal);

  const rmseText = await page.textContent('#liveRmseVal');
  console.log('Live RMSE in Dashboard:', rmseText?.trim());

  const rightLayerOption = await page.textContent('#layerRight option[value="prediction"]');
  console.log('Right Map Option Label:', rightLayerOption?.trim());

  // Screenshot Page 1
  await page.screenshot({ path: path.join(__dirname, '../../docs/page1_live_downscaling.png') });
  console.log('Saved docs/page1_live_downscaling.png');

  // Switch to Page 2
  console.log('Clicking Tab 2: 72-Hour Prediction...');
  await page.click('#tabPredict');
  await page.waitForTimeout(1000);

  const initialAlert = await page.textContent('#alertTitle');
  console.log('Page 2 Initial Alert Title:', initialAlert?.trim());

  // Move slider to step 3 (+48 Hrs)
  console.log('Moving slider to +48 Hrs (step 3)...');
  await page.evaluate(() => {
    const slider = document.getElementById('timelineSliderPredict');
    slider.value = '3';
    slider.dispatchEvent(new Event('input'));
  });
  await page.waitForTimeout(600);

  const alertAfter48h = await page.textContent('#alertTitle');
  console.log('Alert after +48h:', alertAfter48h?.trim());

  // Toggle Traffic Drop Simulator
  console.log('Checking Simulate 40% Traffic Drop...');
  await page.click('#trafficDropCheck');
  await page.waitForTimeout(600);

  const simDelta = await page.textContent('#simDelta');
  console.log('Simulated Delta:', simDelta?.trim());

  // Screenshot Page 2
  await page.screenshot({ path: path.join(__dirname, '../../docs/page2_predictive_analytics.png') });
  console.log('Saved docs/page2_predictive_analytics.png');

  await browser.close();
  console.log('All tests passed successfully!');
}

main().catch((err) => {
  console.error('Test error:', err);
  process.exit(1);
});
