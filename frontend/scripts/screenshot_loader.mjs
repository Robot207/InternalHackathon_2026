import { chromium } from 'playwright-core';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // Unhide tech loader overlay manually to inspect
  await page.evaluate(() => {
    const loader = document.getElementById('techLoader');
    loader.classList.remove('hidden');
    document.getElementById('techLoaderText').textContent = '☁️ Imputing Gaps under Cloudy Conditions...';
  });

  await page.waitForTimeout(300);

  // Screenshot the loader card specifically
  const loaderCard = page.locator('.tech-loader-box');
  await loaderCard.screenshot({ path: path.join(__dirname, '../../docs/centered_loader.png') });
  console.log('Saved docs/centered_loader.png');

  await browser.close();
}

main().catch(console.error);
