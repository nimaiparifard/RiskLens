// Capture README / review screenshots of the running app:
//   E2E_BASE_URL=http://localhost:3000 node e2e/screenshots.mjs  -> docs/screenshots/*.png
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const BASE = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const PASSWORD = process.env.DEMO_PASSWORD ?? "risklens-demo";
const OUT = fileURLToPath(new URL("../../docs/screenshots/", import.meta.url));
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
for (const theme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: theme });
  await page.goto(`${BASE}/login`);
  await page.getByLabel("Username").fill("officer");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("heading", { name: "Loan applications" }).waitFor();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${OUT}applications-${theme}.png`, fullPage: true });

  await page.getByLabel("Status").selectOption("scored");
  await page.waitForTimeout(800);
  await page.locator("tbody a").first().click();
  await page.getByText("probability of default").waitFor();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${OUT}application-detail-${theme}.png`, fullPage: true });

  for (const [path, name] of [["/fraud", "fraud"], ["/models", "models"], ["/monitoring", "monitoring"]]) {
    await page.goto(`${BASE}${path}`);
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${OUT}${name}-${theme}.png`, fullPage: true });
  }
  await page.close();
}
await browser.close();
console.log("screenshots written to", OUT);
