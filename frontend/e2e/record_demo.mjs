// Record the README demo video of the running app (~40 s):
//   E2E_BASE_URL=http://localhost:5173 node e2e/record_demo.mjs   -> docs/media/demo.webm
// Then: python scripts/make_demo_media.py   (webm -> mp4 + gif)
// Tip: stream traffic first (`make simulate`) so the fraud monitor is live.
import { chromium } from "@playwright/test";
import { mkdirSync, renameSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

const BASE = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const PASSWORD = process.env.DEMO_PASSWORD ?? "risklens-demo";
const OUT = fileURLToPath(new URL("../../docs/media/", import.meta.url));
const SIZE = { width: 1280, height: 800 };
mkdirSync(OUT, { recursive: true });

// Video capture has no OS cursor: draw one, plus a ripple on click and a caption bar.
const OVERLAY = () => {
  const install = () => {
    if (document.getElementById("demo-cursor")) return;
    const style = document.createElement("style");
    style.textContent = `
      #demo-cursor{position:fixed;z-index:99999;width:22px;height:22px;margin:-3px 0 0 -3px;pointer-events:none;
        background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 2l7 19 2.5-8L20 10z' fill='%23111' stroke='white' stroke-width='1.5'/%3E%3C/svg%3E") no-repeat;}
      .demo-ripple{position:fixed;z-index:99998;width:36px;height:36px;margin:-18px 0 0 -18px;border-radius:50%;
        border:3px solid #2a78d6;pointer-events:none;animation:demo-r .5s ease-out forwards}
      @keyframes demo-r{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
      #demo-caption{position:fixed;z-index:99997;left:50%;bottom:22px;transform:translateX(-50%);
        background:rgba(17,17,17,.88);color:#fff;font:600 17px system-ui,sans-serif;padding:10px 20px;
        border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.25);transition:opacity .25s;white-space:nowrap}
      #demo-caption:empty{opacity:0}`;
    document.head.appendChild(style);
    const cursor = Object.assign(document.createElement("div"), { id: "demo-cursor" });
    const caption = Object.assign(document.createElement("div"), { id: "demo-caption" });
    document.body.append(cursor, caption);
    caption.textContent = sessionStorage.getItem("demo-caption") ?? "";
    const pos = JSON.parse(sessionStorage.getItem("demo-pos") ?? "[640,400]");
    cursor.style.left = `${pos[0]}px`;
    cursor.style.top = `${pos[1]}px`;
    window.addEventListener("mousemove", (e) => {
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
      sessionStorage.setItem("demo-pos", JSON.stringify([e.clientX, e.clientY]));
    }, true);
    window.addEventListener("mousedown", (e) => {
      const r = Object.assign(document.createElement("div"), { className: "demo-ripple" });
      r.style.left = `${e.clientX}px`;
      r.style.top = `${e.clientY}px`;
      document.body.appendChild(r);
      setTimeout(() => r.remove(), 600);
    }, true);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
  else install();
};

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: SIZE,
  colorScheme: "light",
  recordVideo: { dir: OUT, size: SIZE },
});
await context.addInitScript(OVERLAY);
const page = await context.newPage();

const pause = (ms) => page.waitForTimeout(ms);
const caption = async (text) => {
  await page.evaluate((t) => {
    sessionStorage.setItem("demo-caption", t);
    const el = document.getElementById("demo-caption");
    if (el) el.textContent = t;
  }, text);
};
async function glide(locator) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 22 });
}
async function click(locator, wait = 350) {
  await glide(locator);
  await pause(180);
  await locator.click();
  await pause(wait);
}
async function type(locator, text) {
  await click(locator, 100);
  await locator.pressSequentially(text, { delay: 45 });
}
async function smoothScroll(dy, steps = 12) {
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, dy / steps);
    await pause(35);
  }
}

// 1 · sign in
await page.goto(`${BASE}/login`);
await caption("Credit officer signs in (JWT, role-based access)");
await pause(700);
await type(page.getByLabel("Username"), "officer");
await type(page.getByLabel("Password"), PASSWORD);
await click(page.getByRole("button", { name: "Sign in" }), 900);

// 2 · applications
await caption("Loan applications: real Home Credit test applicants");
await pause(1600);

// 3 · new application
await click(page.getByRole("button", { name: /New application/ }), 600);
await caption("New application – a weak external credit score");
await type(page.getByLabel("Full name"), "Sofia Martins");
const ext2 = page.getByLabel("Bureau 2");
await click(ext2, 100);
await ext2.fill("");
await ext2.pressSequentially("0.12", { delay: 60 });
await pause(400);
await click(page.getByRole("button", { name: "Create application" }), 900);

// 4 · score
await caption("Scored by the model server: calibrated PD, grade, recommendation");
await click(page.getByRole("button", { name: /Score application/ }), 300);
await page.getByText("probability of default").waitFor();
await pause(2000);

// 5 · reasons
await caption("Every decision is explained: top-5 SHAP reason codes");
await smoothScroll(430);
await pause(2600);

// 6 · override with justification
await smoothScroll(-430);
await caption("Overriding the model requires a written justification");
const approve = page.getByRole("button", { name: "Approve" });
await glide(approve);
await pause(900);
await type(page.getByLabel(/Comment/), "Verified new employment contract and payslips");
await click(approve, 1200);
await caption("Decision + override are written to the immutable audit trail");
await glide(page.getByText("Decision recorded"));
await pause(2200);

// 7 · fraud monitor
await caption("Fraud monitor: live card transactions scored in ~15 ms");
await click(page.getByRole("link", { name: "Fraud monitor" }), 300);
await pause(3800);

// 8 · models
await caption("Model registry: champion vs. challenger, fairness gate");
await click(page.getByRole("link", { name: "Models" }), 300);
await pause(2800);

// 9 · monitoring
await caption("Drift monitoring: PSI per feature vs. the training data");
await click(page.getByRole("link", { name: "Monitoring" }), 300);
await pause(2800);
await caption("");
await pause(300);

const video = page.video();
await context.close();
await browser.close();
const target = `${OUT}demo.webm`;
rmSync(target, { force: true });
renameSync(await video.path(), target);
console.log("recorded", target);
