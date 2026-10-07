/**
 * E2E: "officer scores and approves" against the full docker-compose stack
 * (seeded with `make seed`). Run: E2E_BASE_URL=http://localhost:3000 npx playwright test
 */
import { expect, test } from "@playwright/test";

const PASSWORD = process.env.DEMO_PASSWORD ?? "risklens-demo";

test("officer creates, scores and approves an application with a comment", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("officer");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Loan applications" })).toBeVisible();

  // self-contained: create a fresh application through the form (other fields have defaults)
  await page.getByRole("button", { name: /New application/ }).click();
  await page.getByLabel("Full name").fill(`E2E Applicant ${Date.now()}`);
  await page.getByRole("button", { name: "Create application" }).click();
  await expect(page.getByText("Not scored yet.")).toBeVisible();

  await page.getByRole("button", { name: /Score application/ }).click();
  await expect(page.getByText("probability of default")).toBeVisible();
  await expect(page.getByText(/Top \d reasons/)).toBeVisible();

  await page.getByLabel(/Comment/).fill("E2E: income documents verified");
  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.getByText("Approved").first()).toBeVisible();
  await expect(page.getByText("Decision recorded")).toBeVisible(); // audit trail
});

test("viewer cannot decide", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("viewer");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("button", { name: /New application/ })).toHaveCount(0);
});
