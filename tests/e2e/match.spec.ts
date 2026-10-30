import { expect, test } from "@playwright/test";

const MATCH_TIMEOUT_MS = 60_000;

test("the live match plays every game and reports the score", async ({ page }) => {
  test.setTimeout(MATCH_TIMEOUT_MS + 10_000);
  await page.goto("/demo/match.html?games=2&time=10&parallel=2");
  await expect(page.locator("body")).toHaveAttribute("data-state", "idle");

  await page.locator("#start").click();

  await expect(page.locator("#boards figure")).toHaveCount(2);
  await expect(page.locator("body")).toHaveAttribute("data-state", "done", {
    timeout: MATCH_TIMEOUT_MS,
  });
  await expect(page.locator("#status")).toHaveText(/^Match over: 2 games in \d+ s\.$/);
  await expect(page.locator("#stat-games")).toHaveText("2 / 2");
  await expect(page.locator("#stat-record")).toHaveText(/^\d \/ \d \/ \d$/);
  await expect(page.locator("#stat-score")).toHaveText(/^\d+\.\d %$/);
  await expect(page.locator("#stat-score-range")).toHaveText(/^95 %: \d+\.\d % to \d+\.\d %$/);
  await expect(page.locator("#stat-first-visits")).toHaveText(/^\d+\.\d$/);
  await expect(page.locator("#stat-first-ms")).toHaveText(/^\d+ ms per move$/);
  await expect(page.locator("#results tr")).toHaveCount(2);

  const colors = await page.locator("#results tr td:nth-child(2)").allTextContents();

  expect(colors.sort()).toEqual(["black", "white"]);
});

test("the live match can be stopped", async ({ page }) => {
  await page.goto("/demo/match.html?games=20&time=50&parallel=1");
  await page.locator("#start").click();
  await expect(page.locator("body")).toHaveAttribute("data-state", "running");
  await expect(page.locator("#stat-first-visits")).toHaveText(/^\d+\.\d$/);
  await expect(page.locator("#stat-games")).toHaveText("0 / 20");

  await page.locator("#stop").click();

  await expect(page.locator("body")).toHaveAttribute("data-state", "stopped");
  await expect(page.locator("#start")).toBeEnabled();
});
