import { expect, test, type Page } from "@playwright/test";

const CENTER_MOVE = 40;
const GAME_TIMEOUT_MS = 60_000;

const openDemo = async (page: Page, query = "") => {
  await page.goto(`/demo/index.html${query}`);
  await expect(page.locator("body")).toHaveAttribute("data-state", "human");
};

test("the demo answers a black move with a Zig move timed against Moka", async ({ page }) => {
  await openDemo(page);
  await expect(page.locator("#initialization")).toContainText("Initialization: Zig");

  await page.locator(`[data-move="${CENTER_MOVE}"]`).click();

  await expect(page.locator("body")).toHaveAttribute("data-move-count", "2");
  await expect(page.locator("body")).toHaveAttribute("data-state", "human");
  await expect(page.locator("[data-stone]")).toHaveCount(2);
  await expect(page.locator("#status")).toHaveText(/^Zig played [A-HJ]\d\. Your move \(black\)\.$/);

  const cells = page.locator("#timings tr").first().locator("td");

  await expect(cells).toHaveCount(5);
  await expect(cells.nth(0)).toHaveText("2");
  await expect(cells.nth(4)).toHaveText("yes");

  for (const index of [2, 3]) {
    expect(Number(await cells.nth(index).textContent())).toBeGreaterThan(0);
  }
});

test("the demo refuses a move on an occupied point", async ({ page }) => {
  await openDemo(page);
  await page.locator(`[data-move="${CENTER_MOVE}"]`).click();
  await expect(page.locator("body")).toHaveAttribute("data-move-count", "2");
  await expect(page.locator("body")).toHaveAttribute("data-state", "human");

  await page.locator(`[data-move="${CENTER_MOVE}"]`).click();

  await expect(page.locator("#status")).toHaveText("Illegal move (E5). Your move (black).");
  await expect(page.locator("body")).toHaveAttribute("data-move-count", "2");
});

test("Zig playing both sides reaches a scored end of game", async ({ page }) => {
  test.setTimeout(GAME_TIMEOUT_MS + 10_000);
  await openDemo(page, "?delay=0");

  await page.locator("#autoplay").check();

  await expect(page.locator("body")).toHaveAttribute("data-state", "over", {
    timeout: GAME_TIMEOUT_MS,
  });
  await expect(page.locator("#status")).toHaveText(
    /^Game over: (B\+\d+|W\+\d+|Jigo) \(\d+ dead stones removed, komi 7\)\.$/,
  );
  await expect(page.locator("#same-count")).toHaveText(/^(\d+) \/ \1$/);
});
