import { expect, test, type Page } from "@playwright/test";

const CENTER_MOVE = 40;

// Loading the two wasm modules and the weights takes a few seconds on the
// first run; a beginner reply is a single inference.
const LOADING_TIMEOUT_MS = 30_000;
const ENGINE_REPLY_TIMEOUT_MS = 30_000;
const TEST_TIMEOUT_MS = 120_000;
// A full game until both sides pass stays well below the 81 points of the
// board, even with the engine answering passes by playing.
const MAX_PASS_ROUNDS = 130;

const YOUR_MOVE = "Your move (black).";

const openSite = async (page: Page, query = "") => {
  await page.goto(`/site/index.html${query}`);
  await expect(page.locator("#level")).toHaveValue("beginner");
  await expect(page.locator("body")).toHaveAttribute("data-state", "human", {
    timeout: LOADING_TIMEOUT_MS,
  });
  await expect(page.locator("#status")).toHaveText(YOUR_MOVE);
};

const playOneExchange = async (page: Page) => {
  await page.locator(`#board [data-move="${CENTER_MOVE}"]`).click();

  await expect(page.locator("#board [data-stone]")).toHaveCount(2, {
    timeout: ENGINE_REPLY_TIMEOUT_MS,
  });
  await expect(page.locator("body")).toHaveAttribute("data-state", "human", {
    timeout: ENGINE_REPLY_TIMEOUT_MS,
  });
};

test("the site answers a human move under a strict CSP", async ({ page }) => {
  test.setTimeout(TEST_TIMEOUT_MS);
  await openSite(page, "?csp=strict");

  await playOneExchange(page);

  await expect(page.locator('#board [data-stone="black"]')).toHaveCount(1);
  await expect(page.locator('#board [data-stone="white"]')).toHaveCount(1);
  await expect(page.locator("#status")).toHaveText(YOUR_MOVE);
});

test("undo after one exchange restores the initial position", async ({ page }) => {
  test.setTimeout(TEST_TIMEOUT_MS);
  await openSite(page);
  await playOneExchange(page);

  await page.locator("#undo").click();

  await expect(page.locator("#board [data-stone]")).toHaveCount(0);
  await expect(page.locator("body")).toHaveAttribute("data-state", "human");
  await expect(page.locator("#status")).toHaveText(YOUR_MOVE);
});

test("passing until both sides pass ends the game with a score", async ({ page }) => {
  test.setTimeout(TEST_TIMEOUT_MS);
  await openSite(page);

  // Answering a pass with a move is normal: the engine only passes once the
  // position is settled. Keep passing (the pass button auto-waits for the
  // engine reply in between) until both sides pass in a row or the move cap
  // ends the game. The engine reply grows the move list by one more entry, a
  // one-sided condition immune to fast replies that return to "human" between
  // two polling frames.
  let rounds = 0;
  while (
    (await page.locator("body").getAttribute("data-state")) !== "over" &&
    rounds < MAX_PASS_ROUNDS
  ) {
    const before = await page.locator("#moves li").count();
    await page.locator("#pass").click();
    await expect
      .poll(
        async () => {
          const [count, state] = await Promise.all([
            page.locator("#moves li").count(),
            page.locator("body").getAttribute("data-state"),
          ]);
          return state === "over" ? Number.POSITIVE_INFINITY : count;
        },
        { timeout: ENGINE_REPLY_TIMEOUT_MS },
      )
      .toBeGreaterThanOrEqual(before + 2);
    rounds += 1;
  }

  await expect(page.locator("body")).toHaveAttribute("data-state", "over");
  await expect(page.locator("#status")).toHaveText(
    /^Game over: (B|W)\+\d+ \(komi 7\)/,
  );
});
