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

/**
 * Passes until both sides pass in a row, which opens the scoring step.
 *
 * Answering a pass with a move is normal: the engine only passes once the
 * position is settled. Keep passing (the pass button auto-waits for the engine
 * reply in between) until both sides pass in a row or the move cap ends the
 * game. The engine reply grows the move list by one more entry, a one-sided
 * condition immune to fast replies that return to "human" between two polling
 * frames.
 */
const passUntilScoring = async (page: Page) => {
  let rounds = 0;
  while (
    (await page.locator("body").getAttribute("data-state")) !== "scoring" &&
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
          return state === "scoring" ? Number.POSITIVE_INFINITY : count;
        },
        { timeout: ENGINE_REPLY_TIMEOUT_MS },
      )
      .toBeGreaterThanOrEqual(before + 2);
    rounds += 1;
  }

  await expect(page.locator("body")).toHaveAttribute("data-state", "scoring");
};

/** The click target of the point under the first white stone. */
const findWhiteStoneMove = (page: Page) =>
  page.evaluate(() => {
    const stone = document.querySelector('#board [data-stone="white"]');

    if (stone === null) {
      throw new Error("No white stone on the board.");
    }

    const target = [...document.querySelectorAll<SVGElement>("#board [data-move]")].find(
      (point) =>
        point.getAttribute("cx") === stone.getAttribute("cx") &&
        point.getAttribute("cy") === stone.getAttribute("cy"),
    );

    return Number(target?.dataset.move);
  });

const SCORE = "((B|W)\\+\\d+|Jigo) \\(komi 7\\)";

test("after both sides pass, the player marks dead stones and accepts the score", async ({ page }) => {
  test.setTimeout(TEST_TIMEOUT_MS);
  await openSite(page);
  await passUntilScoring(page);

  const status = page.locator("#status");
  const deadStones = page.locator("#board [data-dead]");
  await expect(status).toHaveText(new RegExp(`^Mark the dead stones, then accept\\. Score: ${SCORE}$`));
  await expect(page.locator("#accept")).toBeEnabled();

  // Clicking a chain flips it between dead and alive, and the score follows.
  const estimate = await status.textContent();
  const estimatedDead = await deadStones.count();
  const move = await findWhiteStoneMove(page);

  await page.locator(`#board [data-move="${move}"]`).click();
  await expect(deadStones).not.toHaveCount(estimatedDead);
  await expect(status).not.toHaveText(estimate ?? "");

  await page.locator(`#board [data-move="${move}"]`).click();
  await expect(deadStones).toHaveCount(estimatedDead);
  await expect(status).toHaveText(estimate ?? "");

  await page.locator("#accept").click();
  await expect(page.locator("body")).toHaveAttribute("data-state", "over");
  await expect(status).toHaveText(new RegExp(`^Game over: ${SCORE}$`));
  await expect(page.locator("#accept")).toBeDisabled();
});

test("the 19×19 page answers a human move", async ({ page }) => {
  test.setTimeout(TEST_TIMEOUT_MS);
  // The directory itself, as Cloudflare Pages serves it.
  await page.goto("/site/19x19/");
  await expect(page.locator("body")).toHaveAttribute("data-state", "human", {
    timeout: LOADING_TIMEOUT_MS,
  });
  await expect(page.locator("#board [data-move]")).toHaveCount(19 * 19);

  // The center point, K10.
  await page.locator('#board [data-move="180"]').click();

  await expect(page.locator("#board [data-stone]")).toHaveCount(2, {
    timeout: ENGINE_REPLY_TIMEOUT_MS,
  });
  await expect(page.locator("body")).toHaveAttribute("data-state", "human", {
    timeout: ENGINE_REPLY_TIMEOUT_MS,
  });
  await expect(page.locator("#moves li").first()).toContainText("K10");
});
