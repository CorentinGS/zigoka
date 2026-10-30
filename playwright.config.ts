import { defineConfig, devices } from "@playwright/test";

const PORT = 4174;

export default defineConfig({
  testDir: "tests/e2e",
  forbidOnly: true,
  reporter: "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `node bench/serve.mjs --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/e2e/index.html`,
    reuseExistingServer: false,
  },
});
