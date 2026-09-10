import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.pw.ts",
  fullyParallel: false,
  retries: 0,
  reporter: "line",
  outputDir: "./node_modules/.cache/playwright-results",
  use: {
    baseURL: "http://127.0.0.1:5180",
    headless: true,
  },
  webServer: {
    command: "pnpm dev --host 127.0.0.1",
    url: "http://127.0.0.1:5180",
    reuseExistingServer: false,
  },
});
