import { defineConfig } from '@playwright/test'

// End-to-end acceptance tests run against the real FastAPI backend serving the built frontend.
// Build first (npm run build), then: npm run test:e2e
// Set E2E_BASE_URL to test an already running server; CHROME_PATH to use a specific Chromium.
const base = process.env.E2E_BASE_URL || 'http://127.0.0.1:8765'
export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: base,
    viewport: { width: 1400, height: 900 },
    screenshot: 'only-on-failure',
    launchOptions: process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  },
  webServer: process.env.E2E_BASE_URL ? undefined : {
    command: `${process.env.PYTHON || 'python'} -m uvicorn main:app --port 8765`,
    cwd: '../backend',
    url: `${base}/api/health`,
    reuseExistingServer: true,
    timeout: 60_000,
    env: { DATAFORGE_DATA_DIR: process.env.E2E_DATA_DIR || '../backend/data-e2e', LOG_LEVEL: 'WARNING', WORKSPACE_LISTING: 'true' },
  },
})
