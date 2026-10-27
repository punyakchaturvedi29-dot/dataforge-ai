import { expect, test } from '@playwright/test'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'

const SAMPLE = path.resolve('..', 'sample_data.csv')
const shots = process.env.E2E_SHOTS

async function shot(page, name) {
  if (shots) await page.screenshot({ path: path.join(shots, `${name}.png`), fullPage: true })
}

const nav = (page, name) => page.getByRole('navigation', { name: 'Workspace sections' }).getByRole('link', { name, exact: true })

function tmpFile(name, content) {
  const p = path.join(os.tmpdir(), `df-e2e-${Date.now()}-${name}`)
  fs.writeFileSync(p, content)
  return p
}

test('rejects unsupported and empty files with clear messages', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('file-input').setInputFiles(tmpFile('virus.exe', 'MZ...'))
  await expect(page.getByText(/files are not supported/)).toBeVisible()
  await page.getByTestId('file-input').setInputFiles(tmpFile('empty.csv', ''))
  await expect(page.getByText(/empty/i).first()).toBeVisible()
  await page.getByTestId('file-input').setInputFiles(tmpFile('header.csv', 'a,b,c\n'))
  await expect(page.getByText(/header but no data rows/)).toBeVisible()
})

test('full workflow: upload, validate, clean, analyse, model, predict, forecast, report', async ({ page }) => {
  // ── Upload & validate ──
  await page.goto('/')
  await page.getByTestId('file-input').setInputFiles(SAMPLE)
  await page.waitForURL(/\/w\/[a-f0-9]+\/data/)
  await expect(page.getByRole('table', { name: 'Dataset rows' })).toBeVisible()
  await expect(page.getByText('Jack Taylor')).toBeVisible()
  await expect(page.getByText('100 of 100 rows')).toBeVisible()
  await page.getByRole('tab', { name: /Validation report/ }).click()
  await expect(page.getByText('order_date', { exact: true })).toBeVisible()
  const wid = page.url().split('/w/')[1].split('/')[0]

  // server-side sorting
  await page.getByRole('tab', { name: 'Rows' }).click()
  await page.getByRole('button', { name: /^revenue/ }).click()
  await page.getByRole('button', { name: /^revenue/ }).click()
  await expect(page.locator('tbody tr').first()).toContainText('699')

  // ── Profile ──
  await nav(page, 'Profile').click()
  await expect(page.getByRole('heading', { name: 'Descriptive statistics' })).toBeVisible()
  await page.getByRole('tab', { name: 'Correlations' }).click()
  await expect(page.getByText('Tested pairs')).toBeVisible()

  // ── Clean: preview, confirm a destructive step, undo ──
  await nav(page, 'Clean').click()
  await page.getByRole('tab', { name: /Add a step/ }).click()
  await page.getByLabel('Operation').selectOption('remove_rows_where')
  await page.getByLabel('Column', { exact: true }).selectOption('quantity')
  await page.getByLabel('Condition').selectOption('>')
  await page.getByLabel('Value').fill('2')
  await page.getByRole('button', { name: 'Preview impact' }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByText(/rows? removed/).first()).toBeVisible()
  const apply = dialog.getByRole('button', { name: 'Apply and remove data' })
  await expect(apply).toBeDisabled()
  await dialog.getByRole('checkbox').check()
  await apply.click()
  await expect(page.getByText(/Remove rows matching a condition: .* removed/)).toBeVisible()
  await shot(page, 'e2e-clean-applied')
  await page.getByRole('tab', { name: /History/ }).click()
  await expect(page.getByText('1. Remove rows matching a condition')).toBeVisible()
  await page.getByRole('button', { name: 'Undo last step' }).click()
  await expect(page.getByText(/No cleaning steps yet/)).toBeVisible()

  // ── Insights & charts ──
  await nav(page, 'Insights').click()
  await expect(page.getByText("'revenue' differs across 'customer_segment'")).toBeVisible()
  await nav(page, 'Visualize').click()
  await page.getByRole('tab', { name: /Chart builder/ }).click()
  await page.getByLabel('Chart type').selectOption('bar')
  await page.getByLabel('Category').selectOption('region')
  await page.getByRole('button', { name: 'Draw chart' }).click()
  await expect(page.locator('figure.chart-card')).toBeVisible()

  // ── Model ──
  await nav(page, 'Train models').click()
  await page.getByLabel('Target column').selectOption('revenue')
  await page.getByRole('button', { name: 'Quick' }).click()
  await page.getByRole('button', { name: 'Train and compare models' }).click()
  await expect(page.getByText(/predicts “revenue”/)).toBeVisible({ timeout: 150_000 })
  await expect(page.getByRole('heading', { name: 'Held-out test set' })).toBeVisible()
  await expect(page.getByText(/was selected because it had the best mean RMSE/)).toBeVisible()
  await shot(page, 'e2e-model')

  // ── Predict ──
  await page.getByRole('button', { name: 'Use for predictions' }).click()
  await page.waitForURL(/predict\?model=/)
  await page.getByRole('button', { name: 'Predict', exact: true }).click()
  await expect(page.getByText('Why this prediction')).toBeVisible()
  await shot(page, 'e2e-predict')

  // ── Forecast ──
  await nav(page, 'Forecast').click()
  await page.getByLabel('Measure').selectOption('revenue')
  await page.getByRole('button', { name: 'Run forecast' }).click()
  await expect(page.getByText('Method comparison')).toBeVisible({ timeout: 60_000 })
  await shot(page, 'e2e-forecast')

  // ── Diagnose ──
  await nav(page, 'Diagnose').click()
  await page.getByLabel('Outcome', { exact: true }).selectOption('revenue')
  await page.getByRole('button', { name: 'Run diagnostics' }).click()
  await expect(page.getByText(/cannot prove causation/)).toBeVisible({ timeout: 60_000 })
  await shot(page, 'e2e-diagnose')

  // ── Copilot ──
  await nav(page, 'Copilot').click()
  await page.getByLabel('Question').fill('Compare revenue across customer segments')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByText(/highest total revenue is/)).toBeVisible()
  await expect(page.getByText('computed from data').last()).toBeVisible()
  await shot(page, 'e2e-copilot')

  // ── Report ──
  await nav(page, 'Report').click()
  await page.getByRole('tab', { name: /Report builder/ }).click()
  await page.getByRole('button', { name: 'Preview on screen' }).click()
  await expect(page.getByRole('article', { name: 'Report preview' })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByRole('heading', { name: /Machine learning: predicting 'revenue'/ })).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'PDF' }).click()
  const d = await download
  expect(d.suggestedFilename()).toMatch(/\.pdf$/)
  await shot(page, 'e2e-report')

  // ── Workflow rail shows completed stages ──
  await page.goto(`/w/${wid}`)
  const rail = page.getByRole('navigation', { name: 'Analysis workflow' })
  for (const s of ['Upload', 'Validate', 'Profile', 'Clean', 'Analyze', 'Model', 'Visualize', 'Export']) {
    await expect(rail.getByRole('link', { name: new RegExp(s) })).toContainText('Done')
  }
  await shot(page, 'e2e-overview-done')
})

test('keyboard: command palette navigates and dialogs close with Escape', async ({ page, request }) => {
  const r = await request.post('/api/workspaces/sample')
  const ws = await r.json()
  await page.goto(`/w/${ws.id}`)
  await expect(page.getByRole('heading', { name: 'Top findings' })).toBeVisible()
  await page.keyboard.press('Control+k')
  await page.getByRole('dialog', { name: 'Search' }).getByLabel('Search').fill('forecast')
  await page.keyboard.press('Enter')
  await page.waitForURL(/forecast/)
  await page.getByRole('button', { name: 'Help for this page' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('dark theme renders with its own tokens', async ({ page, request }) => {
  const ws = await (await request.post('/api/workspaces/sample')).json()
  await page.goto(`/w/${ws.id}/visualize`)
  await page.getByRole('button', { name: 'Switch to dark theme' }).click()
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
  expect(bg).toBe('rgb(14, 17, 21)')
  await page.waitForTimeout(1500)
  await shot(page, 'e2e-dark')
})
