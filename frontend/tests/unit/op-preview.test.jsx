import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const api = { previewOp: vi.fn(), applyOp: vi.fn() }
const dataChanged = vi.fn(async () => {})
const notify = vi.fn()
vi.mock('../../src/api/endpoints', () => ({ api }))
vi.mock('../../src/state/workspace', () => ({ useWorkspace: () => ({ id: 'w1', dataChanged }) }))
vi.mock('../../src/state/ui', () => ({ useUi: () => ({ notify }) }))
vi.mock('../../src/components/layout/Shell', () => ({ PageHeader: () => null }))

const { OpPreview } = await import('../../src/pages/Clean')

const preview = (impact) => ({
  impact: { rows_before: 100, rows_after: 100 - (impact.rows_removed || 0), cells_modified: 0, values_filled: 0, non_null_values_lost: 0, ...impact },
  applies_to_new_data: false, notes: [], fitted: {}, sample: { columns: [], changed_rows: [], removed_rows: [] },
})

beforeEach(() => { vi.clearAllMocks() })

describe('OpPreview', () => {
  it('requires explicit acknowledgement before a destructive step', async () => {
    api.previewOp.mockResolvedValue(preview({ rows_removed: 4, requires_confirmation: true }))
    api.applyOp.mockResolvedValue({ applied: { label: 'Remove duplicate rows', impact: { rows_removed: 4 } } })
    const onClose = vi.fn()
    render(<OpPreview op="drop_duplicates" params={{}} onClose={onClose} />)
    const btn = await screen.findByRole('button', { name: 'Apply and remove data' })
    expect(btn).toBeDisabled()
    await userEvent.click(screen.getByRole('checkbox'))
    expect(btn).toBeEnabled()
    await userEvent.click(btn)
    await waitFor(() => expect(api.applyOp).toHaveBeenCalledWith('w1', { op: 'drop_duplicates', params: {}, confirm: true }))
    expect(dataChanged).toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })
  it('allows non-destructive steps straight away', async () => {
    api.previewOp.mockResolvedValue(preview({ cells_modified: 5 }))
    render(<OpPreview op="trim_whitespace" params={{ columns: ['a'] }} onClose={() => {}} />)
    expect(await screen.findByRole('button', { name: 'Apply step' })).toBeEnabled()
    expect(screen.queryByRole('checkbox')).toBeNull()
  })
  it('shows server errors from the preview', async () => {
    api.previewOp.mockRejectedValue(Object.assign(new Error('Column not found'), { status: 422 }))
    render(<OpPreview op="trim_whitespace" params={{}} onClose={() => {}} />)
    expect(await screen.findByText('Column not found')).toBeInTheDocument()
  })
})
