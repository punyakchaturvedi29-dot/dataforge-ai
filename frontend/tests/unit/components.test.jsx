import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DataTable, ErrorState, Markdown } from '../../src/components/ui'
import { ClaimBadge, FindingCard } from '../../src/components/Findings'
import { impactText } from '../../src/pages/Clean'

vi.mock('../../src/api/endpoints', () => ({ api: {} }))

describe('Markdown', () => {
  it('renders bold, lists and headings', () => {
    const { container } = render(<Markdown text={'# Title\nSome **bold** text\n- one\n- two'} />)
    expect(container.querySelector('h3').textContent).toBe('Title')
    expect(container.querySelector('strong').textContent).toBe('bold')
    expect(container.querySelectorAll('li')).toHaveLength(2)
  })
  it('never renders HTML from the text', () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> <script>bad()</script>'} />)
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
    expect(container.textContent).toContain('<img src=x')
  })
})

describe('DataTable', () => {
  it('marks missing values and shows an empty state', () => {
    const cols = [{ key: 'a', label: 'A' }, { key: 'b', label: 'B', numeric: true }]
    const { container, rerender } = render(<DataTable columns={cols} rows={[{ a: 'x', b: null }]} />)
    expect(container.querySelector('.null')).not.toBeNull()
    rerender(<DataTable columns={cols} rows={[]} empty="Nothing here" />)
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
  })
})

describe('ErrorState', () => {
  it('shows the message, request id and a retry button', async () => {
    const retry = vi.fn()
    render(<ErrorState error={{ message: 'Boom', requestId: 'req-1' }} onRetry={retry} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Boom')
    expect(screen.getByText(/req-1/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retry).toHaveBeenCalled()
  })
})

describe('Findings', () => {
  const f = {
    id: 'f1', title: 'Revenue differs by segment', observation: 'Premium spends more.', claim_type: 'association', confidence: 'high',
    evidence_text: 'p < 0.001', score: 0.9, why_it_matters: 'Pricing', interpretations: ['Mix'], follow_up: ['Check by region'],
  }
  it('labels associations without implying causation', () => {
    render(<ClaimBadge type="association" />)
    expect(screen.getByText('Association')).toHaveAttribute('title', expect.stringMatching(/does not show cause/))
  })
  it('expands details on demand', async () => {
    render(<FindingCard f={f} />)
    expect(screen.queryByText('Check by region')).toBeNull()
    await userEvent.click(screen.getByRole('button', { name: /why it matters/i }))
    expect(screen.getByText('Check by region')).toBeInTheDocument()
  })
})

describe('impactText', () => {
  it('summarises an operation impact', () => {
    expect(impactText({ rows_removed: 3, cells_modified: 1 })).toBe('3 rows removed, 1 value changed')
    expect(impactText({})).toBe('no changes')
  })
})
