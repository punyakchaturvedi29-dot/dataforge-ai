import { describe, expect, it, vi } from 'vitest'
vi.mock('../../src/api/endpoints', () => ({ api: {} }))
import { validateFile } from '../../src/pages/Home'

const f = (name, size) => ({ name, size })

describe('validateFile (client-side checks before upload)', () => {
  it('accepts supported formats', () => {
    for (const n of ['a.csv', 'b.TSV', 'c.xlsx', 'd.json', 'e.jsonl']) expect(validateFile(f(n, 10), {})).toBeNull()
  })
  it('rejects unsupported or extension-less files', () => {
    expect(validateFile(f('model.pkl', 10), {})).toMatch(/not supported/)
    expect(validateFile(f('README', 10), {})).toMatch(/not supported/)
  })
  it('rejects empty files', () => {
    expect(validateFile(f('a.csv', 0), {})).toBe('The file is empty.')
  })
  it('enforces the server-provided size limit', () => {
    expect(validateFile(f('a.csv', 6 * 1024 * 1024), { max_upload_mb: 5 })).toMatch(/limit on this server is 5 MB/)
    expect(validateFile(f('a.csv', 4 * 1024 * 1024), { max_upload_mb: 5 })).toBeNull()
  })
})
