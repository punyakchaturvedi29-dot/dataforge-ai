import { describe, expect, it } from 'vitest'
import { bytes, fmtCell, fmtNum, fmtP, fmtPct, fmtTick, plural } from '../../src/lib/format'

describe('fmtCell', () => {
  it('returns null for missing values so the UI can mark them', () => {
    expect(fmtCell(null)).toBeNull()
    expect(fmtCell(undefined)).toBeNull()
  })
  it('never rewrites ordinary text that contains a capital T', () => {
    expect(fmtCell('Jack Taylor')).toBe('Jack Taylor')
    expect(fmtCell('Tokyo T-shirt')).toBe('Tokyo T-shirt')
  })
  it('shortens midnight ISO timestamps to dates and formats other timestamps', () => {
    expect(fmtCell('2024-03-05T00:00:00')).toBe('2024-03-05')
    expect(fmtCell('2024-03-05T13:45:10')).toBe('2024-03-05 13:45:10')
  })
  it('shows raw numbers without rounding away precision', () => {
    expect(fmtCell(1234567, { raw: true })).toBe('1234567')
    expect(fmtCell(0.1 + 0.2, { raw: true })).toBe('0.3')
  })
  it('formats booleans explicitly', () => {
    expect(fmtCell(false)).toBe('false')
  })
})

describe('number helpers', () => {
  it('fmtNum handles missing, integers, compact and small values', () => {
    expect(fmtNum(null)).toBe('—')
    expect(fmtNum(NaN)).toBe('—')
    expect(fmtNum(0)).toBe('0')
    expect(fmtNum(25000, { compact: true })).toMatch(/^25K$/)
    expect(fmtNum(3_400_000, { compact: true })).toMatch(/^3\.4M$/)
    expect(fmtNum(0.0001234)).toBe('0.000123')
  })
  it('fmtP never prints p = 0', () => {
    expect(fmtP(0)).toBe('< 0.001')
    expect(fmtP(0.0421)).toBe('0.042')
    expect(fmtP(null)).toBe('n/a')
  })
  it('fmtPct, plural and bytes', () => {
    expect(fmtPct(12.345)).toBe('12.3%')
    expect(plural(1, 'row')).toBe('1 row')
    expect(plural(3, 'row')).toBe('3 rows')
    expect(plural(2, 'entry', 'entries')).toBe('2 entries')
    expect(bytes(2_500_000)).toBe('2.5 MB')
    expect(bytes(500)).toBe('500 B')
  })
  it('fmtTick uses compact labels for wide spans', () => {
    expect(fmtTick(12000, 50000)).toBe('12K')
    expect(fmtTick(0.12345, 1)).toBe('0.123')
    expect(fmtTick(NaN, 1)).toBe('')
  })
})
