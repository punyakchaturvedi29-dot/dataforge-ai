export function fmtNum(v, { digits = 2, compact = false } = {}) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  if (typeof v !== 'number') return String(v)
  const a = Math.abs(v)
  if (compact && a >= 1e6) return (v / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 }) + 'M'
  if (compact && a >= 1e4) return (v / 1e3).toLocaleString(undefined, { maximumFractionDigits: 1 }) + 'K'
  if (Number.isInteger(v)) return v.toLocaleString()
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (a >= 1) return v.toLocaleString(undefined, { maximumFractionDigits: digits })
  if (a === 0) return '0'
  return v.toLocaleString(undefined, { maximumSignificantDigits: 3 })
}
export const fmtPct = (v, digits = 1) => (v === null || v === undefined ? '—' : `${Number(v).toFixed(digits)}%`)
export const fmtP = p => (p === null || p === undefined ? 'n/a' : p < 0.001 ? '< 0.001' : p.toFixed(3))
export function fmtCell(v, { raw = false } = {}) {
  if (v === null || v === undefined) return null
  if (typeof v === 'number') {
    if (raw) return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(10)))
    return fmtNum(v, { digits: 4 })
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  const s = String(v)
  // Only ISO timestamps get reformatted; all other text is shown exactly as stored.
  if (/^\d{4}-\d{2}-\d{2}T00:00:00(\.0+)?$/.test(s)) return s.slice(0, 10)
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(s)) return s.replace('T', ' ')
  return s
}

/** Nice axis tick label for a value given the axis span. */
export function fmtTick(v, span) {
  if (v === null || v === undefined || Number.isNaN(v)) return ''
  if (Math.abs(span) >= 20 || Math.abs(v) >= 1e4) return fmtNum(Math.round(v), { compact: true })
  return fmtNum(Number(v.toPrecision(3)))
}
export function timeAgo(ts) {
  if (!ts) return ''
  const s = Date.now() / 1000 - ts
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}
export const fmtDate = t => (t ? String(t).slice(0, 10) : '')
export const plural = (n, word, many) => `${(n ?? 0).toLocaleString()} ${n === 1 ? word : many || word + 's'}`
export const bytes = n => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n > 1e3 ? `${(n / 1e3).toFixed(0)} KB` : `${n} B`)
