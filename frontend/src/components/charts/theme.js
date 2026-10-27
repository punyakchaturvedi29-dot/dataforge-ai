import { useEffect, useState } from 'react'

const KEYS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'grid', 'axis', 'ink', 'ink-2', 'ink-3', 'chart-surface',
  'seq-1', 'seq-3', 'seq-5', 'seq-7', 'seq-9', 'div-neg', 'div-mid', 'div-pos', 'band', 'band-2', 'line-2', 'bad', 'good']

function read() {
  const cs = getComputedStyle(document.documentElement)
  const out = {}
  KEYS.forEach(k => { out[k] = cs.getPropertyValue(`--${k}`).trim() || '#888' })
  out.series = [out.s1, out.s2, out.s3, out.s4, out.s5, out.s6, out.s7, out.s8]
  return out
}

/** Chart colours resolved from CSS tokens; updates when the theme (attribute or OS) changes. */
export function useChartTheme() {
  const [t, setT] = useState(read)
  useEffect(() => {
    const update = () => setT(read())
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    mq?.addEventListener?.('change', update)
    return () => { mo.disconnect(); mq?.removeEventListener?.('change', update) }
  }, [])
  return t
}

function hexToRgb(h) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(h || '')
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128]
}
function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b)
  return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`
}
/** Diverging colour for v in [-1, 1]: neg pole -> neutral midpoint -> pos pole. */
export function divergingColor(theme, v) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'transparent'
  const t = Math.max(-1, Math.min(1, v))
  return t >= 0 ? mix(theme['div-mid'], theme['div-pos'], t) : mix(theme['div-mid'], theme['div-neg'], -t)
}
/** Sequential (one hue, light -> dark) for t in [0, 1]. */
export function sequentialColor(theme, t) {
  const stops = [theme['seq-1'], theme['seq-3'], theme['seq-5'], theme['seq-7'], theme['seq-9']]
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1)
  const i = Math.min(Math.floor(x), stops.length - 2)
  return mix(stops[i], stops[i + 1], x - i)
}
