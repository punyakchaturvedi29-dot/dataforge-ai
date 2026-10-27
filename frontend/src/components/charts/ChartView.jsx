import { useEffect, useRef, useState } from 'react'
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, Pie, PieChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fmtNum, fmtTick } from '../../lib/format'
import { divergingColor, sequentialColor, useChartTheme } from './theme'

const AX = { tickLine: false, tick: { fontSize: 11 } }

export function ChartTooltip({ active, payload, label, title, rows }) {
  if (!active || !payload?.length) return null
  const lines = rows ? rows(payload, label) : payload.map(p => [p.name, fmtNum(p.value)])
  return (
    <div className="tooltip">
      {(title ? title(payload, label) : label) !== undefined && <div className="tt-title">{title ? title(payload, label) : label}</div>}
      {lines.map(([k, v], i) => <div key={i} className="num"><span className="muted">{k}: </span>{v}</div>)}
    </div>
  )
}

export function Legend({ items }) {
  if (!items || items.length < 2) return null
  return (
    <div className="legend" aria-label="Legend">
      {items.map(it => (
        <span key={it.label} className="legend-item">
          <span className="legend-swatch" style={{ background: it.color, borderRadius: it.line ? 1 : 3, height: it.line ? 3 : 10 }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

function useWidth() {
  const ref = useRef(null)
  const [w, setW] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(e => setW(e[0].contentRect.width || 600))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

export function ChartCard({ chart, title, subtitle, height = 240, actions, footer }) {
  return (
    <figure className="chart-card" style={{ margin: 0 }}>
      <div className="row-between">
        <div className="grow">
          <figcaption className="chart-title">{title ?? chart?.title}</figcaption>
          {(subtitle || chart?.reason) && <div className="chart-sub">{subtitle ?? chart.reason}</div>}
        </div>
        {actions}
      </div>
      {chart ? <ChartBody chart={chart} height={height} /> : null}
      {chart?.notes?.length > 0 && <div className="chart-notes">{chart.notes.map((n, i) => <span key={i}>{n}</span>)}</div>}
      {footer}
    </figure>
  )
}

export function ChartBody({ chart, height = 240 }) {
  const t = useChartTheme()
  if (!chart) return null
  switch (chart.type) {
    case 'histogram': return <Histogram c={chart} t={t} h={height} />
    case 'bar': return <Bars c={chart} t={t} h={height} />
    case 'missing': return <MissingBars c={chart} t={t} h={height} />
    case 'pie': return <Donut c={chart} t={t} h={height} />
    case 'line': return <Lines c={chart} t={t} h={height} />
    case 'scatter': return <Scatterplot c={chart} t={t} h={height} />
    case 'box': return <BoxPlot c={chart} t={t} h={height} />
    case 'heatmap': return <Heatmap c={chart} t={t} />
    default: return <div className="muted small">Unsupported chart type: {chart.type}</div>
  }
}

function Histogram({ c, t, h }) {
  const data = c.data.map(d => ({ ...d, name: d.label }))
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }} barCategoryGap={1}>
        <CartesianGrid vertical={false} stroke={t.grid} />
        <XAxis dataKey="name" {...AX} interval="preserveStartEnd" minTickGap={18} stroke={t.axis} />
        <YAxis {...AX} width={44} stroke={t.axis} allowDecimals={false} />
        <Tooltip cursor={{ fill: t.band }} content={<ChartTooltip title={(p) => `${c.x}: ${p[0].payload.label}`} rows={p => [['Rows', fmtNum(p[0].value)]]} />} />
        <Bar dataKey="count" fill={t.s1} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function Bars({ c, t, h }) {
  const data = c.data.map(d => ({ ...d, name: d.label }))
  const height = Math.max(h, data.length * 26 + 30)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }} barCategoryGap={4}>
        <CartesianGrid horizontal={false} stroke={t.grid} />
        <XAxis type="number" {...AX} stroke={t.axis} tickFormatter={v => fmtNum(v, { compact: true })} />
        <YAxis type="category" dataKey="name" {...AX} width={120} stroke={t.axis} tickFormatter={v => (String(v).length > 18 ? String(v).slice(0, 17) + '…' : v)} />
        <Tooltip cursor={{ fill: t.band }} content={<ChartTooltip rows={p => [[c.y_label || 'Value', fmtNum(p[0].value)], ['Rows', fmtNum(p[0].payload.n)]]} />} />
        <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={d.other ? t['line-2'] : t.s1} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function MissingBars({ c, t, h }) {
  if (!c.data.length) return <div className="muted small" style={{ padding: 20 }}>No missing values.</div>
  const data = c.data.map(d => ({ ...d, name: d.label }))
  return (
    <ResponsiveContainer width="100%" height={Math.max(h, data.length * 24 + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke={t.grid} />
        <XAxis type="number" domain={[0, 100]} unit="%" {...AX} stroke={t.axis} />
        <YAxis type="category" dataKey="name" {...AX} width={120} stroke={t.axis} />
        <Tooltip cursor={{ fill: t.band }} content={<ChartTooltip rows={p => [['Missing', `${p[0].value.toFixed(1)}%`], ['Rows', fmtNum(p[0].payload.count)]]} />} />
        <Bar dataKey="pct" fill={t.s2} radius={[0, 3, 3, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function Donut({ c, t, h }) {
  const data = c.data.map((d, i) => ({ ...d, name: d.label, fill: d.other ? t['line-2'] : t.series[i % 8] }))
  return (
    <div>
      <ResponsiveContainer width="100%" height={h}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="85%" paddingAngle={1} stroke={t['chart-surface']} strokeWidth={2} isAnimationActive={false}>
            {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Pie>
          <Tooltip content={<ChartTooltip title={p => p[0].name} rows={p => [['Value', fmtNum(p[0].value)], ['Share', `${p[0].payload.pct.toFixed(1)}%`]]} />} />
        </PieChart>
      </ResponsiveContainer>
      <Legend items={data.map(d => ({ label: `${d.name} (${d.pct.toFixed(0)}%)`, color: d.fill }))} />
    </div>
  )
}

function Lines({ c, t, h }) {
  const keys = c.series.map((s, i) => ({ key: `s${i}`, name: s.name, color: t.series[i % 8] }))
  const byT = new Map()
  c.series.forEach((s, i) => s.points.forEach(p => {
    const row = byT.get(p.t) || { t: p.t }
    row[`s${i}`] = p.value
    row[`n${i}`] = p.n
    byT.set(p.t, row)
  }))
  const data = [...byT.values()].sort((a, b) => (a.t < b.t ? -1 : 1))
  const partial = new Set(c.partial_periods || [])
  const tick = v => String(v).slice(0, c.freq === 'h' ? 13 : 10)
  return (
    <div>
      <ResponsiveContainer width="100%" height={h}>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis dataKey="t" {...AX} stroke={t.axis} tickFormatter={tick} minTickGap={28} />
          <YAxis {...AX} width={52} stroke={t.axis} tickFormatter={v => fmtNum(v, { compact: true })} />
          <Tooltip content={<ChartTooltip title={(p, l) => tick(l) + (partial.has(l) ? ' (possibly incomplete)' : '')}
            rows={p => p.map(x => [x.name, `${fmtNum(x.value)} (n=${x.payload[`n${x.dataKey.slice(1)}`] ?? 0})`])} />} />
          {keys.map(k => (
            <Line key={k.key} type="linear" dataKey={k.key} name={k.name} stroke={k.color} strokeWidth={2} isAnimationActive={false}
              connectNulls={false} dot={p => partial.has(p.payload.t)
                ? <circle key={p.key} cx={p.cx} cy={p.cy} r={4} fill={t['chart-surface']} stroke={k.color} strokeWidth={2} />
                : <g key={p.key} />} activeDot={{ r: 4 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <Legend items={keys.map(k => ({ label: k.name, color: k.color, line: true }))} />
    </div>
  )
}

function Scatterplot({ c, t, h }) {
  const tr = c.trend
  const groups = c.color ? [...new Set(c.points.map(p => p.c))].slice(0, 3) : [null]
  return (
    <div>
      <ResponsiveContainer width="100%" height={h}>
        <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 14 }}>
          <CartesianGrid stroke={t.grid} />
          <XAxis type="number" dataKey="x" name={c.x} {...AX} stroke={t.axis} tickFormatter={v => fmtNum(v, { compact: true })}
            label={{ value: c.x, position: 'insideBottom', offset: -8, fontSize: 11 }} domain={['auto', 'auto']} />
          <YAxis type="number" dataKey="y" name={c.y} {...AX} width={52} stroke={t.axis} tickFormatter={v => fmtNum(v, { compact: true })} domain={['auto', 'auto']} />
          <Tooltip cursor={{ stroke: t.axis }} content={<ChartTooltip title={() => ''} rows={p => [[c.x, fmtNum(p[0].payload.x)], [c.y, fmtNum(p[0].payload.y)]]} />} />
          {groups.map((g, i) => (
            <Scatter key={String(g)} name={g ?? c.y} data={g === null ? c.points : c.points.filter(p => p.c === g)} fill={t.series[i]} fillOpacity={0.55}
              shape={props => <circle cx={props.cx} cy={props.cy} r={3.5} fill={t.series[i]} fillOpacity={0.55} stroke={t['chart-surface']} strokeWidth={0.6} />} isAnimationActive={false} />
          ))}
          {tr && <ReferenceLine segment={[{ x: tr.x_min, y: tr.intercept + tr.slope * tr.x_min }, { x: tr.x_max, y: tr.intercept + tr.slope * tr.x_max }]}
            stroke={t.ink} strokeDasharray="5 4" strokeWidth={1.5} ifOverflow="hidden" />}
        </ScatterChart>
      </ResponsiveContainer>
      {tr && <div className="chart-sub num">Trend: slope {fmtNum(tr.slope)}, R² {fmtNum(tr.r2)}</div>}
    </div>
  )
}

function BoxPlot({ c, t, h }) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState(null)
  const groups = c.groups || []
  const vals = groups.flatMap(g => [g.whisker_low, g.whisker_high, ...(g.outliers || [])]).filter(v => v !== null && v !== undefined)
  if (!vals.length) return <div ref={ref} />
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.06
  lo -= pad; hi += pad
  const left = 48, right = 8, top = 8, bottom = 46
  const H = h
  const plotW = Math.max(100, width - left - right)
  const bw = plotW / groups.length
  const y = v => top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4)
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width="100%" height={H} role="img" aria-label={`Box plot of ${c.y}${c.x ? ' by ' + c.x : ''}`}>
        {ticks.map((v, i) => (
          <g key={i}>
            <line x1={left} x2={left + plotW} y1={y(v)} y2={y(v)} stroke={t.grid} />
            <text x={left - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill={t['ink-3']}>{fmtTick(v, hi - lo)}</text>
          </g>
        ))}
        {groups.map((g, i) => {
          const cx = left + bw * i + bw / 2
          const w = Math.min(46, bw * 0.55)
          return (
            <g key={g.label} onMouseEnter={() => setHover({ g, x: cx })} onMouseLeave={() => setHover(null)}>
              <rect x={left + bw * i} y={top} width={bw} height={H - top - bottom} fill="transparent" />
              <line x1={cx} x2={cx} y1={y(g.whisker_low)} y2={y(g.q1)} stroke={t['ink-3']} />
              <line x1={cx} x2={cx} y1={y(g.q3)} y2={y(g.whisker_high)} stroke={t['ink-3']} />
              <line x1={cx - w / 4} x2={cx + w / 4} y1={y(g.whisker_low)} y2={y(g.whisker_low)} stroke={t['ink-3']} />
              <line x1={cx - w / 4} x2={cx + w / 4} y1={y(g.whisker_high)} y2={y(g.whisker_high)} stroke={t['ink-3']} />
              <rect x={cx - w / 2} y={y(g.q3)} width={w} height={Math.max(1, y(g.q1) - y(g.q3))} rx={3} fill={t.band} stroke={t.s1} strokeWidth={1.5} />
              <line x1={cx - w / 2} x2={cx + w / 2} y1={y(g.median)} y2={y(g.median)} stroke={t.s1} strokeWidth={2.5} />
              {(g.outliers || []).slice(0, 60).map((o, j) => <circle key={j} cx={cx} cy={y(o)} r={2.5} fill="none" stroke={t.s2} strokeWidth={1.2} />)}
              <text x={cx} y={H - bottom + 16} textAnchor="middle" fontSize="11" fill={t['ink-2']}>{String(g.label).slice(0, Math.max(4, Math.floor(bw / 7)))}</text>
              <text x={cx} y={H - bottom + 30} textAnchor="middle" fontSize="10" fill={t['ink-3']}>n={g.n}</text>
            </g>
          )
        })}
      </svg>
      {hover && (
        <div className="tooltip" style={{ position: 'absolute', top: 6, left: Math.min(hover.x + 10, width - 190), width: 180, pointerEvents: 'none' }}>
          <div className="tt-title">{hover.g.label}</div>
          {[['Median', hover.g.median], ['Q1 – Q3', `${fmtNum(hover.g.q1)} – ${fmtNum(hover.g.q3)}`], ['Mean', hover.g.mean], ['Rows', hover.g.n], ['Outliers', hover.g.n_outliers]]
            .map(([k, v]) => <div key={k} className="num"><span className="muted">{k}: </span>{typeof v === 'number' ? fmtNum(v) : v}</div>)}
        </div>
      )}
    </div>
  )
}

function Heatmap({ c, t }) {
  const n = c.columns.length
  const [hover, setHover] = useState(null)
  const cell = n > 14 ? 26 : 36
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `140px repeat(${n}, ${cell}px)`, gap: 2, alignItems: 'center', fontSize: 11, width: 'max-content' }}
        role="table" aria-label="Correlation matrix">
        <div />
        {c.columns.map(col => <div key={col} role="columnheader" title={col} style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 96, overflow: 'hidden', color: t['ink-2'] }}>{col}</div>)}
        {c.matrix.map((row, i) => (
          <div key={i} style={{ display: 'contents' }} role="row">
            <div role="rowheader" className="truncate" title={c.columns[i]} style={{ color: t['ink-2'], paddingRight: 6, textAlign: 'right' }}>{c.columns[i]}</div>
            {row.map((v, j) => (
              <div key={j} role="cell" tabIndex={0} aria-label={`${c.columns[i]} and ${c.columns[j]}: ${v === null ? 'n/a' : v.toFixed(2)}`}
                onMouseEnter={() => setHover({ i, j, v })} onFocus={() => setHover({ i, j, v })} onMouseLeave={() => setHover(null)}
                style={{ height: cell, borderRadius: 3, background: divergingColor(t, v), display: 'grid', placeItems: 'center',
                  color: Math.abs(v ?? 0) > 0.55 ? '#fff' : t.ink, fontVariantNumeric: 'tabular-nums', outline: hover?.i === i && hover?.j === j ? `2px solid ${t.ink}` : 'none' }}>
                {cell >= 36 && v !== null ? v.toFixed(2).replace('0.', '.').replace('-0.', '-.') : ''}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="row tiny muted" style={{ marginTop: 8 }}>
        <span>−1</span>
        <span style={{ width: 140, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${divergingColor(t, -1)}, ${divergingColor(t, 0)}, ${divergingColor(t, 1)})` }} />
        <span>+1</span>
        {hover && <span className="num" style={{ color: t.ink }}>{c.columns[hover.i]} × {c.columns[hover.j]}: {hover.v === null ? 'n/a' : hover.v.toFixed(3)}</span>}
      </div>
    </div>
  )
}

/** Forecast with history, point forecast and 80/95% interval bands. */
export function ForecastChart({ fc, method, height = 300 }) {
  const t = useChartTheme()
  const f = fc.forecasts[method]
  if (!f) return null
  const hist = fc.history.slice(-120).map(h => ({ t: h.t, actual: h.value, filled: h.filled }))
  const last = hist[hist.length - 1]
  const fut = fc.future_index.map((ti, i) => ({ t: ti, forecast: f.forecast[i], b80: [f.lower80[i], f.upper80[i]], b95: [f.lower95[i], f.upper95[i]] }))
  const data = [...hist.slice(0, -1), { ...last, forecast: last.actual, b80: [last.actual, last.actual], b95: [last.actual, last.actual] }, ...fut]
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke={t.grid} />
          <XAxis dataKey="t" {...AX} stroke={t.axis} tickFormatter={v => String(v).slice(0, 10)} minTickGap={30} />
          <YAxis {...AX} width={56} stroke={t.axis} tickFormatter={v => fmtNum(v, { compact: true })} />
          <Tooltip content={<ChartTooltip title={(p, l) => String(l).slice(0, 10)} rows={p => {
            const d = p[0].payload
            const r = []
            if (d.actual !== undefined) r.push(['Actual', fmtNum(d.actual) + (d.filled ? ' (filled gap)' : '')])
            if (d.forecast !== undefined && d.actual === undefined) {
              r.push(['Forecast', fmtNum(d.forecast)])
              r.push(['80% interval', `${fmtNum(d.b80[0])} – ${fmtNum(d.b80[1])}`])
              r.push(['95% interval', `${fmtNum(d.b95[0])} – ${fmtNum(d.b95[1])}`])
            }
            return r
          }} />} />
          <Area dataKey="b95" stroke="none" fill={t['band-2']} isAnimationActive={false} name="95% interval" />
          <Area dataKey="b80" stroke="none" fill={t.band} isAnimationActive={false} name="80% interval" />
          <Line dataKey="actual" stroke={t.ink} strokeWidth={2} dot={false} isAnimationActive={false} name="Actual" />
          <Line dataKey="forecast" stroke={t.s1} strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} name="Forecast" />
          <ReferenceLine x={last.t} stroke={t.axis} strokeDasharray="3 3" label={{ value: 'forecast starts', position: 'insideTopRight', fontSize: 10, fill: t['ink-3'] }} />
        </ComposedChart>
      </ResponsiveContainer>
      <Legend items={[{ label: 'Actual', color: t.ink, line: true }, { label: 'Forecast', color: t.s1, line: true },
        { label: '80% interval', color: t.band }, { label: '95% interval', color: t['band-2'] }]} />
    </div>
  )
}

/** Generic XY line chart for ROC / PR / calibration curves with an optional reference diagonal. */
export function CurveChart({ data, x, y, xLabel, yLabel, diagonal, height = 220, color }) {
  const t = useChartTheme()
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 14 }}>
        <CartesianGrid stroke={t.grid} />
        <XAxis type="number" dataKey={x} domain={[0, 1]} {...AX} stroke={t.axis} label={{ value: xLabel, position: 'insideBottom', offset: -8, fontSize: 11 }} />
        <YAxis type="number" domain={[0, 1]} {...AX} width={40} stroke={t.axis} />
        <Tooltip content={<ChartTooltip title={() => ''} rows={p => [[xLabel, fmtNum(p[0].payload[x])], [yLabel, fmtNum(p[0].payload[y])]]} />} />
        {diagonal && <ReferenceLine segment={[{ x: 0, y: diagonal === 'up' ? 0 : diagonal }, { x: 1, y: diagonal === 'up' ? 1 : diagonal }]} stroke={t.axis} strokeDasharray="4 4" />}
        <Line dataKey={y} stroke={color || t.s1} strokeWidth={2} dot={false} isAnimationActive={false} name={yLabel} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function ConfusionMatrix({ labels, matrix }) {
  const t = useChartTheme()
  const max = Math.max(1, ...matrix.flat())
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="table" style={{ width: 'auto' }} aria-label="Confusion matrix">
        <thead><tr><th scope="col">Actual ↓ / Predicted →</th>{labels.map(l => <th key={l} className="n" scope="col">{l}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row" style={{ position: 'static' }}>{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="n" style={{ background: sequentialColor(t, v / max * 0.85), color: v / max > 0.5 ? '#fff' : t.ink, fontWeight: i === j ? 600 : 400, minWidth: 64 }}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function XYScatter({ points, x, y, xLabel, yLabel, identity, zeroLine, height = 240 }) {
  const t = useChartTheme()
  const xs = points.map(p => p[x]), ys = points.map(p => p[y])
  const lo = Math.min(...xs, ...(identity ? ys : [])), hi = Math.max(...xs, ...(identity ? ys : []))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 14 }}>
        <CartesianGrid stroke={t.grid} />
        <XAxis type="number" dataKey={x} {...AX} stroke={t.axis} domain={['auto', 'auto']} tickFormatter={v => fmtNum(v, { compact: true })} label={{ value: xLabel, position: 'insideBottom', offset: -8, fontSize: 11 }} />
        <YAxis type="number" dataKey={y} {...AX} width={52} stroke={t.axis} domain={['auto', 'auto']} tickFormatter={v => fmtNum(v, { compact: true })} />
        <Tooltip content={<ChartTooltip title={() => ''} rows={p => [[xLabel, fmtNum(p[0].payload[x])], [yLabel, fmtNum(p[0].payload[y])]]} />} />
        {identity && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke={t.axis} strokeDasharray="4 4" />}
        {zeroLine && <ReferenceLine y={0} stroke={t.axis} strokeDasharray="4 4" />}
        <Scatter data={points} fill={t.s1} fillOpacity={0.5} isAnimationActive={false}
          shape={props => <circle cx={props.cx} cy={props.cy} r={3} fill={t.s1} fillOpacity={0.5} />} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function SimpleBars({ data, x, y, height = 220, color, yLabel }) {
  const t = useChartTheme()
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }} barCategoryGap={1}>
        <CartesianGrid vertical={false} stroke={t.grid} />
        <XAxis dataKey={x} {...AX} stroke={t.axis} interval="preserveStartEnd" minTickGap={18} />
        <YAxis {...AX} width={44} stroke={t.axis} />
        <Tooltip cursor={{ fill: t.band }} content={<ChartTooltip rows={p => [[yLabel || y, fmtNum(p[0].value)]]} />} />
        <Bar dataKey={y} fill={color || t.s1} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function HBar({ data, label, value, height, format = fmtNum, valueLabel = 'Value' }) {
  const t = useChartTheme()
  return (
    <ResponsiveContainer width="100%" height={height || Math.max(140, data.length * 26 + 24)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke={t.grid} />
        <XAxis type="number" {...AX} stroke={t.axis} tickFormatter={v => format(v)} />
        <YAxis type="category" dataKey={label} {...AX} width={130} stroke={t.axis} tickFormatter={v => (String(v).length > 20 ? String(v).slice(0, 19) + '…' : v)} />
        <Tooltip cursor={{ fill: t.band }} content={<ChartTooltip rows={p => [[valueLabel, format(p[0].value)]]} />} />
        <Bar dataKey={value} radius={[0, 3, 3, 0]} isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={d[value] < 0 ? t.s2 : t.s1} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
