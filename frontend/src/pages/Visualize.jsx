import { useEffect, useMemo, useState } from 'react'
import { LayoutDashboard, PencilRuler } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ChartCard } from '../components/charts/ChartView'
import { Badge, Button, Callout, Empty, ErrorState, LoadingBlock, Metric, Panel, Select, Tabs } from '../components/ui'
import { fmtNum } from '../lib/format'

export default function Visualize() {
  const [tab, setTab] = useState('auto')
  return (
    <div className="page">
      <PageHeader title="Visualize" description="Charts are chosen for each column's type and cardinality and aggregated on the server, so large datasets stay fast." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'auto', label: 'Auto dashboard', icon: LayoutDashboard }, { id: 'builder', label: 'Chart builder', icon: PencilRuler }]} />
      {tab === 'auto' ? <AutoDashboard /> : <Builder />}
    </div>
  )
}

function AutoDashboard() {
  const { id, dataKey } = useWorkspace()
  const { data, error, loading, reload } = useApi(() => api.dashboard(id), [id, dataKey])
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data) return <LoadingBlock rows={2} h={260} />
  return (
    <div className="stack">
      <div className="row-between">
        <div><h2>{data.domain_description}</h2><p className="small muted">{data.domain_detail.note}{data.domain_detail.matched_terms.length ? ` Matched: ${data.domain_detail.matched_terms.join(', ')}.` : ''}</p></div>
        <Badge tone="accent">{data.charts.length} charts</Badge>
      </div>
      <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))' }}>
        {data.kpis.map(k => <Metric key={k.label} label={k.label} value={fmtNum(k.value, { compact: true })} sub={k.trend != null ? `${k.trend > 0 ? '▲' : '▼'} ${Math.abs(k.trend).toFixed(1)}% vs previous period` : k.n ? `n = ${fmtNum(k.n)}` : null} />)}
      </div>
      {data.charts.length === 0 ? <Empty title="No charts could be generated">The dataset needs at least one numeric, categorical or date column.</Empty> : (
        <div className="grid-auto" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))' }}>
          {data.charts.map((c, i) => <ChartCard key={i} chart={c} height={c.type === 'heatmap' ? 260 : 230} />)}
        </div>
      )}
    </div>
  )
}

const TYPES = [
  { value: 'bar', label: 'Bar — compare categories' }, { value: 'histogram', label: 'Histogram — distribution of a number' },
  { value: 'line', label: 'Line — change over time' }, { value: 'scatter', label: 'Scatter — relationship of two numbers' },
  { value: 'box', label: 'Box plot — spread by group' }, { value: 'heatmap', label: 'Heatmap — correlation matrix' },
  { value: 'pie', label: 'Donut — share of a total (≤ 6 parts)' }, { value: 'missing', label: 'Missing values by column' },
]

function Builder() {
  const { id, dataKey } = useWorkspace()
  const { data: sch } = useApi(() => api.schema(id), [id, dataKey])
  const [spec, setSpec] = useState({ type: 'bar', agg: 'count' })
  const [chart, setChart] = useState(null)
  const build = useAction(async s => setChart(await api.chart(id, s)))
  const cols = sch?.columns || []
  const by = t => cols.filter(c => t.includes(c.logical_type)).map(c => c.name)
  const numeric = by(['numeric'])
  const cats = cols.filter(c => ['categorical', 'boolean'].includes(c.logical_type) && c.n_unique <= 50).map(c => c.name)
  const dates = by(['datetime'])
  const set = (k, v) => setSpec(s => ({ ...s, [k]: v || undefined }))
  const clean = useMemo(() => Object.fromEntries(Object.entries(spec).filter(([, v]) => v !== undefined && v !== null && v !== '')), [spec])
  const t = spec.type
  useEffect(() => { setChart(null); build.setError(null) }, [t]) // eslint-disable-line react-hooks/exhaustive-deps
  const ready = t === 'heatmap' || t === 'missing' || (t === 'histogram' && spec.x) || (['bar', 'pie'].includes(t) && spec.x) || (t === 'line' && spec.x) || (t === 'scatter' && spec.x && spec.y) || (t === 'box' && spec.y)
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(260px, 340px) minmax(0, 1fr)', alignItems: 'start' }}>
      <Panel title="Chart settings">
        <div className="stack-sm">
          <Select label="Chart type" value={t} onChange={v => setSpec({ type: v, agg: v === 'bar' ? 'count' : undefined })} options={TYPES} />
          {t === 'histogram' && <Select label="Numeric column" value={spec.x} onChange={v => set('x', v)} placeholder="Choose…" options={numeric} />}
          {(t === 'bar' || t === 'pie') && <>
            <Select label="Category" value={spec.x} onChange={v => set('x', v)} placeholder="Choose…" options={cats} />
            <Select label="Measure (optional)" value={spec.y} onChange={v => setSpec(s => ({ ...s, y: v || undefined, agg: v ? (s.agg === 'count' ? (t === 'pie' ? 'sum' : 'mean') : s.agg) : 'count' }))} placeholder="Count rows" options={numeric} />
            {spec.y && t === 'bar' && <Select label="Aggregation" value={spec.agg} onChange={v => set('agg', v)} options={['mean', 'median', 'sum', 'min', 'max']} />}
          </>}
          {t === 'line' && <>
            <Select label="Date column" value={spec.x} onChange={v => set('x', v)} placeholder="Choose…" options={dates} hint={!dates.length ? 'No date columns. Convert text dates in the Clean step.' : undefined} />
            <Select label="Measure (optional)" value={spec.y} onChange={v => set('y', v)} placeholder="Count records" options={numeric} />
            {spec.y && <Select label="Aggregation" value={spec.agg} onChange={v => set('agg', v)} placeholder="Automatic (sum or mean)" options={['sum', 'mean', 'median', 'min', 'max', 'count']} />}
            <Select label="Frequency" value={spec.freq} onChange={v => set('freq', v)} placeholder="Automatic" options={[{ value: 'D', label: 'Day' }, { value: 'W', label: 'Week' }, { value: 'MS', label: 'Month' }, { value: 'QS', label: 'Quarter' }, { value: 'YS', label: 'Year' }]} />
            <Select label="Split by (optional)" value={spec.group} onChange={v => set('group', v)} placeholder="No split" options={cats} />
          </>}
          {t === 'scatter' && <>
            <Select label="X axis" value={spec.x} onChange={v => set('x', v)} placeholder="Choose…" options={numeric} />
            <Select label="Y axis" value={spec.y} onChange={v => set('y', v)} placeholder="Choose…" options={numeric} />
            <Select label="Colour by (optional, up to 3 groups)" value={spec.color} onChange={v => set('color', v)} placeholder="None" options={cats} />
          </>}
          {t === 'box' && <>
            <Select label="Measure" value={spec.y} onChange={v => set('y', v)} placeholder="Choose…" options={numeric} />
            <Select label="Group by (optional)" value={spec.x} onChange={v => set('x', v)} placeholder="No grouping" options={cats} />
          </>}
          {t === 'heatmap' && <Select label="Method" value={spec.method || 'pearson'} onChange={v => set('method', v)} options={[{ value: 'pearson', label: 'Pearson (linear)' }, { value: 'spearman', label: 'Spearman (rank)' }]} />}
          <div><Button variant="primary" busy={build.busy} disabled={!ready} onClick={() => build.run(clean).catch(() => {})}>Draw chart</Button></div>
        </div>
      </Panel>
      <div className="stack">
        {build.error && <Callout tone="warn" title="This chart was not drawn">{build.error.message}</Callout>}
        {chart ? <ChartCard chart={chart} height={340} /> : !build.error && <Panel><Empty icon={PencilRuler} title="Choose what to plot">The builder blocks charts that would mislead, such as pies of negative values or bar charts of continuous numbers.</Empty></Panel>}
      </div>
    </div>
  )
}
