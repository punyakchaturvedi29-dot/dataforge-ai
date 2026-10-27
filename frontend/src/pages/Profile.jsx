import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Download } from 'lucide-react'
import { api } from '../api/endpoints'
import { useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ChartBody, ChartCard } from '../components/charts/ChartView'
import { Badge, Button, Callout, DataTable, Empty, ErrorState, LoadingBlock, Metric, Panel, Segmented, Select, Tabs } from '../components/ui'
import { fmtNum, fmtP, fmtPct, plural } from '../lib/format'

export default function Profile() {
  const { id, dataKey } = useWorkspace()
  const { notify } = useUi()
  const [tab, setTab] = useState('columns')
  const { data: p, error, loading, reload } = useApi(() => api.profile(id), [id, dataKey])
  return (
    <div className="page">
      <PageHeader title="Profile" description="Descriptive statistics for every column, computed on the full working dataset."
        actions={<Button icon={Download} onClick={() => api.profileCsv(id).catch(e => notify(e.message, { kind: 'error' }))}>Statistics CSV</Button>} />
      {error && <ErrorState error={error} onRetry={reload} />}
      {!p && loading && <LoadingBlock rows={2} h={260} />}
      {p && (
        <>
          <div className="grid-4" style={{ marginBottom: 16 }}>
            <Metric label="Rows × columns" value={`${fmtNum(p.shape.rows)} × ${p.shape.columns}`} sub={`${p.memory_mb.toFixed(1)} MB in memory`} />
            <Metric label="Missing cells" value={fmtPct(p.missing.pct, 2)} sub={`${fmtNum(p.missing.cells)} cells, ${fmtNum(p.missing.rows_with_missing)} rows`} />
            <Metric label="Duplicate rows" value={fmtNum(p.duplicates.rows)} sub={p.duplicates.duplicate_columns.length ? `${p.duplicates.duplicate_columns.length} identical column group(s)` : 'no identical columns'} />
            <Metric label="Quality score" value={p.quality.score.toFixed(0)} sub={p.quality.label} help={p.quality.formula} />
          </div>
          <Tabs value={tab} onChange={setTab} tabs={[{ id: 'columns', label: 'Columns' }, { id: 'correlations', label: 'Correlations' },
            { id: 'missing', label: 'Missing values' }, { id: 'target', label: 'Target analysis' }]} />
          {tab === 'columns' && <Columns p={p} />}
          {tab === 'correlations' && <Correlations />}
          {tab === 'missing' && <Missing p={p} />}
          {tab === 'target' && <Target p={p} />}
        </>
      )}
    </div>
  )
}

function columnCharts(c) {
  const out = []
  const n = c.numeric
  if (n?.histogram) {
    const h = n.histogram
    const data = h.discrete
      ? h.values.map((v, i) => ({ label: fmtNum(v), value: v, count: h.counts[i] }))
      : h.counts.map((cnt, i) => ({ x0: h.edges[i], x1: h.edges[i + 1], label: `${fmtNum(h.edges[i])}–${fmtNum(h.edges[i + 1])}`, count: cnt }))
    out.push({ title: 'Distribution', chart: { type: 'histogram', x: c.name, data, discrete: h.discrete, notes: [] } })
    if (n.box?.n) out.push({ title: 'Spread and outliers', chart: { type: 'box', y: c.name, groups: [{ label: c.name, ...n.box }], notes: [] } })
  } else if (c.categorical?.top_values?.length) {
    const cat = c.categorical
    const data = cat.top_values.map(t => ({ label: t.value, value: t.count, n: t.count }))
    if (cat.other_count) data.push({ label: `Other (${cat.other_levels})`, value: cat.other_count, n: cat.other_count, other: true })
    out.push({ title: 'Most frequent values', chart: { type: 'bar', x: c.name, data, y_label: 'Rows', notes: cat.other_count ? [`${cat.other_levels} less frequent values are combined into “Other”.`] : [] } })
  }
  if (c.datetime?.counts_by_period) {
    const d = c.datetime.counts_by_period
    out.push({ title: 'Records per period', chart: { type: 'bar', x: c.name, data: d.labels.map((l, i) => ({ label: l, value: d.counts[i], n: d.counts[i] })), y_label: 'Rows', notes: [] } })
  }
  return out
}

function Columns({ p }) {
  const [params, setParams] = useSearchParams()
  const [filter, setFilter] = useState('')
  const [type, setType] = useState('all')
  const cols = p.columns.filter(c => (type === 'all' || c.logical_type === type) && c.name.toLowerCase().includes(filter.toLowerCase()))
  const selName = params.get('column') || (p.columns.find(c => c.logical_type === 'numeric') || p.columns[0])?.name
  const sel = p.columns.find(c => c.name === selName) || p.columns[0]
  const types = ['all', ...Object.keys(p.type_counts)]
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(220px, 300px) minmax(0, 1fr)', alignItems: 'start' }}>
      <Panel pad={false}>
        <div style={{ padding: 12 }} className="stack-sm">
          <input className="input" placeholder="Filter columns" aria-label="Filter columns" value={filter} onChange={e => setFilter(e.target.value)} />
          <Select label="Type" value={type} onChange={v => setType(v || 'all')} options={types.map(t => ({ value: t, label: t === 'all' ? 'All types' : `${t} (${p.type_counts[t]})` }))} />
        </div>
        <ul className="list-plain" style={{ maxHeight: 560, overflowY: 'auto', padding: '0 6px 8px' }} role="listbox" aria-label="Columns">
          {cols.map(c => (
            <li key={c.name}>
              <button type="button" role="option" aria-selected={c.name === sel.name} className="palette-item" style={{ width: '100%', border: 0, background: c.name === sel.name ? 'var(--accent-wash)' : 'none', textAlign: 'left' }}
                onClick={() => setParams({ column: c.name }, { replace: true })}>
                <span className="grow truncate">{c.name}</span>
                {c.missing_pct > 0 && <span className="tiny muted num">{c.missing_pct.toFixed(0)}% missing</span>}
                <Badge>{c.logical_type}</Badge>
              </button>
            </li>
          ))}
          {cols.length === 0 && <li className="small muted" style={{ padding: 12 }}>No columns match.</li>}
        </ul>
      </Panel>
      {sel && <ColumnDetail c={sel} />}
    </div>
  )
}

function ColumnDetail({ c }) {
  const n = c.numeric
  const cat = c.categorical
  const d = c.datetime
  const charts = columnCharts(c)
  const o = n?.outliers
  return (
    <div className="stack">
      <Panel title={c.name} subtitle={`${c.logical_type} · stored as ${c.dtype} · meaning from name: ${c.semantic_type}`}>
        <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
          <Metric label="Values" value={fmtNum(c.count)} />
          <Metric label="Missing" value={fmtNum(c.n_missing)} sub={fmtPct(c.missing_pct)} />
          <Metric label="Distinct" value={fmtNum(c.n_unique)} sub={`${c.unique_pct.toFixed(1)}% of values`} />
          {n && <Metric label="Mean / median" value={`${fmtNum(n.mean)} / ${fmtNum(n.median)}`} />}
          {cat && !n && <Metric label="Most frequent" value={<span className="truncate" style={{ display: 'block', fontSize: '1.1rem' }}>{cat.mode}</span>} sub={fmtPct(cat.mode_pct)} />}
          {d && <Metric label="Range" value={<span style={{ fontSize: '1rem' }}>{d.min.slice(0, 10)} → {d.max.slice(0, 10)}</span>} sub={`${fmtNum(d.span_days)} days`} />}
        </div>
      </Panel>
      {c.logical_type === 'identifier' && <Callout tone="info">This column looks like an identifier (unique per row). It is excluded from insights, charts and models; it is still useful as a join key.</Callout>}
      {charts.length > 0 && <div className="grid-2">{charts.map(ch => <ChartCard key={ch.title} title={ch.title} chart={ch.chart} height={220} />)}</div>}
      {n && (
        <div className="grid-2">
          <Panel title="Descriptive statistics">
            <dl className="kv">
              <dt>Mean</dt><dd>{fmtNum(n.mean)}</dd><dt>Median</dt><dd>{fmtNum(n.median)}</dd>
              <dt>Mode</dt><dd>{n.mode === null ? '— (all values unique)' : `${fmtNum(n.mode)} (${n.mode_count}×)`}</dd>
              <dt>Standard deviation</dt><dd>{fmtNum(n.std)}</dd><dt>Variance</dt><dd>{fmtNum(n.variance)}</dd>
              <dt>Coefficient of variation</dt><dd>{n.cv_pct === null ? '—' : fmtPct(n.cv_pct)}</dd>
              <dt>Minimum / maximum</dt><dd>{fmtNum(n.min)} / {fmtNum(n.max)}</dd><dt>Range</dt><dd>{fmtNum(n.range)}</dd>
              <dt>Q1 / Q3 (IQR)</dt><dd>{fmtNum(n.q1)} / {fmtNum(n.q3)} ({fmtNum(n.iqr)})</dd>
              <dt>Sum</dt><dd>{fmtNum(n.sum)}</dd><dt>Zeros / negatives</dt><dd>{fmtNum(n.zeros)} / {fmtNum(n.negatives)}</dd>
            </dl>
          </Panel>
          <Panel title="Shape and outliers">
            <dl className="kv">
              <dt>Shape</dt><dd>{n.shape}</dd>
              <dt>Skewness</dt><dd>{fmtNum(n.skewness)}</dd><dt>Excess kurtosis</dt><dd>{fmtNum(n.kurtosis)}</dd>
              <dt>Normality test</dt><dd>{n.normality?.test ? `${n.normality.test}: p ${fmtP(n.normality.p_value)}${n.normality.p_value < 0.05 ? ' (not normal)' : ''}` : 'not enough data'}</dd>
              <dt>Beyond 1.5×IQR</dt><dd>{fmtNum(o?.iqr_mild)}{o?.fences_mild && ` (outside ${fmtNum(o.fences_mild[0])} – ${fmtNum(o.fences_mild[1])})`}</dd>
              <dt>Beyond 3×IQR</dt><dd>{fmtNum(o?.iqr_extreme)}</dd><dt>Robust z &gt; 3.5</dt><dd>{fmtNum(o?.mad)}</dd>
            </dl>
            <h4 style={{ margin: '14px 0 6px' }}>Percentiles</h4>
            <DataTable columns={Object.keys(n.percentiles).map(k => ({ key: k, label: k.toUpperCase(), numeric: true }))} rows={[n.percentiles]} />
          </Panel>
        </div>
      )}
      {cat && !n && (
        <Panel title="Frequency table" subtitle={cat.other_levels ? `Top 20 of ${fmtNum(c.n_unique)} values` : undefined}>
          <DataTable columns={[{ key: 'value', label: 'Value' }, { key: 'count', label: 'Rows', numeric: true }, { key: 'pct', label: '%', numeric: true, render: v => v.toFixed(1) }]} rows={cat.top_values} />
          {cat.avg_length != null && <p className="small muted" style={{ marginTop: 8 }}>Text length: {fmtNum(cat.min_length)}–{fmtNum(cat.max_length)} characters, average {fmtNum(cat.avg_length)}.</p>}
        </Panel>
      )}
      {d && (
        <Panel title="Time coverage">
          <dl className="kv">
            <dt>Typical gap</dt><dd>{fmtNum(d.frequency.median_gap_days)} days (native frequency: {d.frequency.native_frequency})</dd>
            <dt>Distinct timestamps</dt><dd>{fmtNum(d.frequency.n_unique_timestamps)}</dd>
            <dt>Repeated timestamps</dt><dd>{fmtNum(d.duplicate_timestamps)}</dd>
          </dl>
        </Panel>
      )}
    </div>
  )
}

function Correlations() {
  const { id, dataKey } = useWorkspace()
  const [method, setMethod] = useState('spearman')
  const { data: c, error, loading, reload } = useApi(() => api.correlations(id), [id, dataKey])
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!c) return <LoadingBlock rows={1} h={300} />
  if (c.columns.length < 2) return <Empty title="Not enough numeric columns">Correlations need at least two numeric measure columns (identifiers are excluded).</Empty>
  return (
    <div className="stack">
      <ChartCard title="Correlation matrix" subtitle={c.notes[0]} actions={<Segmented label="Method" value={method} onChange={setMethod} options={[{ value: 'spearman', label: 'Spearman' }, { value: 'pearson', label: 'Pearson' }]} />}
        chart={{ type: 'heatmap', method, columns: c.columns, matrix: c[method], notes: c.truncated ? ['Showing the 30 most variable numeric columns.'] : [] }} />
      <Panel title="Tested pairs" subtitle={c.notes[1]}>
        <DataTable rowKey={r => r.a + r.b} columns={[
          { key: 'a', label: 'Column A' }, { key: 'b', label: 'Column B' },
          { key: 'pearson', label: 'Pearson r', numeric: true, render: v => fmtNum(v, { digits: 3 }) },
          { key: 'spearman', label: 'Spearman ρ', numeric: true, render: v => fmtNum(v, { digits: 3 }) },
          { key: 'ci95', label: '95% CI (r)', render: v => (v ? `${v[0].toFixed(2)} to ${v[1].toFixed(2)}` : '—') },
          { key: 'p_adjusted', label: 'Adjusted p', numeric: true, render: v => fmtP(v) },
          { key: 'n', label: 'n', numeric: true },
          { key: 'strength', label: 'Strength', render: (v, r) => <>{v}{r.nonlinear_hint && <Badge tone="info" title="Spearman is notably stronger than Pearson">non-linear?</Badge>}</> },
        ]} rows={c.pairs} />
      </Panel>
      <div className="grid-2">
        <Panel title="Categorical associations" subtitle="Bias-corrected Cramér's V (0 = independent, 1 = identical).">
          <DataTable columns={[{ key: 'a', label: 'A' }, { key: 'b', label: 'B' }, { key: 'v', label: "Cramér's V", numeric: true, render: v => fmtNum(v, { digits: 3 }) }, { key: 'p_value', label: 'p', numeric: true, render: fmtP }]}
            rows={c.categorical_associations} empty="No categorical pairs to compare." />
        </Panel>
        <Panel title="Measures by category" subtitle="η²: share of a measure's variance explained by a category.">
          <DataTable columns={[{ key: 'numeric', label: 'Measure' }, { key: 'categorical', label: 'Category' }, { key: 'eta_squared', label: 'η²', numeric: true, render: v => fmtNum(v, { digits: 3 }) }, { key: 'effect', label: 'Effect' }]}
            rows={c.numeric_by_category} empty="No measure/category pairs." />
        </Panel>
      </div>
    </div>
  )
}

function Missing({ p }) {
  const m = p.missing
  if (!m.cells) return <Empty title="No missing values">Every cell in the dataset has a value.</Empty>
  return (
    <div className="grid-2" style={{ alignItems: 'start' }}>
      <ChartCard title="Missing values by column" chart={{ type: 'missing', data: m.by_column.map(x => ({ label: x.column, pct: x.pct, count: x.missing })), notes: [] }} height={Math.max(200, m.by_column.length * 24)} />
      <div className="stack">
        <Panel title="Rows">
          <dl className="kv"><dt>Complete rows</dt><dd>{fmtNum(m.complete_rows)} ({fmtPct(m.complete_rows / p.shape.rows * 100)})</dd><dt>Rows with a missing value</dt><dd>{fmtNum(m.rows_with_missing)}</dd></dl>
        </Panel>
        <Panel title="Missing together" subtitle="Column pairs that tend to be missing in the same rows (Jaccard ≥ 0.5) — a hint that values are not missing at random.">
          <DataTable columns={[{ key: 'a', label: 'Column A' }, { key: 'b', label: 'Column B' }, { key: 'both_missing', label: 'Both missing', numeric: true }, { key: 'jaccard', label: 'Jaccard', numeric: true, render: v => v.toFixed(2) }]}
            rows={m.co_missing} empty="No columns are missing together." />
        </Panel>
      </div>
    </div>
  )
}

function Target({ p }) {
  const { id, dataKey, ws, refresh } = useWorkspace()
  const options = p.columns.filter(c => ['numeric', 'categorical', 'boolean'].includes(c.logical_type)).map(c => c.name)
  const [target, setTarget] = useState(ws.settings?.target && options.includes(ws.settings.target) ? ws.settings.target : '')
  const { data, error, loading } = useApi(() => api.targetAnalysis(id, target), [id, dataKey, target], { enabled: !!target })
  const choose = async v => {
    setTarget(v || '')
    if (v) { try { await api.patchWorkspace(id, { target: v }); refresh() } catch { /* non-critical */ } }
  }
  return (
    <div className="stack">
      <div className="form-row" style={{ maxWidth: 520 }}>
        <Select label="Target (outcome) column" value={target} onChange={choose} placeholder="Choose a column…" options={options}
          hint="The chosen target is remembered and used to prioritise insights and as the default model target." />
      </div>
      {!target && <Empty title="Choose a target">See how each column relates to the outcome you care about.</Empty>}
      {target && error && <ErrorState error={error} />}
      {target && loading && <LoadingBlock rows={1} h={240} />}
      {target && data && (
        <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.4fr)', alignItems: 'start' }}>
          <Panel title={data.task_hint === 'classification' ? 'Class balance' : 'Distribution'} subtitle={data.missing ? `${plural(data.missing, 'row')} without a target value` : undefined}>
            {data.class_balance ? (
              <>
                <ChartBody chart={{ type: 'bar', x: target, data: data.class_balance.map(c => ({ label: c.class, value: c.count, n: c.count })), y_label: 'Rows' }} height={200} />
                {data.minority_pct < 20 && <Callout tone="warn">The smallest class is {fmtPct(data.minority_pct)} of rows. Accuracy will be misleading; use balanced accuracy, recall and PR-AUC.</Callout>}
              </>
            ) : data.distribution && (
              <ChartBody chart={{ type: 'histogram', x: target, data: (() => { const h = data.distribution.histogram; return h.discrete ? h.values.map((v, i) => ({ label: fmtNum(v), count: h.counts[i] })) : h.counts.map((c, i) => ({ label: `${fmtNum(h.edges[i])}–${fmtNum(h.edges[i + 1])}`, count: c })) })() }} height={200} />
            )}
          </Panel>
          <Panel title="Association with the target" subtitle={data.note}>
            <DataTable columns={[{ key: 'feature', label: 'Column' }, { key: 'measure', label: 'Measure' }, { key: 'value', label: 'Value', numeric: true, render: v => fmtNum(v, { digits: 3 }) },
              { key: 'effect', label: 'Strength' }, { key: 'p_adjusted', label: 'Adjusted p', numeric: true, render: fmtP }]} rows={data.features} />
          </Panel>
        </div>
      )}
    </div>
  )
}
