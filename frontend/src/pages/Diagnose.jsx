import { useEffect, useState } from 'react'
import { Target } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { DiagnosticCard } from '../components/Findings'
import { HBar } from '../components/charts/ChartView'
import { Badge, Button, Callout, Check, DataTable, Empty, ErrorState, Input, LoadingBlock, Metric, Panel, Select, Tabs } from '../components/ui'
import { fmtNum, fmtP, fmtPct } from '../lib/format'

export default function Diagnose() {
  const { id, dataKey } = useWorkspace()
  const { data: sch } = useApi(() => api.schema(id), [id, dataKey])
  const { data: latest } = useApi(() => api.latest(id, 'diagnostics').catch(() => null), [id])
  const [cfg, setCfg] = useState({ outcome: '', positive_class: '', custom: false })
  const [res, setRes] = useState(null)
  const [tab, setTab] = useState('findings')
  const run = useAction(async () => {
    const body = { outcome: cfg.outcome }
    if (cfg.positive_class) body.positive_class = cfg.positive_class
    if (cfg.custom && cfg.a0 && cfg.a1 && cfg.b0 && cfg.b1) { body.period_a = [cfg.a0, cfg.a1]; body.period_b = [cfg.b0, cfg.b1] }
    setRes(await api.diagnostics(id, body))
  })
  useEffect(() => { if (latest && !res && !latest.stale) { setRes(latest.result); setCfg(c => ({ ...c, outcome: latest.params.outcome, positive_class: latest.params.positive_class || '' })) } }, [latest]) // eslint-disable-line
  const cols = sch?.columns || []
  const outcomes = cols.filter(c => ['numeric', 'categorical', 'boolean'].includes(c.logical_type) && (c.logical_type === 'numeric' || c.n_unique <= 30)).map(c => c.name)
  const needLevels = run.error?.code === 'needs_positive_class' ? run.error.details.levels : null
  const hasDate = cols.some(c => c.logical_type === 'datetime')
  return (
    <div className="page">
      <PageHeader title="Diagnose" description="Investigate what is associated with an outcome. Results are labelled as observations, potential drivers or hypotheses — observational data cannot prove causes." />
      <Panel title="Outcome to investigate" style={{ marginBottom: 16 }}>
        <div className="form-row">
          <Select label="Outcome" value={cfg.outcome} onChange={v => { setCfg(c => ({ ...c, outcome: v || '', positive_class: '' })); run.setError(null) }} placeholder="Choose a column…" options={outcomes} />
          {(needLevels || cfg.positive_class) && <Select label="Count as the outcome" value={cfg.positive_class} onChange={v => setCfg(c => ({ ...c, positive_class: v || '' }))} placeholder="Choose a value…" options={needLevels || [cfg.positive_class]} hint="For categories, the share of rows with this value is analysed." />}
          <Button variant="primary" icon={Target} busy={run.busy} disabled={!cfg.outcome || (needLevels && !cfg.positive_class)} onClick={() => run.run().catch(() => {})}>Run diagnostics</Button>
        </div>
        {hasDate && <div style={{ marginTop: 10 }}><Check label="Compare two specific periods (default: last complete month vs the month before)" checked={cfg.custom} onChange={v => setCfg(c => ({ ...c, custom: v }))} /></div>}
        {cfg.custom && (
          <div className="form-row" style={{ marginTop: 8 }}>
            <Input label="Period A from" type="date" value={cfg.a0 || ''} onChange={e => setCfg(c => ({ ...c, a0: e.target.value }))} />
            <Input label="Period A to" type="date" value={cfg.a1 || ''} onChange={e => setCfg(c => ({ ...c, a1: e.target.value }))} />
            <Input label="Period B from" type="date" value={cfg.b0 || ''} onChange={e => setCfg(c => ({ ...c, b0: e.target.value }))} />
            <Input label="Period B to" type="date" value={cfg.b1 || ''} onChange={e => setCfg(c => ({ ...c, b1: e.target.value }))} />
          </div>
        )}
        {run.error && !needLevels && <div style={{ marginTop: 10 }}><ErrorState error={run.error} compact /></div>}
        {needLevels && !cfg.positive_class && <p className="small" style={{ marginTop: 8 }}>“{cfg.outcome}” has several values. Choose which value counts as the outcome.</p>}
      </Panel>
      {run.busy && <LoadingBlock rows={3} h={100} />}
      {!res && !run.busy && <Empty icon={Target} title="Choose an outcome">For example revenue, delivery time or churned = yes.</Empty>}
      {res && !run.busy && <Results r={res} tab={tab} setTab={setTab} />}
    </div>
  )
}

function Results({ r, tab, setTab }) {
  const s = r.summary
  return (
    <div className="stack">
      <Callout tone="info">{r.causal_note}</Callout>
      <div className="grid-4">
        <Metric label="Outcome" value={<span style={{ fontSize: '1.05rem' }}>{r.outcome_kind === 'rate' ? r.outcome_description : `${r.additive ? 'total' : 'mean'} ${r.outcome}`}</span>} />
        <Metric label={r.outcome_kind === 'rate' ? 'Rate' : 'Mean'} value={r.outcome_kind === 'rate' ? fmtPct(s.mean * 100) : fmtNum(s.mean)} sub={s.ci95 ? `95% CI ${fmtNum(s.ci95[0])} – ${fmtNum(s.ci95[1])}` : null} />
        <Metric label="Median" value={fmtNum(s.median)} sub={`std ${fmtNum(s.std)}`} />
        <Metric label="Rows analysed" value={fmtNum(s.n)} />
      </div>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'findings', label: 'Findings', badge: r.findings.length }, { id: 'segments', label: 'Segments' },
        { id: 'period', label: 'Period change' }, { id: 'numeric', label: 'Relationships' }, { id: 'model', label: 'Driver model' }, { id: 'contrast', label: 'High vs low' }, { id: 'anomalies', label: 'Anomalies' }]} />
      {tab === 'findings' && (r.findings.length ? <div className="stack-sm">{r.findings.map((f, i) => <DiagnosticCard key={i} f={f} />)}</div>
        : <Empty title="No supported findings">None of the checks found a statistically supported difference or relationship for this outcome.</Empty>)}
      {tab === 'segments' && <Segments segs={r.segments} kind={r.outcome_kind} />}
      {tab === 'period' && <Period p={r.period_change} />}
      {tab === 'numeric' && (
        <Panel title="Numeric relationships" subtitle="Spearman rank correlation with the outcome; partial correlation controls for the other strongest numeric columns.">
          <DataTable columns={[{ key: 'feature', label: 'Column' }, { key: 'spearman', label: 'ρ', numeric: true, render: v => v.toFixed(3) },
            { key: 'ci95', label: '95% CI', render: v => (v ? `${v[0].toFixed(2)} to ${v[1].toFixed(2)}` : '—') }, { key: 'p_adjusted', label: 'Adjusted p', numeric: true, render: fmtP },
            { key: 'partial_spearman', label: 'Partial ρ', numeric: true, render: v => (v == null ? '—' : v.toFixed(3)) }, { key: 'partial_note', label: 'After controls', render: v => v || '—' }]}
            rows={r.numeric_relationships} empty="No numeric columns besides the outcome." />
        </Panel>
      )}
      {tab === 'model' && (r.driver_model.available ? (
        <Panel title={`Driver model (${r.driver_model.metric} ${r.driver_model.cv_score.toFixed(3)} ± ${r.driver_model.cv_std.toFixed(3)})`} subtitle={r.driver_model.interpretation}>
          <HBar data={r.driver_model.importance} label="feature" value="importance" valueLabel="Importance" format={v => fmtNum(v, { digits: 3 })} />
          <p className="small muted">{r.driver_model.notes.join(' ')}</p>
        </Panel>
      ) : <Callout tone="info">{r.driver_model.reason}</Callout>)}
      {tab === 'contrast' && (r.high_vs_low.available ? (
        <div className="grid-2">
          <Panel title="Numeric differences" subtitle={`Top quartile (≥ ${fmtNum(r.high_vs_low.threshold_high)}, ${r.high_vs_low.rows_high} rows) vs bottom quartile (≤ ${fmtNum(r.high_vs_low.threshold_low)}).`}>
            <DataTable columns={[{ key: 'feature', label: 'Column' }, { key: 'mean_high', label: 'Mean (high)', numeric: true }, { key: 'mean_low', label: 'Mean (low)', numeric: true },
              { key: 'smd', label: 'SMD', numeric: true, render: v => (v == null ? '—' : v.toFixed(2)) }, { key: 'effect', label: 'Effect' }]} rows={r.high_vs_low.numeric} />
          </Panel>
          <Panel title="Over-represented categories" subtitle="Share of rows in each group (percentage points).">
            <DataTable columns={[{ key: 'dimension', label: 'Column' }, { key: 'level', label: 'Value' }, { key: 'share_high', label: 'High', numeric: true, render: v => fmtPct(v * 100) },
              { key: 'share_low', label: 'Low', numeric: true, render: v => fmtPct(v * 100) }, { key: 'difference_pp', label: 'Difference', numeric: true, render: v => `${v > 0 ? '+' : ''}${v.toFixed(1)} pp` }]} rows={r.high_vs_low.categorical} />
          </Panel>
        </div>
      ) : <Callout tone="info">{r.high_vs_low.reason}</Callout>)}
      {tab === 'anomalies' && (
        <div className="grid-2">
          <Panel title="Unusual records" subtitle={r.anomalies.records?.method}>
            {r.anomalies.records ? <DataTable columns={[{ key: 'row', label: 'Row', numeric: true, render: v => Number(v) + 1 }, { key: 'value', label: 'Value', numeric: true }, { key: 'robust_z', label: 'Robust z', numeric: true, render: v => v.toFixed(1) }]} rows={r.anomalies.records.top} empty="None found." />
              : <p className="small muted">Not applicable for this outcome.</p>}
          </Panel>
          <Panel title="Unusual periods" subtitle={r.anomalies.periods?.method}>
            {r.anomalies.periods ? <DataTable columns={[{ key: 'period', label: 'Period', render: v => v.slice(0, 10) }, { key: 'value', label: 'Value', numeric: true }, { key: 'robust_z', label: 'Robust z', numeric: true, render: v => v.toFixed(1) }]} rows={r.anomalies.periods.items} empty="None found." />
              : <p className="small muted">Needs a date column with at least 8 periods.</p>}
          </Panel>
        </div>
      )}
    </div>
  )
}

function Segments({ segs, kind }) {
  const [sel, setSel] = useState(segs[0]?.dimension)
  const s = segs.find(x => x.dimension === sel)
  if (!segs.length) return <Empty title="No categorical dimensions">The dataset has no categorical columns with 2–50 values to compare.</Empty>
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(240px, 320px) minmax(0, 1fr)', alignItems: 'start' }}>
      <Panel title="Dimensions" subtitle="Sorted by variance explained (η²).">
        <ul className="list-plain stack-sm">
          {segs.map(x => (
            <li key={x.dimension}>
              <button type="button" className="palette-item" aria-pressed={sel === x.dimension} style={{ width: '100%', border: 0, background: sel === x.dimension ? 'var(--accent-wash)' : 'none', textAlign: 'left' }} onClick={() => setSel(x.dimension)}>
                <span className="grow">{x.dimension}</span>
                <span className="tiny num">η² {x.eta_squared == null ? '—' : x.eta_squared.toFixed(3)}</span>
                {x.significant ? <Badge tone="info">p {fmtP(x.p_adjusted)}</Badge> : <Badge>n.s.</Badge>}
              </button>
            </li>
          ))}
        </ul>
      </Panel>
      {s && (
        <div className="stack">
          <Panel title={`${kind === 'rate' ? 'Rate' : 'Mean'} by ${s.dimension}`} subtitle={`η² = ${s.eta_squared?.toFixed(3)} (${s.effect}); Kruskal–Wallis adjusted p = ${fmtP(s.p_adjusted)}`}>
            <HBar data={s.table.slice(0, 15).map(t => ({ ...t, value: kind === 'rate' ? t.mean * 100 : t.mean }))} label="level" value="value" valueLabel={kind === 'rate' ? 'Rate %' : 'Mean'} />
            <DataTable columns={[{ key: 'level', label: 'Group' }, { key: 'n', label: 'Rows', numeric: true }, { key: 'mean', label: kind === 'rate' ? 'Rate' : 'Mean', numeric: true, render: v => (kind === 'rate' ? fmtPct(v * 100) : fmtNum(v)) },
              { key: 'ci95', label: '95% CI', render: v => (v ? `${fmtNum(v[0])} – ${fmtNum(v[1])}` : '—') }, { key: 'median', label: 'Median', numeric: true },
              { key: 'share_of_total_pct', label: 'Share of total', numeric: true, render: v => (v == null ? '—' : fmtPct(v)) }]} rows={s.table} maxHeight={260} />
          </Panel>
          {s.confounding_check && (
            <Callout tone={s.confounding_check.adjusted_to_raw_ratio != null && s.confounding_check.adjusted_to_raw_ratio < 0.5 ? 'warn' : 'info'} title="Confounding check">
              {s.confounding_check.verdict} {s.confounding_check.method}
              {s.confounding_check.gap_adjusted != null && <span className="num"> Raw gap {fmtNum(s.confounding_check.gap_raw)}, adjusted {fmtNum(s.confounding_check.gap_adjusted)}.</span>}
            </Callout>
          )}
        </div>
      )}
    </div>
  )
}

function Period({ p }) {
  if (!p) return <Empty title="No date column">Period comparisons need a date/time column.</Empty>
  if (!p.available) return <Callout tone="info">{p.reason}</Callout>
  return <PeriodBody p={p} />
}

function PeriodBody({ p }) {
  const [dim, setDim] = useState(p.contributions[0]?.dimension)
  const c = p.contributions.find(x => x.dimension === dim)
  return (
    <div className="stack">
      <div className="grid-4">
        <Metric label={`Period A (${p.period_a.start.slice(0, 10)} – ${p.period_a.end.slice(0, 10)})`} value={fmtNum(p.period_a.value)} sub={`${fmtNum(p.period_a.rows)} rows`} />
        <Metric label={`Period B (${p.period_b.start.slice(0, 10)} – ${p.period_b.end.slice(0, 10)})`} value={fmtNum(p.period_b.value)} sub={`${fmtNum(p.period_b.rows)} rows`} />
        <Metric label={`Change in ${p.statistic}`} value={`${p.change_pct > 0 ? '+' : ''}${p.change_pct?.toFixed(1)}%`} sub={`rows ${p.row_count_change_pct > 0 ? '+' : ''}${p.row_count_change_pct.toFixed(0)}%`} />
        <Metric label="Is the change real?" value={p.test.mann_whitney_p != null && p.test.mann_whitney_p < 0.05 ? 'Likely' : 'Unclear'} sub={`Mann–Whitney p = ${fmtP(p.test.mann_whitney_p)}`} />
      </div>
      <Panel title="What contributed to the change" subtitle={p.method} actions={<Select label="Dimension" value={dim} onChange={setDim} options={p.contributions.map(x => x.dimension)} />}>
        {c && <>
          <HBar data={c.rows} label="level" value="contribution" valueLabel="Contribution" />
          <DataTable columns={[{ key: 'level', label: 'Group' }, { key: 'rows_a', label: 'Rows A', numeric: true }, { key: 'rows_b', label: 'Rows B', numeric: true },
            { key: 'value_a', label: 'Value A', numeric: true }, { key: 'value_b', label: 'Value B', numeric: true }, { key: 'contribution', label: 'Contribution', numeric: true },
            { key: 'mix_effect', label: 'Mix effect', numeric: true }, { key: 'rate_effect', label: 'Rate effect', numeric: true },
            { key: 'share_of_change_pct', label: 'Share of change', numeric: true, render: v => (v == null ? '—' : fmtPct(v, 0)) }]} rows={c.rows} maxHeight={300} />
        </>}
      </Panel>
    </div>
  )
}
