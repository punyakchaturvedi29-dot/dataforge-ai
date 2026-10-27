import { useEffect, useState } from 'react'
import { CalendarX2, TrendingUp } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ForecastChart } from '../components/charts/ChartView'
import { Badge, Button, Callout, DataTable, Empty, ErrorState, Input, LoadingBlock, Metric, Panel, Select } from '../components/ui'
import { fmtNum, fmtPct } from '../lib/format'

export default function Forecast() {
  const { id, dataKey } = useWorkspace()
  const { data: opts, error, loading } = useApi(() => api.forecastOptions(id), [id, dataKey])
  const { data: latest } = useApi(() => api.latest(id, 'forecast').catch(() => null), [id])
  const [cfg, setCfg] = useState({ horizon: 6, frequency: 'auto', fill: 'auto' })
  const [res, setRes] = useState(null)
  const [method, setMethod] = useState(null)
  useEffect(() => {
    if (opts?.available && !cfg.date_column) setCfg(c => ({ ...c, date_column: opts.date_columns[0].column, target: opts.measures[0]?.column || '' }))
  }, [opts]) // eslint-disable-line
  useEffect(() => { if (latest && !res && !latest.stale) { setRes(latest.result); setMethod(latest.result.selected_method) } }, [latest]) // eslint-disable-line
  const run = useAction(async () => {
    const body = { date_column: cfg.date_column, horizon: Number(cfg.horizon), frequency: cfg.frequency, fill: cfg.fill }
    if (cfg.target) body.target = cfg.target
    if (cfg.aggregation) body.aggregation = cfg.aggregation
    const r = await api.forecast(id, body)
    setRes(r)
    setMethod(r.selected_method || null)
  })
  if (error) return <div className="page"><PageHeader title="Forecast" /><ErrorState error={error} /></div>
  if (!opts && loading) return <div className="page"><PageHeader title="Forecast" /><LoadingBlock rows={2} /></div>
  const measure = opts.measures.find(m => m.column === cfg.target)
  return (
    <div className="page">
      <PageHeader title="Forecast" description="Methods are compared with rolling-origin backtests that train only on the past, then the most accurate one projects the series with approximate prediction intervals." />
      {!opts.available ? <Empty icon={CalendarX2} title="Forecasting is not available for this dataset">{opts.reason}</Empty> : (
        <div className="stack">
          <Panel title="Series to forecast">
            <div className="form-row">
              <Select label="Date column" value={cfg.date_column} onChange={v => setCfg(c => ({ ...c, date_column: v }))} options={opts.date_columns.map(d => ({ value: d.column, label: `${d.column} (${fmtNum(d.span_days)} days)` }))} />
              <Select label="Measure" value={cfg.target} onChange={v => setCfg(c => ({ ...c, target: v || '' }))} placeholder="Number of records" options={opts.measures.map(m => m.column)} />
              <Select label="Aggregate per period" value={cfg.aggregation} onChange={v => setCfg(c => ({ ...c, aggregation: v }))}
                placeholder={`Automatic (${measure ? measure.default_agg : 'count'})`} options={cfg.target ? ['sum', 'mean', 'median', 'last', 'min', 'max'] : ['count']} />
              <Select label="Frequency" value={cfg.frequency} onChange={v => setCfg(c => ({ ...c, frequency: v || 'auto' }))} options={[{ value: 'auto', label: 'Automatic' }, { value: 'D', label: 'Daily' }, { value: 'W', label: 'Weekly' }, { value: 'MS', label: 'Monthly' }, { value: 'QS', label: 'Quarterly' }, { value: 'YS', label: 'Yearly' }]} />
              <Input label="Horizon (periods)" type="number" min={1} max={365} value={cfg.horizon} onChange={e => setCfg(c => ({ ...c, horizon: e.target.value }))} />
              <Select label="Periods without data" value={cfg.fill} onChange={v => setCfg(c => ({ ...c, fill: v || 'auto' }))} options={[{ value: 'auto', label: 'Automatic (0 for totals, interpolate means)' }, { value: 'zero', label: 'Treat as zero' }, { value: 'interpolate', label: 'Interpolate' }, { value: 'ffill', label: 'Carry last value forward' }]} />
            </div>
            <div style={{ marginTop: 12 }}><Button variant="primary" icon={TrendingUp} busy={run.busy} disabled={!cfg.date_column || !(cfg.horizon >= 1)} onClick={() => run.run().catch(() => {})}>Run forecast</Button></div>
            {run.error && <div style={{ marginTop: 10 }}><ErrorState error={run.error} compact /></div>}
          </Panel>
          {run.busy && <LoadingBlock rows={2} h={160} />}
          {res && !run.busy && (!res.available ? <Callout tone="warn" title="Not enough history">{res.reason}</Callout> : <Result r={res} method={method || res.selected_method} setMethod={setMethod} />)}
        </div>
      )}
    </div>
  )
}

function Result({ r, method, setMethod }) {
  const sel = r.comparison.find(c => c.method === r.selected_method)
  const shown = r.comparison.find(c => c.method === method)
  const f = r.forecasts[method]
  const d = r.diagnostics
  return (
    <div className="stack">
      {r.warnings.length > 0 && <Callout tone="warn" title="Check before relying on this forecast"><ul style={{ margin: 0, paddingLeft: 18 }}>{r.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Callout>}
      <div className="grid-4">
        <Metric label="Selected method" value={<span style={{ fontSize: '1.1rem' }}>{r.selected_label}</span>} sub={`${r.backtest.n_folds} backtest folds`} />
        <Metric label="Scaled error (MASE)" value={fmtNum(sel.mase, { digits: 2 })} sub="below 1 beats a naive forecast" help="Mean absolute scaled error: forecast error divided by the in-sample error of a naive (seasonal) forecast." />
        <Metric label="Skill vs naive" value={sel.skill_vs_naive_pct == null ? '—' : `${sel.skill_vs_naive_pct.toFixed(0)}%`} sub="improvement in MASE" />
        <Metric label="80% interval coverage" value={sel.coverage80 == null ? '—' : fmtPct(sel.coverage80 * 100, 0)} sub="share of backtest actuals inside" />
      </div>
      <Panel title={`${r.aggregation} of ${r.target || 'records'} per ${r.frequency_label}`} subtitle={`${r.history.length} periods of history · horizon ${r.horizon}${r.seasonal_period ? ` · seasonal period ${r.seasonal_period}` : ''}`}
        actions={<Select label="Method" value={method} onChange={setMethod} options={r.comparison.filter(c => c.status === 'ok' && r.forecasts[c.method]).map(c => ({ value: c.method, label: c.label + (c.selected ? ' (selected)' : '') }))} />}>
        <ForecastChart fc={r} method={method} />
        {r.non_negative_clipping && <p className="tiny muted">The series is never negative, so forecasts and lower bounds are clipped at zero.</p>}
      </Panel>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Panel title="Method comparison" subtitle="Average error over the backtest folds; lower is better.">
          <DataTable rowKey={r => r.method} columns={[
            { key: 'label', label: 'Method', render: (v, row) => <span title={row.description}>{v}{row.selected && <Badge tone="accent">selected</Badge>}</span> },
            { key: 'mase', label: 'MASE', numeric: true, render: v => fmtNum(v, { digits: 3 }) },
            { key: 'rmse', label: 'RMSE', numeric: true, render: v => fmtNum(v) },
            { key: 'smape', label: 'sMAPE', numeric: true, render: v => (v == null ? '—' : fmtPct(v)) },
            { key: 'coverage80', label: '80% cov.', numeric: true, render: v => (v == null ? '—' : fmtPct(v * 100, 0)) },
            { key: 'status', label: 'Status', render: v => (v === 'ok' ? 'evaluated' : 'not applicable') },
          ]} rows={r.comparison} />
        </Panel>
        <Panel title="How to read this">
          <ul style={{ margin: 0, paddingLeft: 18 }} className="small">{r.explanation.map((e, i) => <li key={i}>{e}</li>)}</ul>
          <h4 style={{ margin: '12px 0 6px' }}>Limitations</h4>
          <ul style={{ margin: 0, paddingLeft: 18 }} className="small secondary">{r.limitations.map((e, i) => <li key={i}>{e}</li>)}</ul>
          <h4 style={{ margin: '12px 0 6px' }}>Series diagnostics</h4>
          <dl className="kv small">
            <dt>Periods</dt><dd>{d.periods}</dd><dt>Periods without records</dt><dd>{d.empty_periods} ({d.fill_policy})</dd>
            <dt>Timestamps with several records</dt><dd>{fmtNum(d.duplicate_timestamps)}</dd>
            <dt>Rows without a date</dt><dd>{fmtNum(d.rows_missing_date)}</dd>
            {d.rows_missing_value != null && <><dt>Rows without a value</dt><dd>{fmtNum(d.rows_missing_value)}</dd></>}
          </dl>
        </Panel>
      </div>
      {f && (
        <Panel title={`Forecast values (${shown?.label})`}>
          <DataTable columns={[{ key: 't', label: 'Period', render: v => v.slice(0, 10) }, { key: 'f', label: 'Forecast', numeric: true }, { key: 'l80', label: '80% low', numeric: true },
            { key: 'h80', label: '80% high', numeric: true }, { key: 'l95', label: '95% low', numeric: true }, { key: 'h95', label: '95% high', numeric: true }]}
            rows={r.future_index.map((t, i) => ({ t, f: f.forecast[i], l80: f.lower80[i], h80: f.upper80[i], l95: f.lower95[i], h95: f.upper95[i] }))} />
        </Panel>
      )}
    </div>
  )
}
