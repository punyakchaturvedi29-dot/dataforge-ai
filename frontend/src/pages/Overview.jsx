import { Link } from 'react-router-dom'
import { Activity, AlertTriangle, ArrowRight, BrainCircuit, TrendingUp } from 'lucide-react'
import { api } from '../api/endpoints'
import { useApi } from '../hooks/useApi'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { FindingCard } from '../components/Findings'
import { Badge, Callout, Empty, ErrorState, Help, LoadingBlock, Metric, Panel } from '../components/ui'
import { fmtNum, fmtPct, plural, timeAgo } from '../lib/format'

const SEV = { high: 'bad', medium: 'warn', low: undefined, info: 'info' }

export default function Overview() {
  const { id, dataKey, ws } = useWorkspace()
  const { data: o, error, loading, reload } = useApi(() => api.overview(id), [id, dataKey])
  if (loading && !o) return <div className="page"><PageHeader title="Overview" /><LoadingBlock rows={3} /></div>
  if (error) return <div className="page"><PageHeader title="Overview" /><ErrorState error={error} onRetry={reload} /></div>
  const q = o.quality
  const comp = q.components
  return (
    <div className="page">
      <PageHeader title={o.dataset.name}
        description={`${plural(o.shape.rows, 'row')} × ${o.shape.columns} columns${o.dataset.n_ops ? ` · ${plural(o.dataset.n_ops, 'cleaning step')} applied to a copy of the original` : ' · original data, no cleaning applied yet'}`} />

      {o.ingestion_warnings?.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Callout tone="warn" title="Notes from loading the file">
            <ul style={{ margin: 0, paddingLeft: 18 }}>{o.ingestion_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </Callout>
        </div>
      )}

      <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', marginBottom: 16 }}>
        <Metric label="Data quality" value={`${q.score.toFixed(0)}`} sub={`${q.label} · out of 100`}
          help={q.formula} tone={q.score >= 90 ? 'good' : q.score >= 75 ? undefined : 'warn'} />
        <Metric label="Completeness" value={fmtPct(comp.completeness.value)} sub={`${fmtNum(o.missing.cells)} missing cells`} />
        <Metric label="Missing values" value={fmtPct(o.missing.pct, 2)} sub={`${plural(o.shape.rows - o.missing.complete_rows, 'row')} affected`} />
        <Metric label="Duplicate rows" value={fmtNum(o.duplicates.rows)} sub={fmtPct(o.duplicates.pct, 2) + ' of rows'} />
        <Metric label="Rows" value={fmtNum(o.shape.rows)} sub={o.dataset.original_rows !== o.shape.rows ? `${fmtNum(o.dataset.original_rows)} originally` : 'as uploaded'} />
        <Metric label="Columns" value={o.shape.columns} sub={Object.entries(o.type_counts).map(([k, v]) => `${v} ${k}`).join(', ')} />
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1.6fr) minmax(0, 1fr)', alignItems: 'start' }}>
        <div className="stack">
          {o.kpis.length > 1 && (
            <Panel title="Key measures" subtitle="Aggregation shown in each label; change compares the last two complete periods.">
              <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
                {o.kpis.map(k => (
                  <Metric key={k.label} label={k.label} value={fmtNum(k.value, { compact: true })}
                    sub={k.trend != null ? <span title={k.trend_detail?.label}>{k.trend > 0 ? '▲' : '▼'} {Math.abs(k.trend).toFixed(1)}% {k.trend_detail?.label?.replace(/ \(.*\)/, '')}</span> : k.n ? `n = ${fmtNum(k.n)}` : null} />
                ))}
              </div>
            </Panel>
          )}
          <Panel title="Top findings" subtitle="Ranked by evidence strength and impact." actions={<Link className="btn btn-ghost btn-sm" to={`/w/${id}/insights`}>All insights <ArrowRight size={14} /></Link>}>
            {o.findings.length === 0 ? <Empty title="No notable findings">The data shows no statistically supported patterns yet.</Empty>
              : <div className="stack-sm">{o.findings.map(f => <FindingCard key={f.id} f={f} />)}</div>}
          </Panel>
        </div>
        <div className="stack">
          <Panel title="Next steps" subtitle="Based on the current state of this workspace.">
            <ol className="list-plain stack-sm">
              {o.recommendations.map(r => (
                <li key={r.title}>
                  <Link to={`/w/${id}/${r.page}`} className="panel-flat row" style={{ textDecoration: 'none', color: 'inherit' }}>
                    <span className="grow"><strong>{r.title}</strong><br /><span className="small secondary">{r.reason}</span></span>
                    <ArrowRight size={16} aria-hidden />
                  </Link>
                </li>
              ))}
            </ol>
          </Panel>
          {o.issues.top.length > 0 && (
            <Panel title="Data issues" subtitle={`${o.issues.counts.high} high, ${o.issues.counts.medium} medium, ${o.issues.counts.low} low severity`}
              actions={<Link className="btn btn-ghost btn-sm" to={`/w/${id}/clean`}>Review</Link>}>
              <ul className="list-plain stack-sm">
                {o.issues.top.map(i => (
                  <li key={i.id} className="row" style={{ alignItems: 'flex-start' }}>
                    <Badge tone={SEV[i.severity]}>{i.severity}</Badge><span className="grow small">{i.title}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
          <Panel title="Model & signals">
            <div className="stack-sm">
              {o.model ? (
                <Link to={`/w/${id}/model`} className="panel-flat" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div className="row"><BrainCircuit size={16} aria-hidden /><strong>{o.model.model}</strong><span className="muted small">predicts {o.model.target}</span></div>
                  <div className="small secondary num" style={{ marginTop: 4 }}>{modelHeadline(o.model)}</div>
                </Link>
              ) : <p className="small muted">No model trained yet. <Link to={`/w/${id}/model`}>Train one</Link> to see which columns predict an outcome.</p>}
              {o.indicators.forecast && (
                <Link to={`/w/${id}/forecast`} className="panel-flat" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div className="row"><TrendingUp size={16} aria-hidden /><strong>Forecast</strong>{o.indicators.forecast.stale && <Badge tone="warn">data changed since</Badge>}</div>
                  <div className="small secondary num" style={{ marginTop: 4 }}>
                    Next period ({o.indicators.forecast.next_period.slice(0, 10)}): {fmtNum(o.indicators.forecast.next_value)} {o.indicators.forecast.target ?? 'records'}
                    {' '}(80% interval {fmtNum(o.indicators.forecast.lower80)}–{fmtNum(o.indicators.forecast.upper80)}), {o.indicators.forecast.method}
                  </div>
                </Link>
              )}
              {o.indicators.anomalies && (
                <Link to={`/w/${id}/model?tab=anomaly`} className="panel-flat" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div className="row"><AlertTriangle size={16} aria-hidden /><strong>Anomalies</strong>{o.indicators.anomalies.stale && <Badge tone="warn">data changed since</Badge>}</div>
                  <div className="small secondary num" style={{ marginTop: 4 }}>{fmtNum(o.indicators.anomalies.n)} rows flagged ({fmtPct(o.indicators.anomalies.rate_pct)})</div>
                </Link>
              )}
              {o.indicators.diagnostics?.top && (
                <Link to={`/w/${id}/diagnose`} className="panel-flat" style={{ textDecoration: 'none', color: 'inherit' }}>
                  <div className="row"><Activity size={16} aria-hidden /><strong>Diagnostics: {o.indicators.diagnostics.outcome}</strong></div>
                  <div className="small secondary" style={{ marginTop: 4 }}>{o.indicators.diagnostics.top} ({o.indicators.diagnostics.label})</div>
                </Link>
              )}
              {!o.indicators.forecast && !o.indicators.anomalies && <p className="small muted">Forecasts and anomaly results appear here after you run them.</p>}
            </div>
          </Panel>
          <Panel title="Recent activity">
            <ul className="timeline">
              {o.activity.map((a, i) => (
                <li key={i}><div className="small">{a.message}</div><div className="tiny muted">{timeAgo(a.ts)}</div></li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  )
}

function modelHeadline(m) {
  const t = m.test_metrics || {}
  if (m.problem_type === 'regression') return `Test R² ${fmtNum(t.r2, { digits: 3 })} · RMSE ${fmtNum(t.rmse)} · MAE ${fmtNum(t.mae)}`
  return `Test ROC-AUC ${fmtNum(t.roc_auc ?? t.roc_auc_ovr, { digits: 3 })} · balanced accuracy ${fmtNum(t.balanced_accuracy, { digits: 3 })} · F1 macro ${fmtNum(t.f1_macro, { digits: 3 })}`
}
