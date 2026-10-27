import { useEffect, useMemo, useState } from 'react'
import { Eye, FileDown, FileJson, FileText, Sparkles } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ChartBody, ForecastChart } from '../components/charts/ChartView'
import { ClaimBadge } from '../components/Findings'
import { Badge, Button, Callout, Check, DataTable, ErrorState, Input, LoadingBlock, Panel, Select, Tabs } from '../components/ui'
import { fmtNum } from '../lib/format'

const SECTIONS = [
  ['summary', 'Executive summary'], ['overview', 'Dataset overview'], ['quality', 'Data quality assessment'], ['methodology', 'Methodology & cleaning steps'],
  ['findings', 'Key findings'], ['charts', 'Charts'], ['model', 'Machine learning results'], ['forecast', 'Forecast'], ['diagnostics', 'Diagnostics'],
  ['rules', 'Data-quality rules'], ['recommendations', 'Recommendations'], ['limitations', 'Limitations & uncertainty'], ['reproducibility', 'Reproducibility'],
]

export default function Report() {
  const [tab, setTab] = useState('summary')
  return (
    <div className="page">
      <PageHeader title="Report" description="Everything in the report comes from computed results in this workspace. Choose what to include and export it." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'summary', label: 'Executive summary', icon: Sparkles }, { id: 'builder', label: 'Report builder', icon: FileText }]} />
      {tab === 'summary' ? <Summary /> : <Builder />}
    </div>
  )
}

function Summary() {
  const { id, dataKey } = useWorkspace()
  const { data: s, error, loading, reload } = useApi(() => api.summary(id), [id, dataKey])
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!s) return <LoadingBlock rows={3} h={90} />
  return (
    <div className="stack">
      <Panel>
        <div className="row-between">
          <h2 className="grow" style={{ fontSize: '1.3rem' }}>{s.headline}</h2>
          <Badge tone="accent">quality {s.quality_score}/100 · {s.quality_label}</Badge>
        </div>
        <p className="tiny muted" style={{ marginTop: 6 }}>Written by {s.generated_by === 'template' ? 'the analysis engine from computed results' : s.generated_by}.</p>
      </Panel>
      <div className="grid-2">
        {s.sections.map(sec => (
          <Panel key={sec.id} title={sec.title}><p className="secondary">{sec.content}</p></Panel>
        ))}
      </div>
    </div>
  )
}

function Builder() {
  const { id, dataKey, ws } = useWorkspace()
  const { notify } = useUi()
  const { data: ins } = useApi(() => api.insights(id), [id, dataKey])
  const { data: ch } = useApi(() => api.recommendedCharts(id), [id, dataKey])
  const { data: md } = useApi(() => api.models(id), [id])
  const [title, setTitle] = useState('')
  const [sections, setSections] = useState(SECTIONS.map(s => s[0]))
  const [fids, setFids] = useState(null)
  const [charts, setCharts] = useState(null)
  const [model, setModel] = useState('')
  const [stale, setStale] = useState(false)
  const [preview, setPreview] = useState(null)
  useEffect(() => { if (ins && fids === null) setFids(ins.findings.slice(0, 8).map(f => f.id)) }, [ins]) // eslint-disable-line
  useEffect(() => { if (ch && charts === null) setCharts(ch.charts.slice(0, 4).map((_, i) => i)) }, [ch]) // eslint-disable-line
  const body = useMemo(() => ({
    title: title || undefined, sections, finding_ids: fids ?? undefined, include_stale: stale,
    chart_specs: ch && charts ? charts.map(i => ch.charts[i]?.spec).filter(Boolean) : undefined,
    model_id: model || undefined,
  }), [title, sections, fids, charts, ch, model, stale])
  const pv = useAction(async () => setPreview(await api.reportPreview(id, { ...body, format: 'json' })))
  const exp = useAction(async format => {
    const name = await api.reportDownload(id, { ...body, format })
    notify(`Downloaded ${name}`, { kind: 'success', persist: true })
  })
  const toggle = (list, setList, v) => setList(list.includes(v) ? list.filter(x => x !== v) : [...list, v])
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(280px, 380px) minmax(0, 1fr)', alignItems: 'start' }}>
      <div className="stack">
        <Panel title="Contents">
          <div className="stack-sm">
            <Input label="Title" placeholder={`Analysis of ${ws.active_dataset.name}`} value={title} onChange={e => setTitle(e.target.value)} maxLength={160} />
            <fieldset style={{ border: 0, padding: 0, margin: 0 }} className="stack-sm">
              <legend className="label">Sections</legend>
              {SECTIONS.map(([k, l]) => <Check key={k} label={l} checked={sections.includes(k)} onChange={() => toggle(sections, setSections, k)} />)}
            </fieldset>
            {sections.includes('model') && md?.models?.length > 0 && (
              <Select label="Model to report" value={model} onChange={v => setModel(v || '')} placeholder="Latest model" options={md.models.map(m => ({ value: m.id, label: `${m.model} → ${m.target}` }))} />
            )}
            <Check label="Include forecasts/diagnostics computed on an earlier version of the data" checked={stale} onChange={setStale} />
          </div>
        </Panel>
        {sections.includes('findings') && ins && fids && (
          <Panel title="Findings to include" subtitle={`${fids.length} of ${ins.findings.length} selected`}>
            <div className="stack-sm" style={{ maxHeight: 280, overflowY: 'auto' }}>
              {ins.findings.map(f => <Check key={f.id} label={<span className="small">{f.title}</span>} checked={fids.includes(f.id)} onChange={() => toggle(fids, setFids, f.id)} />)}
            </div>
          </Panel>
        )}
        {sections.includes('charts') && ch && charts && (
          <Panel title="Charts to include" subtitle={`${charts.length} selected`}>
            <div className="stack-sm" style={{ maxHeight: 240, overflowY: 'auto' }}>
              {ch.charts.map((c, i) => <Check key={i} label={<span className="small">{c.title}</span>} checked={charts.includes(i)} onChange={() => toggle(charts, setCharts, i)} />)}
            </div>
          </Panel>
        )}
        <Panel title="Export">
          <div className="stack-sm">
            <Button icon={Eye} busy={pv.busy} onClick={() => pv.run().catch(() => {})}>Preview on screen</Button>
            <div className="row">
              <Button variant="primary" icon={FileDown} busy={exp.busy} onClick={() => exp.run('pdf').catch(e => notify(e.message, { kind: 'error' }))}>PDF</Button>
              <Button icon={FileText} onClick={() => exp.run('markdown').catch(e => notify(e.message, { kind: 'error' }))}>Markdown</Button>
              <Button icon={FileJson} onClick={() => exp.run('json').catch(e => notify(e.message, { kind: 'error' }))}>JSON</Button>
            </div>
            {(pv.error || exp.error) && <ErrorState error={pv.error || exp.error} compact />}
          </div>
        </Panel>
      </div>
      <div>
        {!preview && <Panel><div className="empty"><div className="empty-icon"><Eye size={22} /></div><h3>Preview the report</h3><p>Choose sections on the left, then preview. Exports use exactly the same content.</p></div></Panel>}
        {preview && <ReportPreview rep={preview} />}
      </div>
    </div>
  )
}

function ReportPreview({ rep }) {
  return (
    <article className="panel panel-pad stack" aria-label="Report preview">
      <div>
        <h1>{rep.meta.title}</h1>
        <p className="small muted">Dataset {rep.meta.dataset.name} · {fmtNum(rep.meta.dataset.rows)} rows · version {rep.meta.dataset.version} · {new Date(rep.meta.generated_at * 1000).toLocaleString()}</p>
      </div>
      {rep.sections.length === 0 && <Callout tone="info">No section has content with the current selection.</Callout>}
      {rep.sections.map(s => (
        <section key={s.id} className="stack-sm">
          <h2>{s.title}</h2>
          {s.blocks.map((b, i) => <Block key={i} b={b} />)}
        </section>
      ))}
    </article>
  )
}

function Block({ b }) {
  if (b.type === 'paragraph') return <p style={b.style === 'lead' ? { fontWeight: 600 } : undefined}>{b.text}</p>
  if (b.type === 'bullets') return <div>{b.title && <h4>{b.title}</h4>}<ul style={{ margin: '4px 0', paddingLeft: 20 }}>{b.items.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
  if (b.type === 'metrics') return <div className="row">{b.items.map(m => <span key={m.label} className="panel-flat small num"><span className="muted">{m.label}: </span>{typeof m.value === 'number' ? fmtNum(m.value) : m.value ?? '—'}</span>)}</div>
  if (b.type === 'table') return <div>{b.title && <h4 style={{ marginBottom: 6 }}>{b.title}</h4>}<DataTable columns={b.columns.map((c, i) => ({ key: String(i), label: c }))} rows={b.rows.map(r => Object.fromEntries(r.map((v, i) => [String(i), v])))} maxHeight={320} /></div>
  if (b.type === 'finding') return (
    <div className="finding">
      <div className="finding-head"><h3 className="grow" style={{ fontSize: '0.96rem' }}>{b.title}</h3><ClaimBadge type={b.claim_type} /></div>
      <p className="small">{b.observation}</p><p className="small num"><span className="muted">Evidence: </span>{b.evidence}</p>
      {b.follow_up?.length > 0 && <p className="small"><span className="muted">Follow-up: </span>{b.follow_up.join('; ')}</p>}
    </div>
  )
  if (b.type === 'chart') {
    const c = b.chart
    if (c.type === 'forecast') return <div className="chart-card"><div className="chart-title">{c.title}</div><ForecastChart fc={c.forecast} method={c.forecast.selected_method} height={240} /></div>
    return <div className="chart-card"><div className="chart-title">{c.title}</div><ChartBody chart={c} height={220} /></div>
  }
  return null
}
