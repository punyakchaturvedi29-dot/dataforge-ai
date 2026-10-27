import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Download, FlaskConical, UploadCloud } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { HBar } from '../components/charts/ChartView'
import { Badge, Button, Callout, Check, DataTable, Empty, ErrorState, LoadingBlock, Metric, Panel, Select, Tabs } from '../components/ui'
import { fmtNum, fmtPct, timeAgo } from '../lib/format'

export default function Predict() {
  const { id } = useWorkspace()
  const [params, setParams] = useSearchParams()
  const { data, error, loading } = useApi(() => api.models(id), [id])
  const [tab, setTab] = useState('single')
  const models = data?.models || []
  const mid = params.get('model') || models[0]?.id
  const { data: meta } = useApi(() => api.model(id, mid), [id, mid], { enabled: !!mid })
  return (
    <div className="page">
      <PageHeader title="Predict" description="Score new records with a trained model. Inputs are checked against the training schema, and the recorded cleaning steps are applied first." />
      {error && <ErrorState error={error} />}
      {loading && !data && <LoadingBlock rows={2} />}
      {data && !models.length && <Empty icon={FlaskConical} title="No trained models" action={<Link className="btn btn-primary" to={`/w/${id}/model`}>Train a model</Link>}>Train a model first; it will be listed here.</Empty>}
      {models.length > 0 && (
        <div className="stack">
          <div className="form-row" style={{ maxWidth: 640 }}>
            <Select label="Model" value={mid} onChange={v => setParams({ model: v })} options={models.map(m => ({ value: m.id, label: `${m.model} → ${m.target} (${timeAgo(m.created_at)})` }))} />
          </div>
          {meta && <ModelCard m={meta} />}
          <Tabs value={tab} onChange={setTab} tabs={[{ id: 'single', label: 'Enter values' }, { id: 'batch', label: 'Score a file' }]} />
          {meta && tab === 'single' && <Single m={meta} key={meta.id} />}
          {meta && tab === 'batch' && <Batch m={meta} key={meta.id + 'b'} />}
        </div>
      )}
    </div>
  )
}

function ModelCard({ m }) {
  const t = m.test_metrics || {}
  const key = m.problem_type === 'regression' ? [['R²', t.r2], ['RMSE', t.rmse], ['MAE', t.mae]] : [['ROC-AUC', t.roc_auc ?? t.roc_auc_ovr], ['Balanced accuracy', t.balanced_accuracy], ['F1 (macro)', t.f1_macro]]
  return (
    <div className="grid-4">
      <Metric label="Predicts" value={<span style={{ fontSize: '1.1rem' }}>{m.target}</span>} sub={`${m.model} · ${m.problem_type}`} />
      {key.map(([k, v]) => <Metric key={k} label={`Test ${k}`} value={fmtNum(v, { digits: 3 })} />)}
      {m.cleaning_ops?.length > 0 && <div className="panel-flat small" style={{ gridColumn: '1 / -1' }}>Cleaning steps replayed on new data: {m.cleaning_ops.map(o => o.label).join(' → ')}</div>}
    </div>
  )
}

function Single({ m }) {
  const { id } = useWorkspace()
  const feats = m.input_schema.features
  const [vals, setVals] = useState(() => Object.fromEntries(feats.map(f => [f.name, f.typical ?? ''])))
  const [explain, setExplain] = useState(true)
  const [out, setOut] = useState(null)
  const run = useAction(async () => {
    const row = {}
    feats.forEach(f => {
      const v = vals[f.name]
      row[f.name] = v === '' || v === undefined ? null : f.kind === 'numeric' ? (Number.isFinite(Number(v)) ? Number(v) : v) : v
    })
    setOut(await api.predict(id, m.id, { rows: [row], explain }))
  })
  const fieldErr = run.error?.details?.invalid_values
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', alignItems: 'start' }}>
      <Panel title="Input values" subtitle="Pre-filled with typical training values (median or most frequent).">
        <form className="stack-sm" onSubmit={e => { e.preventDefault(); run.run().catch(() => {}) }}>
          <div className="grid-2">
            {feats.map(f => {
              const err = fieldErr?.find(x => x.feature === f.name)
              const hint = f.kind === 'numeric' ? `training range ${fmtNum(f.min)} – ${fmtNum(f.max)}` : f.kind === 'categorical' ? `${f.n_unique} known values` : f.kind === 'datetime' ? `${String(f.min).slice(0, 10)} – ${String(f.max).slice(0, 10)}` : ''
              return (
                <div key={f.name} className="field">
                  <label htmlFor={`f-${f.name}`}>{f.name}</label>
                  {f.kind === 'categorical' || f.kind === 'boolean' ? (
                    <>
                      <input id={`f-${f.name}`} className="input" list={`l-${f.name}`} value={vals[f.name] ?? ''} onChange={e => setVals(v => ({ ...v, [f.name]: e.target.value }))} />
                      <datalist id={`l-${f.name}`}>{(f.categories || []).slice(0, 200).map(c => <option key={c} value={c} />)}</datalist>
                    </>
                  ) : (
                    <input id={`f-${f.name}`} className="input" type={f.kind === 'datetime' ? 'date' : 'text'} inputMode={f.kind === 'numeric' ? 'decimal' : undefined}
                      value={f.kind === 'datetime' ? String(vals[f.name] ?? '').slice(0, 10) : vals[f.name] ?? ''} aria-invalid={!!err}
                      onChange={e => setVals(v => ({ ...v, [f.name]: e.target.value }))} />
                  )}
                  <span className="hint" style={err ? { color: 'var(--bad)' } : undefined}>{err ? `Expected a ${err.expected}` : hint}</span>
                </div>
              )
            })}
          </div>
          <Check label="Explain this prediction" checked={explain} onChange={setExplain} />
          {run.error && !fieldErr && <ErrorState error={run.error} compact />}
          <div><Button type="submit" variant="primary" busy={run.busy}>Predict</Button></div>
        </form>
      </Panel>
      <div className="stack">
        {!out && <Panel><Empty icon={FlaskConical} title="Enter values and press Predict">The result, its confidence and the main reasons appear here.</Empty></Panel>}
        {out && <PredictionResult out={out} m={m} />}
      </div>
    </div>
  )
}

function PredictionResult({ out, m }) {
  const v = out.validation
  const pred = out.predictions[0]
  const probs = out.probabilities?.[0]
  const ex = out.explanations?.[0]
  return (
    <>
      <Panel title="Prediction">
        <div className="metric-value num" style={{ fontSize: '2rem' }}>{typeof pred === 'number' ? fmtNum(pred) : pred}</div>
        {m.problem_type === 'regression' && m.test_metrics?.mae != null && <p className="small muted">Typical error on held-out data: ± {fmtNum(m.test_metrics.mae)} (MAE).</p>}
        {probs && (
          <div style={{ marginTop: 10 }}>
            <HBar data={Object.entries(probs).map(([k, p]) => ({ cls: k, p: p * 100 })).sort((a, b) => b.p - a.p)} label="cls" value="p" valueLabel="Score %" format={x => `${x.toFixed(0)}%`} height={Math.max(90, Object.keys(probs).length * 30)} />
            <p className="tiny muted">{out.notes.join(' ')}</p>
          </div>
        )}
      </Panel>
      {(v.unseen_categories.length > 0 || v.out_of_range.length > 0 || v.cleaning_notes?.length > 0 || out.warnings?.length > 0) && (
        <Callout tone="warn" title="Input warnings">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {v.unseen_categories.map(u => <li key={u.feature}>“{u.examples.join(', ')}” was not seen in training for {u.feature}; the model treats it as an unknown category.</li>)}
            {v.out_of_range.map(u => <li key={u.feature}>{u.feature} is outside the training range {fmtNum(u.training_range[0])} – {fmtNum(u.training_range[1])}; the prediction is an extrapolation.</li>)}
            {[...(v.cleaning_notes || []), ...(out.warnings || [])].map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </Callout>
      )}
      {ex && (
        <Panel title="Why this prediction" subtitle={`${ex.method}. Explains the ${ex.explained_output}.`}>
          {ex.contributions.length ? <HBar data={ex.contributions.slice(0, 10).map(c => ({ ...c, label: `${c.feature} = ${c.value === null ? 'missing' : typeof c.value === 'number' ? fmtNum(c.value) : c.value}` }))} label="label" value="contribution" valueLabel="Effect vs typical value" format={x => fmtNum(x, { digits: 3 })} />
            : <p className="small muted">All inputs equal their typical values.</p>}
          <ul className="tiny muted" style={{ margin: '6px 0 0', paddingLeft: 18 }}>{ex.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </Panel>
      )}
    </>
  )
}

function Batch({ m }) {
  const { id, refresh } = useWorkspace()
  const { notify } = useUi()
  const [res, setRes] = useState(null)
  const run = useAction(async file => { setRes(await api.predictBatch(id, m.id, file)); refresh() })
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ multiple: false, noClick: true, onDrop: f => f[0] && run.run(f[0]).catch(() => {}) })
  const missing = run.error?.details?.missing_features
  return (
    <div className="stack">
      <div {...getRootProps({ className: `dropzone ${isDragActive ? 'active' : ''}` })} onClick={open} role="button" tabIndex={0} aria-label="Upload a file to score"
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && open()}>
        <input {...getInputProps()} aria-label="Choose file to score" />
        <UploadCloud size={26} aria-hidden style={{ color: 'var(--accent-text)' }} />
        <h3 style={{ marginTop: 8 }}>{run.busy ? 'Scoring…' : 'Drop a CSV or Excel file with the same columns'}</h3>
        <p className="small muted">Required columns: {m.input_schema.order.join(', ')}</p>
      </div>
      {missing && <Callout tone="bad" title="The file is missing required columns">{missing.join(', ')}. Add them, or rename existing columns to match.</Callout>}
      {run.error && !missing && <ErrorState error={run.error} />}
      {res && (
        <>
          <div className="grid-4">
            <Metric label="Rows scored" value={fmtNum(res.summary.rows)} sub={res.summary.filename} />
            <Metric label="Rows with input warnings" value={fmtNum(res.summary.rows_with_warnings)} tone={res.summary.rows_with_warnings ? 'warn' : undefined} />
            {res.summary.distribution && <Metric label="Mean prediction" value={fmtNum(res.summary.distribution.mean)} sub={`range ${fmtNum(res.summary.distribution.min)} – ${fmtNum(res.summary.distribution.max)}`} />}
            {res.summary.class_counts && <Metric label="Predicted classes" value={Object.keys(res.summary.class_counts).length} sub={Object.entries(res.summary.class_counts).map(([k, v]) => `${k}: ${v}`).join(', ')} />}
          </div>
          {(res.validation.unseen_categories.length > 0 || res.validation.out_of_range.length > 0 || res.validation.invalid_values.length > 0 || res.validation.unexpected_columns.length > 0) && (
            <Callout tone="warn" title="Validation report">
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {res.validation.invalid_values.map(x => <li key={x.feature}>{fmtNum(x.count)} value(s) in {x.feature} are not a valid {x.expected} (e.g. {x.examples.join(', ')}); they were treated as missing.</li>)}
                {res.validation.unseen_categories.map(x => <li key={x.feature}>{fmtNum(x.count)} row(s) have unseen values in {x.feature} (e.g. {x.examples.join(', ')}).</li>)}
                {res.validation.out_of_range.map(x => <li key={x.feature}>{fmtNum(x.count)} row(s) have {x.feature} outside the training range.</li>)}
                {res.validation.unexpected_columns.length > 0 && <li>Ignored columns not used by the model: {res.validation.unexpected_columns.join(', ')}.</li>}
              </ul>
            </Callout>
          )}
          <Panel title="Predictions" subtitle="First 50 rows. The download contains every row plus probabilities and per-row warnings."
            actions={<Button variant="primary" icon={Download} onClick={() => api.downloadPredictions(id, res.prediction_id).then(n => notify(`Downloaded ${n}`, { kind: 'success' })).catch(e => notify(e.message, { kind: 'error' }))}>Download CSV</Button>}>
            <DataTable columns={res.columns.map(c => ({ key: c, label: c, numeric: false }))} rows={res.preview} maxHeight={420} />
          </Panel>
        </>
      )}
    </div>
  )
}
