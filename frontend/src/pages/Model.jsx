import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { AlertTriangle, BrainCircuit, Download, FlaskConical, GitCompare, Play, ScatterChart as ScatterIcon, Square, Trash2 } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/endpoints'
import { useAction, useApi, useJob } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ChartTooltip, ConfusionMatrix, CurveChart, HBar, Legend, SimpleBars, XYScatter } from '../components/charts/ChartView'
import { useChartTheme } from '../components/charts/theme'
import { Badge, Button, Callout, Check, Confirm, DataTable, Empty, ErrorState, Input, LoadingBlock, Metric, Panel, Progress, Segmented, Select, Tabs } from '../components/ui'
import { fmtNum, fmtPct, timeAgo } from '../lib/format'

const MODEL_NAMES = { linear: 'Linear / logistic', decision_tree: 'Decision tree', random_forest: 'Random forest', hist_gradient_boosting: 'Histogram gradient boosting', xgboost: 'XGBoost', lightgbm: 'LightGBM' }
const PRESETS = {
  quick: { label: 'Quick', desc: '4 models, no tuning — seconds', quick: true, tune: false },
  standard: { label: 'Standard', desc: 'all models, tune the best two', quick: false, tune: true, n_iter: 8 },
  thorough: { label: 'Thorough', desc: 'more tuning, longer budget', quick: false, tune: true, n_iter: 20, time_budget_s: 300 },
}

export default function Model() {
  const [params, setParams] = useSearchParams()
  const tab = params.get('tab') || 'train'
  const setTab = t => setParams({ tab: t }, { replace: true })
  return (
    <div className="page">
      <PageHeader title="Train models" description="Models are compared by cross-validation on a training split, the best ones tuned within a time budget, and the winner evaluated once on a held-out test set next to a naive baseline." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'train', label: 'AutoML', icon: BrainCircuit }, { id: 'experiments', label: 'Experiments', icon: GitCompare },
        { id: 'cluster', label: 'Segmentation', icon: ScatterIcon }, { id: 'anomaly', label: 'Anomaly detection', icon: AlertTriangle }]} />
      {tab === 'train' && <Train />}
      {tab === 'experiments' && <Experiments />}
      {tab === 'cluster' && <Clusters />}
      {tab === 'anomaly' && <Anomalies />}
    </div>
  )
}

function Train() {
  const { id, dataKey, refresh } = useWorkspace()
  const { notify } = useUi()
  const [params] = useSearchParams()
  const { data: o, error } = useApi(() => api.mlOptions(id), [id, dataKey])
  const [cfg, setCfg] = useState({ preset: 'standard', problem_type: 'auto', test_size: 0.2, cv_folds: 5, class_weight: 'auto', random_state: 42, exclude: [], models: null })
  const [adv, setAdv] = useState(false)
  const [exp, setExp] = useState(null)
  const { job, start, setJob } = useJob(api.job)
  useEffect(() => {
    if (o && !cfg.target) setCfg(c => ({ ...c, target: params.get('target') || o.selected_target || o.targets[0]?.column }))
  }, [o]) // eslint-disable-line
  const target = o?.targets.find(t => t.column === cfg.target)
  const metrics = target ? o.metrics[cfg.problem_type !== 'auto' ? cfg.problem_type : target.task_hint] || [] : []
  const startJob = useAction(async () => {
    const p = PRESETS[cfg.preset]
    const body = { target: cfg.target, problem_type: cfg.problem_type, test_size: Number(cfg.test_size), cv_folds: Number(cfg.cv_folds), class_weight: cfg.class_weight,
      random_state: Number(cfg.random_state), quick: p.quick, tune: p.tune, exclude: cfg.exclude }
    if (p.n_iter) body.n_iter = p.n_iter
    if (cfg.time_budget_s || p.time_budget_s) body.time_budget_s = Number(cfg.time_budget_s || p.time_budget_s)
    if (cfg.time_column) body.time_column = cfg.time_column
    if (cfg.primary_metric) body.primary_metric = cfg.primary_metric
    if (cfg.models?.length) body.models = cfg.models
    const r = await api.train(id, body)
    setExp(null)
    start(r.job_id, async j => {
      if (j.status === 'succeeded') {
        const e = await api.experiment(id, j.result.experiment_id)
        setExp(e)
        refresh()
        notify(`Model ready: ${j.result.summary.selected_model} for “${j.result.summary.target}”`, { kind: 'success', persist: true, link: `/w/${id}/model?tab=experiments` })
      } else if (j.status === 'failed') notify(`Training failed: ${j.error?.message}`, { kind: 'error', persist: true })
    })
  })
  if (error) return <ErrorState error={error} />
  if (!o) return <LoadingBlock rows={2} />
  if (!o.targets.length) return <Empty title="No suitable target column">Models need a numeric or categorical column to predict.</Empty>
  const running = job && ['queued', 'running'].includes(job.status)
  const allCols = o.columns.filter(c => c.name !== cfg.target).map(c => c.name)
  return (
    <div className="stack">
      <Panel title="What should the model predict?">
        <div className="form-row">
          <Select label="Target column" value={cfg.target} onChange={v => setCfg(c => ({ ...c, target: v, primary_metric: undefined }))}
            options={o.targets.map(t => ({ value: t.column, label: `${t.column} — ${t.task_hint}${t.missing_pct ? `, ${t.missing_pct.toFixed(0)}% missing` : ''}` }))} />
          <Select label="Problem type" value={cfg.problem_type} onChange={v => setCfg(c => ({ ...c, problem_type: v, primary_metric: undefined }))}
            options={[{ value: 'auto', label: `Automatic (${target?.task_hint || '…'})` }, { value: 'regression', label: 'Regression (number)' }, { value: 'binary', label: 'Binary classification' }, { value: 'multiclass', label: 'Multiclass classification' }]} />
          <div className="field"><span className="label">Search effort</span>
            <Segmented label="Search effort" value={cfg.preset} onChange={v => setCfg(c => ({ ...c, preset: v }))} options={Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.label }))} />
            <span className="hint">{PRESETS[cfg.preset].desc}</span>
          </div>
        </div>
        <div style={{ marginTop: 12 }}><Check label="Show advanced settings" checked={adv} onChange={setAdv} /></div>
        {adv && (
          <div className="stack" style={{ marginTop: 12 }}>
            <div className="form-row">
              <Select label="Time column (chronological split)" value={cfg.time_column} onChange={v => setCfg(c => ({ ...c, time_column: v }))} placeholder="None — random split" options={o.datetime_columns}
                hint="Use when you will predict future records: the test set becomes the most recent rows." />
              <Input label="Test share" type="number" step="0.05" min="0.05" max="0.5" value={cfg.test_size} onChange={e => setCfg(c => ({ ...c, test_size: e.target.value }))} />
              <Input label="CV folds" type="number" min="2" max="10" value={cfg.cv_folds} onChange={e => setCfg(c => ({ ...c, cv_folds: e.target.value }))} />
              <Select label="Selection metric" value={cfg.primary_metric} onChange={v => setCfg(c => ({ ...c, primary_metric: v }))} placeholder="Recommended" options={metrics.map(m => ({ value: m, label: o.metric_labels[m] || m }))} />
              {target?.task_hint !== 'regression' && <Select label="Class weights" value={cfg.class_weight} onChange={v => setCfg(c => ({ ...c, class_weight: v }))} options={[{ value: 'auto', label: 'Balance if imbalanced' }, { value: 'balanced', label: 'Always balance' }, { value: 'none', label: 'None' }]} />}
              <Input label="Time budget (s)" type="number" min="5" max="3600" placeholder={String(o.defaults.time_budget_s)} value={cfg.time_budget_s || ''} onChange={e => setCfg(c => ({ ...c, time_budget_s: e.target.value }))} />
              <Input label="Random seed" type="number" value={cfg.random_state} onChange={e => setCfg(c => ({ ...c, random_state: e.target.value }))} />
            </div>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend className="label">Candidate models</legend>
              <div className="row" style={{ marginTop: 6 }}>
                {Object.entries(o.available_models).filter(([k]) => k !== 'baseline').map(([k, avail]) => (
                  <Check key={k} disabled={!avail} label={`${MODEL_NAMES[k]}${avail ? '' : ' (not installed)'}`} checked={avail && (!cfg.models || cfg.models.includes(k))}
                    onChange={on => setCfg(c => { const cur = c.models || Object.keys(o.available_models).filter(x => x !== 'baseline' && o.available_models[x]); return { ...c, models: on ? [...cur, k] : cur.filter(x => x !== k) } })} />
                ))}
              </div>
              <span className="hint">A naive baseline is always included for comparison.</span>
            </fieldset>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend className="label">Exclude columns</legend>
              <div className="panel-flat" style={{ maxHeight: 150, overflowY: 'auto', marginTop: 6, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 4 }}>
                {allCols.map(c => <Check key={c} label={<span className="truncate">{c}</span>} checked={cfg.exclude.includes(c)} onChange={on => setCfg(s => ({ ...s, exclude: on ? [...s.exclude, c] : s.exclude.filter(x => x !== c) }))} />)}
              </div>
              <span className="hint">Identifiers, constants and free text are excluded automatically. Exclude anything not known at prediction time.</span>
            </fieldset>
          </div>
        )}
        <div className="row" style={{ marginTop: 14 }}>
          <Button variant="primary" icon={Play} busy={startJob.busy} disabled={!cfg.target || running} onClick={() => startJob.run().catch(() => {})}>Train and compare models</Button>
          {startJob.error && <ErrorState error={startJob.error} compact />}
        </div>
      </Panel>
      {job && (running || job.status === 'failed' || job.status === 'cancelled') && <JobPanel job={job} onCancel={async () => setJob(await api.cancelJob(job.id))} />}
      {exp && <ExperimentReport exp={exp} />}
    </div>
  )
}

function JobPanel({ job, onCancel }) {
  const running = ['queued', 'running'].includes(job.status)
  return (
    <Panel title={running ? 'Training in progress' : job.status === 'failed' ? 'Training failed' : 'Training cancelled'}
      actions={running && <Button icon={Square} onClick={onCancel}>Cancel</Button>}>
      {running && <><Progress value={job.progress} label="Training progress" /><p className="small" style={{ marginTop: 8 }} aria-live="polite">{job.message} · {Math.round(job.elapsed_s)}s</p></>}
      {job.status === 'failed' && <Callout tone="bad">{job.error?.message}</Callout>}
      {job.log?.length > 0 && <ol className="tiny muted" style={{ marginTop: 8, paddingLeft: 18 }}>{job.log.slice(-6).map((l, i) => <li key={i}>{l.t}s — {l.message}</li>)}</ol>}
    </Panel>
  )
}

const METRIC_ORDER = ['r2', 'rmse', 'mae', 'median_ae', 'mape_pct', 'roc_auc', 'roc_auc_ovr', 'average_precision', 'balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'mcc', 'log_loss', 'brier']
const METRIC_NAME = { r2: 'R²', rmse: 'RMSE', mae: 'MAE', median_ae: 'Median abs. error', mape_pct: 'MAPE %', roc_auc: 'ROC-AUC', roc_auc_ovr: 'ROC-AUC (OvR)', average_precision: 'PR-AUC', balanced_accuracy: 'Balanced accuracy', accuracy: 'Accuracy', f1: 'F1', precision: 'Precision', recall: 'Recall', f1_macro: 'F1 (macro)', f1_weighted: 'F1 (weighted)', precision_macro: 'Precision (macro)', recall_macro: 'Recall (macro)', mcc: 'Matthews corr.', log_loss: 'Log loss', brier: 'Brier score' }
const LOWER_BETTER = new Set(['rmse', 'mae', 'median_ae', 'mape_pct', 'log_loss', 'brier'])

export function ExperimentReport({ exp }) {
  const { id } = useWorkspace()
  const { notify } = useUi()
  const navigate = useNavigate()
  const r = exp.result
  const pm = r.primary_metric
  const tm = r.test_metrics, bm = r.baseline_test_metrics || {}
  const metricRows = METRIC_ORDER.filter(k => tm[k] !== undefined).map(k => ({ metric: METRIC_NAME[k] || k, k, model: tm[k], baseline: bm[k] }))
  const cands = r.candidates.map(c => ({ ...c, cv_mean: c.cv?.[pm]?.mean, cv_std: c.cv?.[pm]?.std, train: c.train?.[pm] }))
  const leaks = r.leakage || []
  const imp = r.feature_importance.items.filter(i => i.importance > 0).slice(0, 15)
  const dg = r.diagnostics
  return (
    <div className="stack">
      <Panel title={`${r.selected.model} predicts “${r.target}”`} subtitle={`${r.problem_type} · ${r.rows.train.toLocaleString()} training / ${r.rows.test.toLocaleString()} test rows · ${r.split.method} split · trained ${timeAgo(exp.created_at)} in ${fmtNum(r.duration_s)}s`}
        actions={<>
          <Button icon={Download} onClick={() => api.experimentCsv(id, exp.id).catch(e => notify(e.message, { kind: 'error' }))}>Comparison CSV</Button>
          <Button variant="primary" icon={FlaskConical} onClick={() => navigate(`/w/${id}/predict?model=${exp.model_id}`)}>Use for predictions</Button>
        </>}>
        <p>{r.rationale}</p>
      </Panel>
      {leaks.length > 0 && (
        <Callout tone="warn" title="Possible target leakage">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{leaks.map((l, i) => <li key={i}><strong>{l.column}</strong>: {l.reason} — <em>{l.action}</em></li>)}</ul>
        </Callout>
      )}
      {r.warnings.length > 0 && <Callout tone="info" title="Notes"><ul style={{ margin: 0, paddingLeft: 18 }}>{r.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Callout>}
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Panel title="Held-out test set" subtitle="Evaluated once, after model selection, next to the naive baseline.">
          <DataTable rowKey={x => x.k} columns={[{ key: 'metric', label: 'Metric' }, { key: 'model', label: 'Model', numeric: true, render: v => fmtNum(v, { digits: 4 }) },
            { key: 'baseline', label: 'Baseline', numeric: true, render: v => fmtNum(v, { digits: 4 }) },
            { key: 'k', label: '', render: k => (LOWER_BETTER.has(k) ? <span className="tiny muted">lower is better</span> : '') }]} rows={metricRows} maxHeight={360} />
        </Panel>
        <Panel title="Model comparison" subtitle={`${r.primary_metric_label}, ${r.cv_description} on the training split.`}>
          <DataTable rowKey={c => c.key} columns={[
            { key: 'model', label: 'Model', render: (v, c) => <span>{v}{c.key === r.selected.key && <Badge tone="accent">selected</Badge>}{c.tuned && <Badge tone="info">tuned</Badge>}</span> },
            { key: 'cv_mean', label: 'CV mean', numeric: true, render: v => fmtNum(v, { digits: 4 }) },
            { key: 'cv_std', label: '± std', numeric: true, render: v => fmtNum(v, { digits: 3 }) },
            { key: 'train', label: 'Train', numeric: true, render: v => fmtNum(v, { digits: 4 }) },
            { key: 'cv_seconds', label: 'Seconds', numeric: true, render: v => fmtNum(v) },
            { key: 'status', label: 'Status', render: (v, c) => (v === 'ok' ? '' : <span className="tiny muted" title={c.reason}>{v}: {c.reason}</span>) },
          ]} rows={cands} maxHeight={360} />
          <p className="tiny muted" style={{ marginTop: 6 }}>A large gap between Train and CV indicates overfitting.{r.tuning?.length ? ' ' + r.tuning.map(t => `${t.model}: ${t.status === 'ok' ? `tuned ${t.n_iter} configurations (${t.improved ? 'improved' : 'no improvement'})` : t.reason}`).join('; ') + '.' : ''}</p>
        </Panel>
      </div>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Panel title="What the model relies on" subtitle={r.feature_importance.method || 'Not available'}>
          {imp.length ? <HBar data={imp} label="feature" value="importance" valueLabel="Drop in score" format={v => fmtNum(v, { digits: 3 })} /> : <p className="small muted">No feature changed the score measurably when shuffled.</p>}
          <ul className="tiny muted" style={{ margin: '6px 0 0', paddingLeft: 18 }}>{r.feature_importance.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </Panel>
        {r.problem_type === 'regression' ? <RegressionDiag dg={dg} target={r.target} /> : <ClassDiag dg={dg} r={r} />}
      </div>
      {r.problem_type !== 'regression' && dg.roc_curve && (
        <div className="grid-3">
          <Panel title="ROC curve" subtitle={`AUC ${fmtNum(tm.roc_auc, { digits: 3 })}; diagonal = random`}><CurveChart data={dg.roc_curve} x="fpr" y="tpr" xLabel="False positive rate" yLabel="True positive rate" diagonal="up" /></Panel>
          <Panel title="Precision–recall" subtitle={`PR-AUC ${fmtNum(tm.average_precision, { digits: 3 })}; prevalence ${fmtPct(dg.prevalence * 100)}`}><CurveChart data={dg.pr_curve} x="recall" y="precision" xLabel="Recall" yLabel="Precision" diagonal={dg.prevalence} /></Panel>
          <Panel title="Calibration" subtitle={`Brier ${fmtNum(tm.brier, { digits: 3 })}; points on the diagonal = calibrated`}><CurveChart data={dg.calibration} x="mean_predicted" y="observed_rate" xLabel="Predicted probability" yLabel="Observed rate" diagonal="up" /></Panel>
        </div>
      )}
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <Panel title="Features">
          <DataTable columns={[{ key: 'column', label: 'Column' }, { key: 'status', label: 'Status', render: v => <Badge tone={v === 'used' ? 'good' : undefined}>{v}</Badge> },
            { key: 'kind', label: 'Type' }, { key: 'reason', label: 'Note', render: (v, row) => v || row.note || '' }]} rows={r.features} maxHeight={300} />
        </Panel>
        <Panel title="Limitations and reproducibility">
          <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>{r.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
          <dl className="kv tiny" style={{ marginTop: 12 }}>
            <dt>Seed</dt><dd>{r.config.random_state}</dd><dt>Test share</dt><dd>{r.config.test_size}</dd><dt>CV</dt><dd>{r.cv_description}</dd>
            <dt>Class weights</dt><dd>{r.class_weight}</dd>{r.split.test_from && <><dt>Test period from</dt><dd>{r.split.test_from}</dd></>}
            {Object.entries(r.versions).map(([k, v]) => <span key={k} style={{ display: 'contents' }}><dt>{k}</dt><dd>{v || 'not installed'}</dd></span>)}
          </dl>
        </Panel>
      </div>
    </div>
  )
}

function RegressionDiag({ dg, target }) {
  const hist = dg.residual_histogram.counts.map((c, i) => ({ bin: fmtNum((dg.residual_histogram.edges[i] + dg.residual_histogram.edges[i + 1]) / 2), count: c }))
  return (
    <Panel title="Errors on the test set" subtitle={`Residual mean ${fmtNum(dg.residual_mean)} (≈ 0 means unbiased), std ${fmtNum(dg.residual_std)}`}>
      <XYScatter points={dg.actual_vs_predicted} x="actual" y="predicted" xLabel={`Actual ${target}`} yLabel="Predicted" identity height={220} />
      <div className="grid-2" style={{ marginTop: 8 }}>
        <div><div className="chart-sub">Residual distribution</div><SimpleBars data={hist} x="bin" y="count" yLabel="Rows" height={160} /></div>
        <div><div className="chart-sub">Residual vs predicted</div><XYScatter points={dg.residual_vs_predicted} x="predicted" y="residual" xLabel="Predicted" yLabel="Residual" zeroLine height={160} /></div>
      </div>
      {dg.notes?.map((n, i) => <p key={i} className="small" style={{ color: 'var(--warn)' }}>{n}</p>)}
    </Panel>
  )
}

function ClassDiag({ dg, r }) {
  return (
    <Panel title="Errors on the test set" subtitle={`Always predicting the majority class would score ${fmtPct(dg.majority_baseline_accuracy * 100)} accuracy.`}>
      <ConfusionMatrix labels={dg.confusion_matrix.labels} matrix={dg.confusion_matrix.matrix} />
      <div style={{ marginTop: 10 }}>
        <DataTable columns={[{ key: 'class', label: 'Class' }, { key: 'precision', label: 'Precision', numeric: true, render: v => v.toFixed(3) }, { key: 'recall', label: 'Recall', numeric: true, render: v => v.toFixed(3) },
          { key: 'f1', label: 'F1', numeric: true, render: v => v.toFixed(3) }, { key: 'support', label: 'Rows', numeric: true }]} rows={dg.per_class} maxHeight={220} />
      </div>
    </Panel>
  )
}

function Experiments() {
  const { id, dataKey } = useWorkspace()
  const { notify } = useUi()
  const { data, error, loading, reload } = useApi(() => api.experiments(id), [id, dataKey])
  const [open, setOpen] = useState(null)
  const [cmp, setCmp] = useState([])
  const [del, setDel] = useState(null)
  const { data: full } = useApi(() => (open ? api.experiment(id, open) : null), [open], { enabled: !!open })
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data) return <LoadingBlock rows={2} />
  const list = data.experiments
  if (!list.length) return <Empty icon={GitCompare} title="No experiments yet">Train models on the AutoML tab; each run is saved here with its settings and results.</Empty>
  const chosen = list.filter(e => cmp.includes(e.id))
  const keys = [...new Set(chosen.flatMap(e => Object.keys(e.test_metrics || {})))].filter(k => METRIC_NAME[k])
  return (
    <div className="stack">
      <Panel title="Saved experiments" subtitle="Tick two or more to compare their test-set metrics side by side.">
        <DataTable rowKey={e => e.id} columns={[
          { key: 'id', label: '', render: v => <input type="checkbox" aria-label="Compare" checked={cmp.includes(v)} onChange={ev => setCmp(c => ev.target.checked ? [...c, v] : c.filter(x => x !== v))} /> },
          { key: 'target', label: 'Target' }, { key: 'selected_model', label: 'Model' }, { key: 'problem_type', label: 'Type' },
          { key: 'cv_score', label: 'CV score', numeric: true, render: (v, e) => `${v} ${e.primary_metric_label}` },
          { key: 'dataset_version', label: 'Data version', numeric: true },
          { key: 'created_at', label: 'Trained', render: v => timeAgo(v) },
          { key: 'model_id', label: '', render: (v, e) => <div className="row"><Button size="sm" onClick={() => setOpen(e.id)}>Open</Button><Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete experiment" onClick={() => setDel(e)} /></div> },
        ]} rows={list} />
      </Panel>
      {chosen.length >= 2 && (
        <Panel title="Comparison (test set)">
          <DataTable columns={[{ key: 'metric', label: 'Metric' }, ...chosen.map(e => ({ key: e.id, label: `${e.selected_model} → ${e.target}`, numeric: true, render: v => fmtNum(v, { digits: 4 }) }))]}
            rows={keys.map(k => ({ metric: METRIC_NAME[k], ...Object.fromEntries(chosen.map(e => [e.id, e.test_metrics[k]])) }))} />
          {new Set(chosen.map(e => e.target)).size > 1 && <p className="small" style={{ color: 'var(--warn)', marginTop: 6 }}>These experiments predict different targets, so their metrics are not directly comparable.</p>}
          {new Set(chosen.map(e => e.dataset_version)).size > 1 && <p className="small muted">Experiments were trained on different versions of the data.</p>}
        </Panel>
      )}
      {open && full && <ExperimentReport exp={full} />}
      {del && <Confirm title="Delete experiment?" danger confirmLabel="Delete" onCancel={() => setDel(null)} onConfirm={async () => {
        try { await api.deleteExperiment(id, del.id); notify('Experiment and its model deleted', { kind: 'success' }); if (open === del.id) setOpen(null); reload() } catch (e) { notify(e.message, { kind: 'error' }) }
        setDel(null)
      }}><p>The experiment and its trained model ({del.selected_model} → {del.target}) will be deleted.</p></Confirm>}
    </div>
  )
}

function Clusters() {
  const { id } = useWorkspace()
  const t = useChartTheme()
  const { data: latest } = useApi(() => api.latest(id, 'cluster').catch(() => null), [id])
  const [cfg, setCfg] = useState({ n_clusters: 3, method: 'kmeans' })
  const [res, setRes] = useState(null)
  useEffect(() => { if (latest && !res) setRes(latest.result) }, [latest]) // eslint-disable-line
  const run = useAction(async () => setRes(await api.cluster(id, { n_clusters: Number(cfg.n_clusters), method: cfg.method })))
  const groups = res ? [...new Set(res.scatter.points.map(p => p.cluster))].sort((a, b) => a - b) : []
  return (
    <div className="stack">
      <Panel title="Find groups of similar rows" subtitle="Numeric columns are standardised; categorical columns with up to 12 values are one-hot encoded. Identifiers and dates are excluded.">
        <div className="form-row">
          <Select label="Algorithm" value={cfg.method} onChange={v => setCfg(c => ({ ...c, method: v }))} options={[{ value: 'kmeans', label: 'K-means (choose k)' }, { value: 'dbscan', label: 'DBSCAN (finds k, flags noise)' }]} />
          {cfg.method === 'kmeans' && <Input label="Number of clusters (2–8)" type="number" min={2} max={8} value={cfg.n_clusters} onChange={e => setCfg(c => ({ ...c, n_clusters: Math.max(2, Math.min(8, Number(e.target.value) || 2)) }))} />}
          <Button variant="primary" busy={run.busy} onClick={() => run.run().catch(() => {})}>Run segmentation</Button>
        </div>
        {run.error && <div style={{ marginTop: 10 }}><ErrorState error={run.error} compact /></div>}
      </Panel>
      {res && (
        <>
          <div className="grid-4">
            <Metric label="Clusters" value={res.n_clusters} sub={res.noise_points ? `${res.noise_points} noise rows` : res.method} />
            <Metric label="Silhouette" value={fmtNum(res.silhouette_score, { digits: 2 })} sub={res.separation} help="−1 to 1. Above 0.5 = well separated, 0.25–0.5 = moderate, below 0.25 = weak structure." />
            <Metric label="Features" value={res.features_used.length} />
            {res.scatter.explained_variance && <Metric label="Variance shown in 2-D" value={fmtPct((res.scatter.explained_variance[0] + res.scatter.explained_variance[1]) * 100, 0)} sub="by the two principal components" />}
          </div>
          <div className="grid-2" style={{ alignItems: 'start' }}>
            <Panel title="Clusters in two dimensions" subtitle={`Principal component projection${res.scatter.sampled ? ' (random sample of points)' : ''}`}>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke={t.grid} />
                  <XAxis type="number" dataKey="x" tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} domain={['auto', 'auto']} tickFormatter={v => fmtNum(v)} />
                  <YAxis type="number" dataKey="y" tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} width={40} domain={['auto', 'auto']} tickFormatter={v => fmtNum(v)} />
                  <Tooltip content={<ChartTooltip title={p => (p[0].payload.cluster === -1 ? 'Noise' : `Cluster ${p[0].payload.cluster + 1}`)} rows={() => []} />} />
                  {groups.map((g, i) => (
                    <Scatter key={g} data={res.scatter.points.filter(p => p.cluster === g)} isAnimationActive={false}
                      shape={pr => <circle cx={pr.cx} cy={pr.cy} r={3.2} fill={g === -1 ? t['line-2'] : t.series[i % 8]} fillOpacity={0.7} stroke={t['chart-surface']} strokeWidth={0.6} />} />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
              <Legend items={groups.map((g, i) => ({ label: g === -1 ? 'Noise' : `Cluster ${g + 1}`, color: g === -1 ? t['line-2'] : t.series[i % 8] }))} />
            </Panel>
            {res.elbow?.length > 0 && (
              <Panel title="Choosing k" subtitle="Higher silhouette is better; look for the elbow in inertia.">
                <div className="chart-sub">Silhouette by k</div>
                <ResponsiveContainer width="100%" height={130}>
                  <LineChart data={res.elbow} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={t.grid} /><XAxis dataKey="k" tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} /><YAxis width={40} tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} />
                    <Tooltip content={<ChartTooltip title={(p, l) => `k = ${l}`} rows={p => [['Silhouette', fmtNum(p[0].value, { digits: 3 })]]} />} />
                    <Line dataKey="silhouette" stroke={t.s1} strokeWidth={2} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="chart-sub">Inertia by k</div>
                <ResponsiveContainer width="100%" height={130}>
                  <LineChart data={res.elbow} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={t.grid} /><XAxis dataKey="k" tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} /><YAxis width={40} tickLine={false} stroke={t.axis} tick={{ fontSize: 11 }} tickFormatter={v => fmtNum(v, { compact: true })} />
                    <Tooltip content={<ChartTooltip title={(p, l) => `k = ${l}`} rows={p => [['Inertia', fmtNum(p[0].value)]]} />} />
                    <Line dataKey="inertia" stroke={t.s1} strokeWidth={2} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
            )}
          </div>
          <Panel title="Cluster profiles" subtitle="Distinctive features differ from the overall mean by at least half a standard deviation.">
            <DataTable rowKey={c => c.cluster} columns={[{ key: 'label', label: 'Cluster' }, { key: 'size', label: 'Rows', numeric: true }, { key: 'pct', label: 'Share', numeric: true, render: v => fmtPct(v) },
              { key: 'distinctive', label: 'Distinctive features', render: v => v.slice(0, 4).map(d => `${d.feature} ${d.direction} (z ${d.z.toFixed(1)})`).join(', ') || 'close to average' },
              { key: 'top_categories', label: 'Most common categories', render: v => (v ? Object.entries(v).slice(0, 3).map(([k, x]) => `${k}: ${x.value} (${fmtPct(x.share * 100, 0)})`).join('; ') : '') }]} rows={res.cluster_summaries} />
            <ul className="tiny muted" style={{ margin: '8px 0 0', paddingLeft: 18 }}>{res.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </Panel>
        </>
      )}
    </div>
  )
}

function Anomalies() {
  const { id } = useWorkspace()
  const t = useChartTheme()
  const { data: latest } = useApi(() => api.latest(id, 'anomaly').catch(() => null), [id])
  const [contam, setContam] = useState(0.05)
  const [res, setRes] = useState(null)
  useEffect(() => { if (latest && !res) setRes(latest.result) }, [latest]) // eslint-disable-line
  const run = useAction(async () => setRes(await api.anomaly(id, { contamination: Number(contam) })))
  const hist = res ? res.score_distribution.counts.map((c, i) => ({ s: fmtNum((res.score_distribution.edges[i] + res.score_distribution.edges[i + 1]) / 2, { digits: 3 }), c })) : []
  return (
    <div className="stack">
      <Panel title="Find unusual rows" subtitle="Isolation Forest scores how easy each row is to isolate from the rest, using all numeric and low-cardinality categorical columns.">
        <div className="form-row">
          <div className="field" style={{ flex: '1 1 260px' }}>
            <label htmlFor="contam">Share of rows to flag: {fmtPct(contam * 100, 1)}</label>
            <input id="contam" type="range" min="0.005" max="0.2" step="0.005" value={contam} onChange={e => setContam(Number(e.target.value))} style={{ accentColor: 'var(--accent)' }} />
            <span className="hint">This is an assumption you choose, not an estimate of how many errors exist.</span>
          </div>
          <Button variant="primary" busy={run.busy} onClick={() => run.run().catch(() => {})}>Detect anomalies</Button>
        </div>
        {run.error && <div style={{ marginTop: 10 }}><ErrorState error={run.error} compact /></div>}
      </Panel>
      {res && (
        <>
          <div className="grid-4">
            <Metric label="Rows flagged" value={fmtNum(res.n_anomalies)} sub={`${fmtPct(res.anomaly_rate_pct)} of ${fmtNum(res.rows_scored)}`} />
            <Metric label="Features used" value={res.features_used.length} />
          </div>
          <div className="grid-2" style={{ alignItems: 'start' }}>
            <Panel title="Anomaly score distribution" subtitle="Rows to the right of the threshold are flagged."><SimpleBars data={hist} x="s" y="c" yLabel="Rows" height={200} /></Panel>
            <Panel title="Columns that stand out in flagged rows" subtitle="Mean absolute robust z-score, flagged vs normal rows.">
              <DataTable columns={[{ key: 'column', label: 'Column' }, { key: 'mean_abs_z_anomalies', label: 'Flagged', numeric: true, render: v => v.toFixed(2) }, { key: 'mean_abs_z_normal', label: 'Normal', numeric: true, render: v => v.toFixed(2) }]} rows={res.column_signals} />
            </Panel>
          </div>
          <Panel title="Most unusual rows" subtitle="Reasons list the features furthest from typical values.">
            <DataTable rowKey={r => r.row} maxHeight={420} columns={[{ key: 'row', label: 'Row', numeric: true, render: v => Number(v) + 1 }, { key: 'score', label: 'Score', numeric: true, render: v => v.toFixed(3) },
              { key: 'reasons', label: 'Why', render: v => v.map(x => `${x.feature} ${x.direction} (z ${x.robust_z.toFixed(1)})`).join(', ') },
              ...Object.keys(res.top_anomalies[0]?.values || {}).slice(0, 6).map(k => ({ key: k, label: k, render: (_, r) => r.values[k] }))]} rows={res.top_anomalies} />
            <ul className="tiny muted" style={{ margin: '8px 0 0', paddingLeft: 18 }}>{res.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </Panel>
        </>
      )}
    </div>
  )
}
