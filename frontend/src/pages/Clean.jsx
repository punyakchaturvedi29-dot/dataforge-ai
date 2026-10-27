import { useMemo, useState } from 'react'
import { CheckCircle2, Eye, History, ListChecks, Play, RotateCcw, ShieldCheck, Trash2, Undo2, Wand2 } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { Badge, Button, Callout, Check, Confirm, DataTable, Empty, ErrorState, Input, LoadingBlock, Modal, Panel, Select, Tabs } from '../components/ui'
import { fmtCell, fmtNum, plural, timeAgo } from '../lib/format'

const SEV = { high: 'bad', medium: 'warn', low: undefined, info: 'info' }
const OP_LABEL = {
  drop_duplicates: 'Remove duplicate rows', drop_empty_rows: 'Remove empty rows', drop_rows_missing: 'Remove rows with missing values',
  drop_columns: 'Drop columns', rename_column: 'Rename column', impute: 'Fill missing values', convert_type: 'Convert data type',
  trim_whitespace: 'Trim whitespace', standardize_case: 'Standardize letter case', map_values: 'Merge / relabel values',
  replace_null_tokens: 'Treat placeholder text as missing', clip_outliers: 'Cap outliers (winsorise)', remove_outliers: 'Remove outlier rows',
  remove_rows_where: 'Remove rows matching a condition', nullify_where: 'Mark values matching a condition as missing', split_number_unit: 'Split number and unit',
}

export function impactText(imp) {
  const p = []
  if (imp.rows_removed) p.push(`${plural(imp.rows_removed, 'row')} removed`)
  if (imp.columns_removed?.length) p.push(`${plural(imp.columns_removed.length, 'column')} removed`)
  if (imp.columns_added?.length) p.push(`${plural(imp.columns_added.length, 'column')} added`)
  if (imp.cells_modified) p.push(`${plural(imp.cells_modified, 'value')} changed`)
  if (imp.values_filled) p.push(`${plural(imp.values_filled, 'missing value')} filled`)
  if (imp.values_nullified) p.push(`${plural(imp.values_nullified, 'value')} set to missing`)
  return p.join(', ') || 'no changes'
}

export default function Clean() {
  const [tab, setTab] = useState('suggestions')
  const { ws } = useWorkspace()
  return (
    <div className="page">
      <PageHeader title="Clean" description="Preview every change before it runs. Steps are applied to a copy of the original, recorded in order, and can be undone." />
      <Tabs value={tab} onChange={setTab} tabs={[
        { id: 'suggestions', label: 'Suggestions', icon: Wand2 }, { id: 'manual', label: 'Add a step', icon: Play },
        { id: 'history', label: 'History & lineage', icon: History, badge: ws.active_dataset.n_ops || undefined },
        { id: 'rules', label: 'Quality rules', icon: ListChecks },
      ]} />
      {tab === 'suggestions' && <Suggestions />}
      {tab === 'manual' && <ManualStep />}
      {tab === 'history' && <HistoryView />}
      {tab === 'rules' && <Rules />}
    </div>
  )
}

/** Preview → (confirm) → apply dialog shared by suggestions and the manual builder. */
export function OpPreview({ op, params, title, onClose, onApplied }) {
  const { id, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const { data: pv, error, loading } = useApi(() => api.previewOp(id, { op, params }), [id, op, JSON.stringify(params)])
  const [ack, setAck] = useState(false)
  const apply = useAction(async () => {
    const r = await api.applyOp(id, { op, params, confirm: true })
    await dataChanged()
    notify(`${r.applied.label}: ${impactText(r.applied.impact)}`, { kind: 'success' })
    onApplied?.(r)
    onClose()
  })
  const imp = pv?.impact
  const needs = imp?.requires_confirmation
  return (
    <Modal size="lg" title={title || OP_LABEL[op]} onClose={onClose} footer={<>
      <Button variant="ghost" onClick={onClose}>Cancel</Button>
      <Button variant={needs ? 'danger' : 'primary'} busy={apply.busy} disabled={!pv || (needs && !ack)} onClick={() => apply.run().catch(() => {})}>
        {needs ? 'Apply and remove data' : 'Apply step'}
      </Button>
    </>}>
      {loading && <LoadingBlock rows={2} h={80} />}
      {error && <ErrorState error={error} compact />}
      {apply.error && <ErrorState error={apply.error} compact />}
      {pv && (
        <div className="stack">
          <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
            <div className="panel-flat"><div className="small muted">Rows</div><div className="num" style={{ fontWeight: 600 }}>{fmtNum(imp.rows_before)} → {fmtNum(imp.rows_after)}</div></div>
            <div className="panel-flat"><div className="small muted">Values changed</div><div className="num" style={{ fontWeight: 600 }}>{fmtNum(imp.cells_modified)}</div></div>
            <div className="panel-flat"><div className="small muted">Missing values filled</div><div className="num" style={{ fontWeight: 600 }}>{fmtNum(imp.values_filled)}</div></div>
            <div className="panel-flat"><div className="small muted">Non-missing values lost</div><div className="num" style={{ fontWeight: 600, color: imp.non_null_values_lost ? 'var(--bad)' : undefined }}>{fmtNum(imp.non_null_values_lost)}</div></div>
          </div>
          <p className="small">{impactText(imp)}. {pv.applies_to_new_data ? 'This step will also be applied to new data at prediction time.' : 'This step is not applied to new data at prediction time.'}</p>
          {pv.notes?.map((n, i) => <Callout key={i} tone="info">{n}</Callout>)}
          {Object.keys(pv.fitted || {}).length > 0 && <p className="small muted">Values fitted for this step: {Object.entries(pv.fitted).map(([k, v]) => `${k} = ${typeof v === 'object' ? JSON.stringify(v) : fmtCell(v)}`).join('; ')}</p>}
          {pv.sample.changed_rows.length > 0 && (
            <div>
              <h4 style={{ marginBottom: 6 }}>Before and after (sample)</h4>
              <DataTable columns={[{ key: 'row', label: 'Row', numeric: true, render: v => Number(v) + 1 },
                ...pv.sample.columns.map(c => ({ key: c, label: c, render: (_, r) => <span><span className="diff-before">{fmtCell(r.before[c]) ?? 'missing'}</span>{' → '}<span className="diff-after">{fmtCell(r.after[c]) ?? 'missing'}</span></span> }))]}
                rows={pv.sample.changed_rows} maxHeight={240} />
            </div>
          )}
          {pv.sample.removed_rows.length > 0 && (
            <div>
              <h4 style={{ marginBottom: 6 }}>Rows that would be removed (sample)</h4>
              <DataTable columns={[{ key: 'row', label: 'Row', numeric: true, render: v => Number(v) + 1 }, ...Object.keys(pv.sample.removed_rows[0].values).map(c => ({ key: c, label: c, render: (_, r) => fmtCell(r.values[c]) }))]}
                rows={pv.sample.removed_rows} maxHeight={220} />
            </div>
          )}
          {needs && (
            <Callout tone="warn" title="This step removes data">
              <Check checked={ack} onChange={setAck} label={`I understand that ${impactText(imp)}. The original upload is kept and the step can be undone.`} />
            </Callout>
          )}
        </div>
      )}
    </Modal>
  )
}

function Suggestions() {
  const { id, dataKey, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const { data, error, loading, reload } = useApi(() => api.suggestions(id), [id, dataKey])
  const [preview, setPreview] = useState(null)
  const [confirmSafe, setConfirmSafe] = useState(false)
  const safe = useAction(async () => {
    const r = await api.applySafe(id)
    await dataChanged()
    notify(`Applied ${plural(r.applied.length, 'safe fix', 'safe fixes')}`, { kind: 'success' })
    setConfirmSafe(false)
  })
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data) return <LoadingBlock rows={3} h={90} />
  const list = data.suggestions
  const safeOnes = list.filter(s => s.safe && s.recommended)
  const actionable = list.filter(s => s.recommended || s.alternatives?.length)
  const notes = list.filter(s => !s.recommended && !s.alternatives?.length)
  return (
    <div className="stack">
      {list.length === 0 && <Empty icon={CheckCircle2} title="No issues detected">The checks found no missing values, duplicates, type problems or inconsistent labels.</Empty>}
      {safeOnes.length > 0 && (
        <Callout tone="info" title={`${plural(safeOnes.length, 'fix', 'fixes')} can be applied without losing information`}
          action={<Button variant="primary" size="sm" icon={ShieldCheck} onClick={() => setConfirmSafe(true)}>Apply safe fixes</Button>}>
          {safeOnes.map(s => s.title).join('; ')}.
        </Callout>
      )}
      {actionable.map(s => <SuggestionCard key={s.id} s={s} onPreview={(op, title) => setPreview({ ...op, title })} />)}
      {notes.length > 0 && (
        <Panel title="Notes" subtitle="Detected characteristics that need no change.">
          <ul className="list-plain stack-sm">{notes.map(s => <li key={s.id} className="small"><Badge tone="info">note</Badge> <strong>{s.title}.</strong> <span className="secondary">{s.description}</span></li>)}</ul>
        </Panel>
      )}
      {preview && <OpPreview op={preview.op} params={preview.params} title={preview.title} onClose={() => setPreview(null)} onApplied={reload} />}
      {confirmSafe && (
        <Confirm title="Apply safe fixes?" confirmLabel="Apply" busy={safe.busy} onConfirm={() => safe.run().catch(e => notify(e.message, { kind: 'error' }))} onCancel={() => setConfirmSafe(false)}>
          <p>These steps only trim spaces, recognise placeholder text as missing, convert values that convert without loss, and remove empty rows/columns:</p>
          <ul>{safeOnes.map(s => <li key={s.id}>{s.title}</li>)}</ul>
        </Confirm>
      )}
    </div>
  )
}

function opSummary(o) {
  const p = o.params || {}
  if (o.op === 'impute') return `${OP_LABEL.impute}: ${p.strategy}${p.value !== undefined ? ` “${p.value}”` : ''}`
  if (o.op === 'convert_type') return `Convert to ${p.to}`
  if (o.op === 'map_values') return `Relabel ${Object.keys(p.mapping || {}).length} value(s)`
  if (o.op === 'clip_outliers' || o.op === 'remove_outliers') return `${OP_LABEL[o.op]} (${p.method}, k=${p.k})`
  if (o.op === 'remove_rows_where' || o.op === 'nullify_where') return `${OP_LABEL[o.op]}`
  return OP_LABEL[o.op] || o.op
}

function SuggestionCard({ s, onPreview }) {
  const options = [s.recommended, ...(s.alternatives || [])].filter(Boolean)
  const [choice, setChoice] = useState(0)
  const ev = Object.entries(s.evidence || {}).filter(([, v]) => typeof v !== 'object' || v === null).slice(0, 4)
  return (
    <article className="finding">
      <div className="finding-head">
        <div className="grow">
          <h3 style={{ fontSize: '0.98rem' }}>{s.title}</h3>
          <p className="secondary small" style={{ marginTop: 3 }}>{s.description}</p>
        </div>
        <div className="row"><Badge tone={SEV[s.severity]}>{s.severity}</Badge>{s.safe && <Badge tone="good">no information lost</Badge>}</div>
      </div>
      {ev.length > 0 && <p className="tiny muted num">{ev.map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'number' ? fmtNum(v) : v}`).join(' · ')}</p>}
      {s.consequence && <p className="small"><span className="muted">Consequence: </span>{s.consequence}</p>}
      <div className="row">
        {options.length > 1 ? (
          <select className="select" style={{ width: 'auto', minWidth: 260 }} aria-label="Choose a fix" value={choice} onChange={e => setChoice(Number(e.target.value))}>
            {options.map((o, i) => <option key={i} value={i}>{i === 0 && s.recommended ? 'Recommended: ' : ''}{opSummary(o)}</option>)}
          </select>
        ) : <span className="small">{s.recommended ? 'Recommended: ' : ''}{opSummary(options[0])}</span>}
        <Button size="sm" icon={Eye} onClick={() => onPreview(options[choice], s.title)}>Preview</Button>
        {!s.recommended && <span className="tiny muted">No automatic fix is recommended — investigate first.</span>}
      </div>
    </article>
  )
}

function ColumnPicker({ columns, value, onChange, label, single }) {
  const [f, setF] = useState('')
  if (single) return <Select label={label} value={value} onChange={onChange} placeholder="Choose a column…" options={columns} />
  const shown = columns.filter(c => c.toLowerCase().includes(f.toLowerCase()))
  const sel = new Set(value || [])
  return (
    <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
      <legend className="label" style={{ marginBottom: 5 }}>{label}</legend>
      <input className="input" placeholder="Filter" value={f} onChange={e => setF(e.target.value)} aria-label={`Filter ${label}`} />
      <div className="panel-flat" style={{ maxHeight: 160, overflowY: 'auto', marginTop: 6, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 4 }}>
        {shown.map(c => <Check key={c} label={<span className="truncate">{c}</span>} checked={sel.has(c)} onChange={on => onChange(on ? [...sel, c] : [...sel].filter(x => x !== c))} />)}
      </div>
      <span className="hint">{sel.size ? `${sel.size} selected` : 'None selected'}</span>
    </fieldset>
  )
}

function parseValue(v) {
  if (v === '' || v === undefined) return v
  const n = Number(v)
  return Number.isFinite(n) && String(v).trim() !== '' ? n : v
}

function ManualStep() {
  const { id, dataKey } = useWorkspace()
  const { data: sch } = useApi(() => api.schema(id), [id, dataKey])
  const [op, setOp] = useState('impute')
  const [p, setP] = useState({ strategy: 'median', method: 'iqr', k: 1.5, keep: 'first', how: 'any', to: 'numeric', case: 'title', operator: '==' })
  const [preview, setPreview] = useState(null)
  const cols = (sch?.columns || []).map(c => c.name)
  const set = (k, v) => setP(s => ({ ...s, [k]: v }))
  const params = useMemo(() => {
    const listOps = ['in', 'not_in', 'between', 'outside']
    const val = listOps.includes(p.operator) ? String(p.value || '').split(',').map(x => parseValue(x.trim())).filter(x => x !== '') : parseValue(p.value)
    switch (op) {
      case 'drop_duplicates': return { subset: p.columns?.length ? p.columns : null, keep: p.keep }
      case 'drop_empty_rows': return {}
      case 'drop_rows_missing': return { columns: p.columns?.length ? p.columns : null, how: p.how }
      case 'drop_columns': return { columns: p.columns || [] }
      case 'rename_column': return { column: p.column, new_name: p.new_name || '' }
      case 'impute': return { columns: p.columns || [], strategy: p.strategy, ...(p.strategy === 'constant' ? { value: parseValue(p.value ?? '') } : {}), add_indicator: !!p.add_indicator }
      case 'convert_type': return { column: p.column, to: p.to, ...(p.dayfirst ? { dayfirst: p.dayfirst === 'yes' } : {}) }
      case 'trim_whitespace': return { columns: p.columns?.length ? p.columns : null }
      case 'standardize_case': return { columns: p.columns || [], case: p.case }
      case 'map_values': {
        const mapping = {}
        String(p.mapping || '').split('\n').forEach(l => { const m = l.split('=>'); if (m.length === 2 && m[0].trim()) mapping[m[0].trim()] = m[1].trim() })
        return { column: p.column, mapping }
      }
      case 'replace_null_tokens': return { columns: p.columns?.length ? p.columns : null, ...(p.tokens ? { tokens: p.tokens.split(',').map(t => t.trim()) } : {}) }
      case 'clip_outliers': case 'remove_outliers': return { column: p.column, method: p.method, k: Number(p.k) }
      case 'remove_rows_where': case 'nullify_where': return { column: p.column, operator: p.operator, ...(['is_null', 'not_null'].includes(p.operator) ? {} : { value: val }) }
      case 'split_number_unit': return { column: p.column }
      default: return {}
    }
  }, [op, p])
  const needsCols = ['drop_columns', 'impute', 'standardize_case'].includes(op)
  const singleCol = ['rename_column', 'convert_type', 'map_values', 'clip_outliers', 'remove_outliers', 'remove_rows_where', 'nullify_where', 'split_number_unit'].includes(op)
  const optionalCols = ['drop_duplicates', 'drop_rows_missing', 'trim_whitespace', 'replace_null_tokens'].includes(op)
  const ready = (!needsCols || params.columns?.length) && (!singleCol || params.column) && (op !== 'map_values' || Object.keys(params.mapping || {}).length)
  return (
    <Panel title="Add a cleaning step" subtitle="Choose an operation, set its parameters, then preview the exact impact before applying it.">
      <div className="stack">
        <Select label="Operation" value={op} onChange={v => setOp(v)} options={Object.entries(OP_LABEL).map(([value, label]) => ({ value, label }))} />
        {(needsCols || optionalCols) && <ColumnPicker label={optionalCols ? 'Columns (leave empty for all)' : 'Columns'} columns={cols} value={p.columns} onChange={v => set('columns', v)} />}
        {singleCol && <ColumnPicker single label="Column" columns={cols} value={p.column} onChange={v => set('column', v)} />}
        <div className="form-row">
          {op === 'drop_duplicates' && <Select label="Keep" value={p.keep} onChange={v => set('keep', v)} options={[{ value: 'first', label: 'First occurrence' }, { value: 'last', label: 'Last occurrence' }]} />}
          {op === 'drop_rows_missing' && <Select label="Remove a row when" value={p.how} onChange={v => set('how', v)} options={[{ value: 'any', label: 'any selected column is missing' }, { value: 'all', label: 'all selected columns are missing' }]} />}
          {op === 'rename_column' && <Input label="New name" value={p.new_name || ''} onChange={e => set('new_name', e.target.value)} />}
          {op === 'impute' && <>
            <Select label="Strategy" value={p.strategy} onChange={v => set('strategy', v)} options={[
              { value: 'median', label: 'Median (numeric, robust)' }, { value: 'mean', label: 'Mean (numeric)' }, { value: 'mode', label: 'Most frequent value' },
              { value: 'constant', label: 'A constant value' }, { value: 'knn', label: 'K-nearest neighbours (numeric)' }, { value: 'interpolate', label: 'Linear interpolation (ordered data)' },
              { value: 'ffill', label: 'Carry previous value forward' }, { value: 'bfill', label: 'Carry next value backward' }]} />
            {p.strategy === 'constant' && <Input label="Value" value={p.value ?? ''} onChange={e => set('value', e.target.value)} />}
            <Check label="Add a “was missing” indicator column" checked={p.add_indicator} onChange={v => set('add_indicator', v)} />
          </>}
          {op === 'convert_type' && <>
            <Select label="Convert to" value={p.to} onChange={v => set('to', v)} options={['numeric', 'integer', 'datetime', 'text', 'category', 'boolean']} />
            {p.to === 'datetime' && <Select label="Day before month (31/12/2024)?" value={p.dayfirst || ''} onChange={v => set('dayfirst', v)} placeholder="Detect automatically" options={[{ value: 'yes', label: 'Yes' }, { value: 'no', label: 'No' }]} />}
          </>}
          {op === 'standardize_case' && <Select label="Case" value={p.case} onChange={v => set('case', v)} options={[{ value: 'title', label: 'Title Case' }, { value: 'lower', label: 'lower case' }, { value: 'upper', label: 'UPPER CASE' }]} />}
          {op === 'replace_null_tokens' && <Input label="Placeholder values (comma separated, optional)" placeholder="N/A, -, unknown" value={p.tokens || ''} onChange={e => set('tokens', e.target.value)} />}
          {(op === 'clip_outliers' || op === 'remove_outliers') && <>
            <Select label="Method" value={p.method} onChange={v => set('method', v)} options={[{ value: 'iqr', label: 'IQR fences' }, { value: 'zscore', label: 'z-score' }, { value: 'mad', label: 'Robust z (MAD)' }, ...(op === 'clip_outliers' ? [{ value: 'percentile', label: '1st–99th percentile' }] : [])]} />
            <Input label="Threshold k" type="number" step="0.5" min="0.5" value={p.k} onChange={e => set('k', e.target.value)} />
          </>}
          {(op === 'remove_rows_where' || op === 'nullify_where') && <>
            <Select label="Condition" value={p.operator} onChange={v => set('operator', v)} options={[
              { value: '==', label: 'equals' }, { value: '!=', label: 'does not equal' }, { value: '>', label: 'greater than' }, { value: '<', label: 'less than' },
              { value: '>=', label: 'at least' }, { value: '<=', label: 'at most' }, { value: 'in', label: 'is one of (comma list)' }, { value: 'not_in', label: 'is not one of' },
              { value: 'between', label: 'between (low, high)' }, { value: 'outside', label: 'outside (low, high)' }, { value: 'is_null', label: 'is missing' },
              { value: 'not_null', label: 'is not missing' }, { value: 'matches', label: 'matches pattern (regex)' }]} />
            {!['is_null', 'not_null'].includes(p.operator) && <Input label="Value" value={p.value ?? ''} onChange={e => set('value', e.target.value)} />}
          </>}
        </div>
        {op === 'map_values' && (
          <div className="field">
            <label htmlFor="mapping">Mapping (one per line: old =&gt; new)</label>
            <textarea id="mapping" className="textarea" rows={5} placeholder={'NY => New York\nN.Y. => New York'} value={p.mapping || ''} onChange={e => set('mapping', e.target.value)} />
          </div>
        )}
        <div><Button variant="primary" icon={Eye} disabled={!ready} onClick={() => setPreview({ op, params })}>Preview impact</Button></div>
      </div>
      {preview && <OpPreview op={preview.op} params={preview.params} onClose={() => setPreview(null)} />}
    </Panel>
  )
}

function HistoryView() {
  const { id, dataKey, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const { data, error, loading, reload } = useApi(() => api.history(id), [id, dataKey])
  const [confirm, setConfirm] = useState(null)
  const act = useAction(async fn => { await fn(); await dataChanged(); setConfirm(null) })
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data) return <LoadingBlock rows={2} h={80} />
  const steps = data.steps
  const ops = steps.slice(1)
  const lin = steps[0].lineage
  return (
    <div className="stack">
      <div className="row">
        <Button icon={Undo2} disabled={!ops.length} busy={act.busy} onClick={() => act.run(async () => { const r = await api.undo(id); notify(`Undid “${r.undone.label}”`, { kind: 'success' }) }).catch(e => notify(e.message, { kind: 'error' }))}>Undo last step</Button>
        <Button variant="danger-ghost" icon={RotateCcw} disabled={!ops.length} onClick={() => setConfirm({ op_id: null, label: 'the original upload' })}>Reset to original</Button>
        <span className="grow" />
        <span className="small muted num">Now {fmtNum(data.current.rows)} rows × {data.current.columns} columns</span>
      </div>
      <Panel title="Lineage" subtitle="How the current working data was produced, step by step.">
        <ol className="timeline">
          <li>
            <strong>{steps[0].label}</strong> <span className="secondary small">{steps[0].detail}</span>
            {lin?.type === 'join' && <div className="small muted">Joined {lin.sources.map(s => `“${s.name}” (v${s.version}, ${fmtNum(s.rows)} rows)`).join(' and ')} on {lin.left_on} = {lin.right_on} ({lin.how}).</div>}
            {lin?.type === 'stack' && <div className="small muted">Stacked {lin.sources.map(s => `“${s.name}”`).join(', ')} ({lin.mode}).</div>}
          </li>
          {ops.map(s => (
            <li key={s.id}>
              <div className="row-between">
                <div className="grow">
                  <strong>{s.step}. {s.label}</strong> <span className="small secondary">— {s.detail}</span>
                  <div className="tiny muted">{Object.entries(s.params || {}).map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ').slice(0, 220)}</div>
                  {s.notes?.map((n, i) => <div key={i} className="tiny muted">{n}</div>)}
                </div>
                <div className="row">
                  {s.applies_to_new_data ? <Badge tone="info" title="Replayed on prediction inputs">applies to new data</Badge> : <Badge>training data only</Badge>}
                  <span className="tiny muted">{timeAgo(s.at)}</span>
                  {s.step < ops.length && <Button size="sm" variant="ghost" onClick={() => setConfirm({ op_id: s.id, label: `step ${s.step}` })}>Revert to here</Button>}
                </div>
              </div>
            </li>
          ))}
        </ol>
        {!ops.length && <p className="small muted">No cleaning steps yet. The working data is identical to the original upload.</p>}
      </Panel>
      {confirm && (
        <Confirm title="Revert cleaning steps?" danger confirmLabel="Revert" busy={act.busy} onCancel={() => setConfirm(null)}
          onConfirm={() => act.run(async () => { const r = await api.revert(id, { op_id: confirm.op_id, confirm: true }); notify(`Reverted ${plural(r.removed, 'step')}`, { kind: 'success' }) }).catch(e => notify(e.message, { kind: 'error' }))}>
          <p>Steps after {confirm.label} will be removed and the data rebuilt from the original upload. Models already trained keep working with their recorded steps.</p>
        </Confirm>
      )}
    </div>
  )
}

const RULE_TYPES = [
  { value: 'not_null', label: 'Value must be present' }, { value: 'unique', label: 'Values must be unique' },
  { value: 'range', label: 'Within a range' }, { value: 'allowed_values', label: 'One of a list of values' },
  { value: 'pattern', label: 'Matches a format' }, { value: 'max_missing_pct', label: 'Missing share at most' },
  { value: 'compare_columns', label: 'Compared with another column' }, { value: 'max_length', label: 'Text length at most' },
]

function Rules() {
  const { id, dataKey } = useWorkspace()
  const { notify } = useUi()
  const { data: sch } = useApi(() => api.schema(id), [id, dataKey])
  const { data, reload } = useApi(() => api.rules(id), [id])
  const [result, setResult] = useState(null)
  const [r, setR] = useState({ type: 'not_null', operator: '<=', preset: 'email' })
  const cols = (sch?.columns || []).map(c => c.name)
  const add = useAction(async () => {
    const rule = { type: r.type, column: r.column }
    if (r.type === 'range') { if (r.min !== undefined && r.min !== '') rule.min = parseValue(r.min); if (r.max !== undefined && r.max !== '') rule.max = parseValue(r.max) }
    if (r.type === 'allowed_values') rule.values = String(r.values || '').split(',').map(v => v.trim()).filter(Boolean)
    if (r.type === 'pattern') { if (r.preset === 'custom') rule.pattern = r.pattern; else rule.preset = r.preset }
    if (r.type === 'max_missing_pct') rule.threshold = Number(r.threshold)
    if (r.type === 'compare_columns') { rule.other_column = r.other; rule.operator = r.operator }
    if (r.type === 'max_length') rule.max_length = Number(r.max_length)
    await api.addRule(id, rule)
    notify('Rule added', { kind: 'success' })
    reload()
  })
  const evaluate = useAction(async () => setResult(await api.evaluateRules(id)))
  return (
    <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.5fr)', alignItems: 'start' }}>
      <Panel title="New rule" subtitle="Rules describe what valid data looks like; they are checked on demand and included in reports.">
        <div className="stack-sm">
          <Select label="Rule" value={r.type} onChange={v => setR(s => ({ ...s, type: v }))} options={RULE_TYPES} />
          <Select label="Column" value={r.column} onChange={v => setR(s => ({ ...s, column: v }))} placeholder="Choose a column…" options={cols} />
          {r.type === 'range' && <div className="form-row"><Input label="Minimum" value={r.min ?? ''} onChange={e => setR(s => ({ ...s, min: e.target.value }))} /><Input label="Maximum" value={r.max ?? ''} onChange={e => setR(s => ({ ...s, max: e.target.value }))} /></div>}
          {r.type === 'allowed_values' && <Input label="Allowed values (comma separated)" value={r.values ?? ''} onChange={e => setR(s => ({ ...s, values: e.target.value }))} />}
          {r.type === 'pattern' && <>
            <Select label="Format" value={r.preset} onChange={v => setR(s => ({ ...s, preset: v }))} options={[{ value: 'email', label: 'Email address' }, { value: 'url', label: 'URL' }, { value: 'phone', label: 'Phone number' },
              { value: 'postal_code_us', label: 'US ZIP code' }, { value: 'iso_date', label: 'ISO date (YYYY-MM-DD)' }, { value: 'alphanumeric', label: 'Letters and digits only' }, { value: 'custom', label: 'Custom regular expression' }]} />
            {r.preset === 'custom' && <Input label="Regular expression" hint="Max 200 characters; nested quantifiers like (a+)+ are rejected." value={r.pattern ?? ''} onChange={e => setR(s => ({ ...s, pattern: e.target.value }))} />}
          </>}
          {r.type === 'max_missing_pct' && <Input label="Maximum missing %" type="number" value={r.threshold ?? ''} onChange={e => setR(s => ({ ...s, threshold: e.target.value }))} />}
          {r.type === 'max_length' && <Input label="Maximum characters" type="number" value={r.max_length ?? ''} onChange={e => setR(s => ({ ...s, max_length: e.target.value }))} />}
          {r.type === 'compare_columns' && <div className="form-row">
            <Select label="Must be" value={r.operator} onChange={v => setR(s => ({ ...s, operator: v }))} options={['<', '<=', '>', '>=', '==', '!=']} />
            <Select label="Other column" value={r.other} onChange={v => setR(s => ({ ...s, other: v }))} placeholder="Choose…" options={cols} />
          </div>}
          {add.error && <ErrorState error={add.error} compact />}
          <div><Button variant="primary" busy={add.busy} disabled={!r.column} onClick={() => add.run().catch(() => {})}>Add rule</Button></div>
        </div>
      </Panel>
      <Panel title="Rules" actions={<Button icon={Play} busy={evaluate.busy} disabled={!data?.rules?.length} onClick={() => evaluate.run().catch(e => notify(e.message, { kind: 'error' }))}>Check rules</Button>}>
        {!data?.rules?.length ? <Empty icon={ListChecks} title="No rules yet">Add a rule, for example “age must be between 0 and 120”.</Empty> : (
          <ul className="list-plain stack-sm">
            {data.rules.map(rule => {
              const res = result?.results?.find(x => x.rule.id === rule.id)
              return (
                <li key={rule.id} className="panel-flat">
                  <div className="row-between">
                    <span className="grow"><strong>{rule.name}</strong></span>
                    {res && <Badge tone={res.status === 'pass' ? 'good' : res.status === 'fail' ? 'bad' : 'warn'}>{res.status === 'pass' ? 'passed' : res.status === 'fail' ? `${fmtNum(res.violations)} violation(s)` : res.status}</Badge>}
                    <Button size="sm" variant="ghost" icon={Trash2} aria-label={`Delete rule ${rule.name}`} onClick={async () => { await api.deleteRule(id, rule.id); reload(); setResult(null) }} />
                  </div>
                  {res?.message && <div className="small muted">{res.message}</div>}
                  {res?.sample?.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <DataTable maxHeight={180} columns={Object.keys(res.sample[0]).map(k => ({ key: k, label: k === 'row' ? 'Row' : k, numeric: k === 'row', render: k === 'row' ? v => Number(v) + 1 : undefined }))} rows={res.sample} />
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
        {result && <p className="small muted" style={{ marginTop: 10 }}>{result.rules_passed} of {result.rules_total} rules passed · {fmtNum(result.rows_with_violations)} of {fmtNum(result.rows)} rows violate at least one rule.</p>}
      </Panel>
    </div>
  )
}
