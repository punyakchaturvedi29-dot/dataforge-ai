import { useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Combine, Layers, Merge, Trash2, UploadCloud } from 'lucide-react'
import { api } from '../api/endpoints'
import { useAction, useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { Badge, Button, Callout, Check, Confirm, DataTable, Empty, ErrorState, Input, LoadingBlock, Metric, Panel, Segmented, Select, Tabs } from '../components/ui'
import { fmtNum, fmtPct, plural } from '../lib/format'

export default function Sources() {
  const { ws } = useWorkspace()
  const [tab, setTab] = useState('files')
  const n = ws.datasets.length
  return (
    <div className="page">
      <PageHeader title="Sources & joins" description="Combine related files. Join keys are recommended from the values themselves — overlap, uniqueness and row coverage — and every join is validated before it is created." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'files', label: 'Files', badge: n }, { id: 'schema', label: 'Compare schemas' }, { id: 'join', label: 'Join', icon: Merge }, { id: 'stack', label: 'Stack', icon: Layers }]} />
      {tab === 'files' && <Files />}
      {tab !== 'files' && n < 2 && <Empty icon={Combine} title="Add a second file first">Joining and stacking need at least two datasets in this workspace.</Empty>}
      {tab === 'schema' && n >= 2 && <Schema />}
      {tab === 'join' && n >= 2 && <Join />}
      {tab === 'stack' && n >= 2 && <Stack />}
    </div>
  )
}

function Files() {
  const { ws, id, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const [del, setDel] = useState(null)
  const add = useAction(async file => {
    const r = await api.addSource(id, file)
    await dataChanged()
    notify(`Added ${r.dataset.name} (${plural(r.dataset.rows, 'row')})`, { kind: 'success' })
  })
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ multiple: false, noClick: true, onDrop: f => f[0] && add.run(f[0]).catch(() => {}) })
  const activate = async did => { try { await api.activateDataset(id, did); await dataChanged(); notify('Active dataset changed', { kind: 'success' }) } catch (e) { notify(e.message, { kind: 'error' }) } }
  return (
    <div className="stack">
      <div {...getRootProps({ className: `dropzone ${isDragActive ? 'active' : ''}` })} onClick={open} role="button" tabIndex={0} aria-label="Add a related file" onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && open()}>
        <input {...getInputProps()} aria-label="Choose related file" />
        <UploadCloud size={24} aria-hidden style={{ color: 'var(--accent-text)' }} />
        <h3 style={{ marginTop: 6 }}>{add.busy ? 'Adding…' : 'Drop a related file (e.g. customers for an orders table)'}</h3>
      </div>
      {add.error && <ErrorState error={add.error} compact />}
      <Panel title="Datasets in this workspace" subtitle="Analyses run on the active dataset.">
        <DataTable rowKey={d => d.id} columns={[
          { key: 'name', label: 'Name', render: (v, d) => <span>{v} {d.id === ws.active_dataset.id && <Badge tone="accent">active</Badge>}</span> },
          { key: 'source', label: 'Origin', render: (v, d) => v === 'join' ? `join of ${d.lineage.sources.map(s => s.name).join(' + ')}` : v === 'stack' ? `stack of ${d.lineage.sources.length} datasets` : 'upload' },
          { key: 'rows', label: 'Rows', numeric: true }, { key: 'columns', label: 'Columns', numeric: true }, { key: 'n_ops', label: 'Cleaning steps', numeric: true },
          { key: 'id', label: '', render: (v, d) => d.id === ws.active_dataset.id ? null : <div className="row"><Button size="sm" onClick={() => activate(v)}>Make active</Button><Button size="sm" variant="ghost" icon={Trash2} aria-label={`Remove ${d.name}`} onClick={() => setDel(d)} /></div> },
        ]} rows={ws.datasets} />
      </Panel>
      {del && <Confirm title="Remove dataset?" danger confirmLabel="Remove" onCancel={() => setDel(null)} onConfirm={async () => {
        try { await api.deleteDataset(id, del.id); await dataChanged(); notify('Dataset removed', { kind: 'success' }) } catch (e) { notify(e.message, { kind: 'error' }) }
        setDel(null)
      }}><p>“{del.name}” and its cleaning history will be removed from this workspace.</p></Confirm>}
    </div>
  )
}

function Schema() {
  const { id, ws } = useWorkspace()
  const { data, error } = useApi(() => api.compareSources(id), [id, ws.datasets.length])
  if (error) return <ErrorState error={error} />
  if (!data) return <LoadingBlock rows={1} h={240} />
  return (
    <Panel title="Columns across datasets" subtitle={`${data.common_columns.length} column(s) appear in every dataset.`}>
      <DataTable rowKey={r => r.column} maxHeight={560} columns={[
        { key: 'column', label: 'Column', render: (v, r) => <span>{v} {r.type_conflict && <Badge tone="warn" title="Different types across datasets">type conflict</Badge>}</span> },
        ...data.datasets.map(d => ({ key: d.id, label: `${d.name} (${fmtNum(d.rows)})`, render: (_, r) => r.per_dataset[d.id] ? `${r.per_dataset[d.id].logical_type}${r.per_dataset[d.id].missing_pct ? `, ${r.per_dataset[d.id].missing_pct.toFixed(0)}% missing` : ''}` : <span className="null">absent</span> })),
      ]} rows={data.columns} />
    </Panel>
  )
}

function Join() {
  const { id, ws, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const ds = ws.datasets
  const [left, setLeft] = useState(ws.active_dataset.id)
  const [right, setRight] = useState(ds.find(d => d.id !== ws.active_dataset.id)?.id)
  const [pick, setPick] = useState(null)
  const [how, setHow] = useState('left')
  const [ci, setCi] = useState(false)
  const [ack, setAck] = useState(false)
  const [name, setName] = useState('')
  const { data: cands, error, loading } = useApi(() => api.joinCandidates(id, left, right), [id, left, right], { enabled: !!left && !!right && left !== right })
  useEffect(() => { setPick(cands?.candidates?.[0] ? { left_on: cands.candidates[0].left_on, right_on: cands.candidates[0].right_on } : null) }, [cands])
  const body = pick && { left_id: left, right_id: right, left_on: pick.left_on, right_on: pick.right_on, how, case_insensitive: ci }
  const { data: pv, error: pvErr, loading: pvLoading } = useApi(() => api.joinPreview(id, body), [JSON.stringify(body)], { enabled: !!body })
  useEffect(() => setAck(false), [JSON.stringify(body)]) // eslint-disable-line
  const commit = useAction(async () => {
    const r = await api.join(id, { ...body, confirm: ack, name: name || undefined, activate: true })
    await dataChanged()
    notify(`Created “${r.dataset.name}” (${plural(r.dataset.rows, 'row')}) and made it active`, { kind: 'success', persist: true })
  })
  const opts = ds.map(d => ({ value: d.id, label: `${d.name} (${fmtNum(d.rows)} rows)` }))
  const leftCols = ds.find(d => d.id === left)
  return (
    <div className="stack">
      <Panel title="Datasets">
        <div className="form-row">
          <Select label="Left (kept in a left join)" value={left} onChange={setLeft} options={opts} />
          <Select label="Right" value={right} onChange={setRight} options={opts} />
          <div className="field"><span className="label">Join type</span>
            <Segmented label="Join type" value={how} onChange={setHow} options={[{ value: 'left', label: 'Left' }, { value: 'inner', label: 'Inner' }, { value: 'outer', label: 'Outer' }, { value: 'right', label: 'Right' }]} />
          </div>
          <Check label="Ignore letter case in keys" checked={ci} onChange={setCi} />
        </div>
        {left === right && <p className="small" style={{ color: 'var(--warn)', marginTop: 8 }}>Choose two different datasets.</p>}
      </Panel>
      {error && <ErrorState error={error} />}
      {loading && <LoadingBlock rows={1} h={160} />}
      {cands && (
        <Panel title="Recommended join keys" subtitle="Ranked by value evidence; column names count for little.">
          {cands.candidates.length === 0 ? <Callout tone="warn">No column pair shares enough values to be a join key.</Callout> : (
            <ul className="list-plain stack-sm">
              {cands.candidates.map((c, i) => {
                const on = pick?.left_on === c.left_on && pick?.right_on === c.right_on
                return (
                  <li key={i}>
                    <label className="panel-flat row" style={{ alignItems: 'flex-start', cursor: 'pointer', borderColor: on ? 'var(--accent)' : undefined }}>
                      <input type="radio" name="key" checked={on} onChange={() => setPick({ left_on: c.left_on, right_on: c.right_on })} style={{ marginTop: 4, accentColor: 'var(--accent)' }} />
                      <span className="grow">
                        <strong>{c.left_on} = {c.right_on}</strong> <Badge tone={c.cardinality === 'N:M' ? 'warn' : 'info'}>{c.cardinality}</Badge> <span className="tiny muted num">score {c.score.toFixed(2)}</span>
                        <ul className="small secondary" style={{ margin: '4px 0 0', paddingLeft: 18 }}>{c.reasons.map((r, j) => <li key={j}>{r}</li>)}</ul>
                      </span>
                    </label>
                  </li>
                )
              })}
            </ul>
          )}
          {leftCols && <p className="tiny muted" style={{ marginTop: 8 }}>Need a different pair? Keys must hold comparable values in both datasets.</p>}
        </Panel>
      )}
      {pvErr && <ErrorState error={pvErr} />}
      {pvLoading && <LoadingBlock rows={1} h={140} />}
      {pv && !pvLoading && (
        <Panel title="Join validation" subtitle={`${pv.left_on} = ${pv.right_on}, ${pv.how} join`}>
          <div className="grid-4" style={{ marginBottom: 12 }}>
            <Metric label="Rows" value={`${fmtNum(pv.rows_left)} → ${fmtNum(pv.rows_result)}`} sub={`×${pv.explosion_factor.toFixed(2)} vs left`} tone={pv.explosion_factor > 1.05 ? 'warn' : undefined} />
            <Metric label="Left rows matched" value={fmtPct(pv.match_rate_left * 100)} sub={`${fmtNum(pv.unmatched_left_rows)} without a match`} />
            <Metric label="Unmatched right rows" value={fmtNum(pv.unmatched_right_rows)} />
            <Metric label="Relationship" value={pv.cardinality} sub={`duplicate keys: ${pv.duplicate_keys_left} left, ${pv.duplicate_keys_right} right`} />
          </div>
          {pv.warnings.length > 0 && <Callout tone={pv.requires_confirmation ? 'warn' : 'info'}><ul style={{ margin: 0, paddingLeft: 18 }}>{pv.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Callout>}
          {(pv.unmatched_left_samples.length > 0 || pv.unmatched_right_samples.length > 0) && (
            <p className="small muted" style={{ marginTop: 8 }}>Unmatched keys — left: {pv.unmatched_left_samples.slice(0, 6).join(', ') || 'none'}; right: {pv.unmatched_right_samples.slice(0, 6).join(', ') || 'none'}.</p>
          )}
          <div style={{ marginTop: 10 }}><DataTable columns={pv.preview_columns.map(c => ({ key: c, label: c }))} rows={pv.preview} maxHeight={260} /></div>
          <div className="form-row" style={{ marginTop: 12 }}>
            <Input label="Name for the joined dataset" placeholder={`${ds.find(d => d.id === left)?.name} ⋈ ${ds.find(d => d.id === right)?.name}`} value={name} onChange={e => setName(e.target.value)} />
            {pv.requires_confirmation && <Check label="I reviewed the warnings and want this join" checked={ack} onChange={setAck} />}
            <Button variant="primary" icon={Merge} busy={commit.busy} disabled={pv.requires_confirmation && !ack} onClick={() => commit.run().catch(e => notify(e.message, { kind: 'error' }))}>Create joined dataset</Button>
          </div>
        </Panel>
      )}
    </div>
  )
}

function Stack() {
  const { id, ws, dataChanged } = useWorkspace()
  const { notify } = useUi()
  const [sel, setSel] = useState(ws.datasets.slice(0, 2).map(d => d.id))
  const [mode, setMode] = useState('union')
  const run = useAction(async () => {
    const r = await api.stack(id, { ids: sel, mode, activate: true })
    await dataChanged()
    notify(`Stacked into “${r.dataset.name}” (${plural(r.dataset.rows, 'row')})`, { kind: 'success' })
    return r
  })
  const [res, setRes] = useState(null)
  return (
    <Panel title="Stack datasets (append rows)" subtitle="For files with the same structure, e.g. one export per month. A column records which file each row came from.">
      <div className="stack-sm">
        {ws.datasets.map(d => <Check key={d.id} label={`${d.name} (${fmtNum(d.rows)} rows × ${d.columns} columns)`} checked={sel.includes(d.id)} onChange={on => setSel(s => on ? [...s, d.id] : s.filter(x => x !== d.id))} />)}
        <div className="field"><span className="label">Columns</span>
          <Segmented label="Columns" value={mode} onChange={setMode} options={[{ value: 'union', label: 'All columns (fill gaps)' }, { value: 'intersection', label: 'Shared columns only' }]} />
        </div>
        {run.error && <ErrorState error={run.error} compact />}
        <div><Button variant="primary" icon={Layers} busy={run.busy} disabled={sel.length < 2} onClick={() => run.run().then(setRes).catch(() => {})}>Stack {sel.length} datasets</Button></div>
        {res?.report?.warnings?.length > 0 && <Callout tone="warn"><ul style={{ margin: 0, paddingLeft: 18 }}>{res.report.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Callout>}
      </div>
    </Panel>
  )
}
