import { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Link, useNavigate } from 'react-router-dom'
import { Database, FileSpreadsheet, Moon, Sun, Trash2, UploadCloud } from 'lucide-react'
import { api } from '../api/endpoints'
import { useUi } from '../state/ui'
import { Badge, Button, Callout, Confirm, Empty, Panel, Progress, Select } from '../components/ui'
import { bytes, plural, timeAgo } from '../lib/format'

const EXT = ['csv', 'tsv', 'txt', 'xlsx', 'xls', 'json', 'jsonl', 'ndjson']
const STEPS = [
  ['Upload', 'CSV, Excel or JSON; encoding, delimiter and sheet are detected.'],
  ['Validate', 'Malformed rows, duplicate headers and type problems are reported, never silently fixed.'],
  ['Profile', 'Distributions, outliers, missing values and correlations for every column.'],
  ['Clean', 'Preview each fix, confirm anything destructive, undo at any time.'],
  ['Analyze', 'Ranked findings with evidence, diagnostics and forecasts.'],
  ['Model', 'Cross-validated AutoML with a held-out test set and leakage checks.'],
  ['Visualize', 'Charts chosen for the data type, with honest axes and labels.'],
  ['Export', 'PDF, Markdown or JSON reports, cleaned data and predictions.'],
]

export function validateFile(file, cfg) {
  const ext = (file.name.split('.').pop() || '').toLowerCase()
  if (!file.name.includes('.') || !EXT.includes(ext)) return `“.${ext}” files are not supported. Use CSV, TSV, TXT, XLSX, XLS, JSON or JSONL.`
  if (file.size === 0) return 'The file is empty.'
  const limit = (cfg?.max_upload_mb ?? 100) * 1024 * 1024
  if (file.size > limit) return `The file is ${bytes(file.size)}; the limit on this server is ${cfg?.max_upload_mb ?? 100} MB.`
  return null
}

export default function Home() {
  const navigate = useNavigate()
  const { theme, setTheme, notify, recent, forgetWorkspace } = useUi()
  const [cfg, setCfg] = useState(null)
  const [server, setServer] = useState(null)
  const [state, setState] = useState({ phase: 'idle' })
  const [toDelete, setToDelete] = useState(null)

  useEffect(() => {
    api.config().then(setCfg).catch(() => setCfg(null))
    api.listWorkspaces().then(r => setServer(r.workspaces)).catch(() => setServer(null))
  }, [])

  const doUpload = useCallback(async (file, sheet) => {
    setState({ phase: 'uploading', file, progress: 0 })
    try {
      const ws = await api.uploadFile(file, { sheet }, p => setState(s => ({ ...s, progress: p, phase: p >= 1 ? 'processing' : 'uploading' })))
      notify(`Loaded ${ws.active_dataset.name}: ${plural(ws.active_dataset.rows, 'row')}`, { kind: 'success' })
      navigate(`/w/${ws.id}/data`)
    } catch (e) {
      setState({ phase: 'error', file, error: e })
    }
  }, [navigate, notify])

  const onDrop = useCallback(async (accepted, rejected) => {
    const file = accepted[0] || rejected[0]?.file
    if (!file) return
    const problem = validateFile(file, cfg)
    if (problem) { setState({ phase: 'error', file, error: { message: problem } }); return }
    const ext = file.name.split('.').pop().toLowerCase()
    if (ext === 'xlsx' || ext === 'xls') {
      setState({ phase: 'inspecting', file })
      try {
        const info = await api.inspectFile(file)
        if (info.sheets?.length > 1) { setState({ phase: 'sheet', file, sheets: info.sheets, sheet: info.sheets[0] }); return }
      } catch (e) { setState({ phase: 'error', file, error: e }); return }
    }
    doUpload(file)
  }, [cfg, doUpload])

  const busy = ['uploading', 'processing', 'inspecting'].includes(state.phase)
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({ onDrop, multiple: false, disabled: busy, noClick: true })

  const sample = async () => {
    setState({ phase: 'processing', file: { name: 'sample_data.csv' } })
    try { const ws = await api.loadSample(); navigate(`/w/${ws.id}`) } catch (e) { setState({ phase: 'error', error: e }) }
  }

  const list = server ?? recent.map(r => ({ id: r.id, name: r.name, updated_at: r.at / 1000, active_dataset: { rows: r.rows } }))
  const remove = async () => {
    try {
      await api.deleteWorkspace(toDelete.id)
      forgetWorkspace(toDelete.id)
      setServer(s => s?.filter(x => x.id !== toDelete.id))
      notify('Workspace deleted', { kind: 'success' })
    } catch (e) { notify(e.message, { kind: 'error' }) }
    setToDelete(null)
  }
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches)

  return (
    <div style={{ minHeight: '100vh' }}>
      <header className="topbar" style={{ position: 'static' }}>
        <Link to="/" className="brand" style={{ padding: 0 }}>
          <span className="brand-mark" aria-hidden><svg width="16" height="16" viewBox="0 0 32 32"><path d="M9 22V10h7.5a6 6 0 0 1 0 12z" fill="none" stroke="#fff" strokeWidth="3" /><path d="M13 16h9" stroke="#9fe3d6" strokeWidth="3" /></svg></span>
          DataForge AI
        </Link>
        <span className="grow" />
        <Button variant="ghost" icon={dark ? Sun : Moon} aria-label="Toggle theme" onClick={() => setTheme(dark ? 'light' : 'dark')} />
      </header>
      <main className="page" id="main">
        <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)', alignItems: 'start' }}>
          <div className="stack">
            <div>
              <h1 style={{ fontSize: '1.9rem' }}>Start an analysis</h1>
              <p className="secondary" style={{ marginTop: 6, maxWidth: '60ch' }}>
                Drop a file to create a workspace. Your original data is kept unchanged; every cleaning step, model and result is recorded so you can retrace it.
              </p>
            </div>
            <div {...getRootProps({ className: `dropzone ${isDragActive ? 'active' : ''} ${busy ? 'busy' : ''}` })}
              onClick={busy || state.phase === 'sheet' ? undefined : open} role="button" tabIndex={0} aria-label="Upload a data file"
              onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); open() } }}>
              <input {...getInputProps()} aria-label="Choose file" data-testid="file-input" />
              <UploadCloud size={30} aria-hidden style={{ color: 'var(--accent-text)' }} />
              {state.phase === 'idle' || state.phase === 'error' ? (
                <>
                  <h2 style={{ marginTop: 10 }}>{isDragActive ? 'Drop the file to upload' : 'Drop a data file here'}</h2>
                  <p className="secondary" style={{ marginTop: 4 }}>or <span style={{ color: 'var(--accent-text)', fontWeight: 600 }}>browse your computer</span></p>
                  <p className="small muted" style={{ marginTop: 10 }}>
                    CSV, TSV, Excel (.xlsx, .xls), JSON or JSON Lines · up to {cfg?.max_upload_mb ?? 100} MB and {(cfg?.max_rows ?? 2000000).toLocaleString()} rows
                  </p>
                </>
              ) : state.phase === 'sheet' ? (
                <div className="stack-sm" style={{ maxWidth: 360, margin: '12px auto 0', textAlign: 'left' }} onClick={e => e.stopPropagation()}>
                  <h2><FileSpreadsheet size={18} aria-hidden /> {state.file.name}</h2>
                  <Select label="This workbook has several sheets. Which one should be analysed?" value={state.sheet}
                    onChange={v => setState(s => ({ ...s, sheet: v }))} options={state.sheets} />
                  <div className="row">
                    <Button variant="primary" onClick={() => doUpload(state.file, state.sheet)}>Load “{state.sheet}”</Button>
                    <Button variant="ghost" onClick={() => setState({ phase: 'idle' })}>Cancel</Button>
                  </div>
                </div>
              ) : (
                <div style={{ maxWidth: 360, margin: '12px auto 0' }} aria-live="polite">
                  <h2>{state.file?.name}</h2>
                  <p className="secondary small" style={{ margin: '6px 0 10px' }}>
                    {state.phase === 'inspecting' ? 'Reading workbook sheets…' : state.phase === 'uploading' ? `Uploading ${Math.round((state.progress || 0) * 100)}%` : 'Validating and profiling the data…'}
                  </p>
                  <Progress value={state.phase === 'uploading' ? state.progress * 0.7 : 0.85} label="Upload progress" />
                </div>
              )}
            </div>
            {state.phase === 'error' && (
              <Callout tone="bad" title={`Could not load ${state.file?.name || 'the file'}`} action={<Button size="sm" onClick={() => setState({ phase: 'idle' })}>Dismiss</Button>}>
                {state.error.message}
              </Callout>
            )}
            <div className="row">
              <Button icon={Database} onClick={sample} disabled={busy}>Use the sample retail dataset</Button>
              <span className="small muted">100 orders with dates, regions, products and revenue.</span>
            </div>
            <section aria-labelledby="flow" style={{ marginTop: 8 }}>
              <h2 id="flow" style={{ fontSize: '1rem', marginBottom: 10 }}>What happens next</h2>
              <ol className="list-plain" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))', gap: 10 }}>
                {STEPS.map(([t, d], i) => (
                  <li key={t} className="panel-flat" style={{ display: 'flex', gap: 10 }}>
                    <span className="num" style={{ fontWeight: 700, color: 'var(--accent-text)', minWidth: 14 }}>{i + 1}</span>
                    <span><strong>{t}</strong><br /><span className="small secondary">{d}</span></span>
                  </li>
                ))}
              </ol>
            </section>
          </div>
          <Panel title="Workspaces" subtitle={server ? 'Stored on this server, most recent first.' : 'Recently opened in this browser.'}>
            {list.length === 0 ? (
              <Empty icon={Database} title="No workspaces yet">Upload a file or try the sample dataset to create one.</Empty>
            ) : (
              <ul className="list-plain stack-sm">
                {list.slice(0, 12).map(w => (
                  <li key={w.id} className="row-between panel-flat">
                    <Link to={`/w/${w.id}`} className="grow" style={{ textDecoration: 'none', color: 'inherit', minWidth: 0 }}>
                      <div className="truncate" style={{ fontWeight: 600 }}>{w.name}</div>
                      <div className="small muted num">
                        {w.active_dataset?.rows != null && `${plural(w.active_dataset.rows, 'row')} · `}
                        {w.n_models ? `${plural(w.n_models, 'model')} · ` : ''}updated {timeAgo(w.updated_at)}
                      </div>
                    </Link>
                    {w.n_datasets > 1 && <Badge>{w.n_datasets} files</Badge>}
                    <Button variant="ghost" size="sm" icon={Trash2} aria-label={`Delete ${w.name}`} onClick={() => setToDelete(w)} />
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </main>
      {toDelete && (
        <Confirm title="Delete workspace?" danger confirmLabel="Delete permanently" onConfirm={remove} onCancel={() => setToDelete(null)}>
          <p>“{toDelete.name}” and all of its data, models and results will be removed from the server. This cannot be undone.</p>
        </Confirm>
      )}
    </div>
  )
}
