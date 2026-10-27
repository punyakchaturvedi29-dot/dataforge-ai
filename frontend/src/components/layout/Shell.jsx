import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom'
import {
  Bell, BarChart3, Bot, BrainCircuit, Check, ChevronsLeft, ChevronsRight, CircleHelp, Database, FileText, Files,
  FlaskConical, Gauge, LayoutDashboard, LineChart, Menu, Moon, Search, Settings, ShieldCheck, Sparkles, Sun, Table2,
  Target, TrendingUp, Upload, Wand2,
} from 'lucide-react'
import { api } from '../../api/endpoints'
import { useUi } from '../../state/ui'
import { useWorkspace } from '../../state/workspace'
import { Badge, Button, Check as CheckBox, Confirm, Input, Modal, Segmented } from '../ui'
import { plural, timeAgo } from '../../lib/format'

export const NAV = [
  { group: 'Workspace', items: [
    { to: '', label: 'Overview', icon: Gauge, end: true },
    { to: 'copilot', label: 'Copilot', icon: Bot },
  ] },
  { group: 'Prepare', items: [
    { to: 'data', label: 'Data', icon: Table2 },
    { to: 'profile', label: 'Profile', icon: BarChart3 },
    { to: 'clean', label: 'Clean', icon: Wand2 },
    { to: 'sources', label: 'Sources & joins', icon: Files },
  ] },
  { group: 'Analyze', items: [
    { to: 'insights', label: 'Insights', icon: Sparkles },
    { to: 'visualize', label: 'Visualize', icon: LayoutDashboard },
    { to: 'diagnose', label: 'Diagnose', icon: Target },
    { to: 'forecast', label: 'Forecast', icon: TrendingUp },
  ] },
  { group: 'Model', items: [
    { to: 'model', label: 'Train models', icon: BrainCircuit },
    { to: 'predict', label: 'Predict', icon: FlaskConical },
  ] },
  { group: 'Deliver', items: [
    { to: 'report', label: 'Report', icon: FileText },
  ] },
]
const ALL = NAV.flatMap(g => g.items)

export const STAGES = [
  { id: 'upload', label: 'Upload', page: 'data' },
  { id: 'validate', label: 'Validate', page: 'data' },
  { id: 'profile', label: 'Profile', page: 'profile' },
  { id: 'clean', label: 'Clean', page: 'clean' },
  { id: 'analyze', label: 'Analyze', page: 'insights' },
  { id: 'model', label: 'Model', page: 'model' },
  { id: 'visualize', label: 'Visualize', page: 'visualize' },
  { id: 'export', label: 'Export', page: 'report' },
]
const PAGE_STAGE = { data: 'validate', profile: 'profile', clean: 'clean', insights: 'analyze', diagnose: 'analyze', forecast: 'analyze', model: 'model', predict: 'model', visualize: 'visualize', report: 'export' }

const HELP = {
  '': 'The overview summarises the active dataset: quality, key metrics, the latest model and forecasts, and what to do next. All numbers are computed from your data.',
  data: 'Browse every row with server-side paging, sorting and search. Switch between the working (cleaned) copy and the original upload, which is never modified.',
  profile: 'Descriptive statistics, distributions, outliers and correlations for every column. p-values are adjusted for multiple comparisons.',
  clean: 'Cleaning steps are previewed before they run, recorded in a history you can undo, and replayed on new data at prediction time. Steps that remove rows or values ask for confirmation.',
  sources: 'Add related files, compare schemas and join them. Join keys are recommended from value overlap and uniqueness; joins that would duplicate rows need confirmation.',
  insights: 'Findings ranked by evidence and impact. "Association" means a statistical relationship, not cause and effect.',
  visualize: 'An automatically chosen chart set plus a chart builder that refuses misleading combinations (e.g. pies of negative numbers).',
  diagnose: 'Investigate what is associated with an outcome: segment gaps with confounder checks, period-over-period decomposition and a multivariate driver model.',
  forecast: 'Methods are compared by rolling-origin backtesting that only trains on past data. Intervals are approximate; their backtest coverage is shown.',
  model: 'AutoML compares models with cross-validation on a training split, tunes the best within a time budget and evaluates once on a held-out test set next to a baseline.',
  predict: 'Score new rows with a trained model. Inputs are validated against the training schema and the recorded cleaning steps are applied first.',
  copilot: 'Ask questions in plain language. Answers come from deterministic analyses and show their evidence; the copilot never runs code.',
  report: 'Build a report from computed results. Choose sections, findings and charts, then export PDF, Markdown or JSON.',
}

export function useCurrentPage() {
  const loc = useLocation()
  const m = /^\/w\/[^/]+\/?([^/]*)/.exec(loc.pathname)
  return m ? m[1] : null
}

export function Shell({ children }) {
  const { collapsed, setCollapsed } = useUi()
  const { ws, id } = useWorkspace()
  const [mobile, setMobile] = useState(false)
  const loc = useLocation()
  useEffect(() => setMobile(false), [loc.pathname])
  return (
    <div className={`shell ${collapsed ? 'collapsed' : ''} ${mobile ? 'mobile-open' : ''}`}>
      <a href="#main" className="sr-only">Skip to content</a>
      <aside className="sidebar" aria-label="Primary">
        <Link to="/" className="brand" title="DataForge AI home">
          <span className="brand-mark" aria-hidden>
            <svg width="16" height="16" viewBox="0 0 32 32"><path d="M9 22V10h7.5a6 6 0 0 1 0 12z" fill="none" stroke="#fff" strokeWidth="3" /><path d="M13 16h9" stroke="#9fe3d6" strokeWidth="3" /></svg>
          </span>
          <span className="brand-name">DataForge AI</span>
        </Link>
        {ws?.active_dataset && (
          <div className="side-card">
            <div className="truncate" style={{ fontWeight: 600 }} title={ws.active_dataset.name}>{ws.active_dataset.name}</div>
            <div className="muted num">{plural(ws.active_dataset.rows, 'row')} · {ws.active_dataset.columns} columns</div>
            {ws.active_dataset.n_ops > 0 && <div className="muted">{plural(ws.active_dataset.n_ops, 'cleaning step')}</div>}
          </div>
        )}
        <nav aria-label="Workspace sections">
          {NAV.map(g => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map(it => (
                <NavLink key={it.to} to={`/w/${id}${it.to ? '/' + it.to : ''}`} end={it.end} className="nav-link" title={collapsed ? it.label : undefined}>
                  <it.icon size={17} aria-hidden /><span className="nav-label">{it.label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div style={{ marginTop: 'auto', paddingTop: 12 }}>
          <Link to="/" className="nav-link" title="Upload another dataset"><Upload size={17} aria-hidden /><span className="nav-label">New dataset</span></Link>
          <button type="button" className="nav-link" style={{ border: 0, background: 'none', width: '100%', cursor: 'pointer' }}
            onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <ChevronsRight size={17} /> : <ChevronsLeft size={17} />}<span className="nav-label">Collapse</span>
          </button>
        </div>
      </aside>
      <div className="main">
        <Topbar onMenu={() => setMobile(m => !m)} />
        <main id="main" tabIndex={-1} style={{ outline: 'none' }}>{children}</main>
      </div>
    </div>
  )
}

function Topbar({ onMenu }) {
  const { ws, id } = useWorkspace()
  const page = useCurrentPage()
  const item = ALL.find(i => i.to === page)
  const [palette, setPalette] = useState(false)
  const [help, setHelp] = useState(false)
  const [settings, setSettings] = useState(false)
  useEffect(() => {
    const k = e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette(p => !p) } }
    document.addEventListener('keydown', k)
    return () => document.removeEventListener('keydown', k)
  }, [])
  return (
    <header className="topbar">
      <Button variant="ghost" icon={Menu} className="mobile-only" aria-label="Open navigation" onClick={onMenu} />
      <nav className="crumbs grow" aria-label="Breadcrumb">
        <Link to="/">Workspaces</Link><span aria-hidden>/</span>
        <Link to={`/w/${id}`} className="truncate" style={{ maxWidth: 260 }}>{ws?.name || '…'}</Link>
        {item && item.to && <><span aria-hidden>/</span><span className="current" aria-current="page">{item.label}</span></>}
      </nav>
      <button type="button" className="search-btn" onClick={() => setPalette(true)} aria-label="Search pages, columns and actions">
        <Search size={15} aria-hidden /><span className="search-text grow" style={{ textAlign: 'left' }}>Search…</span><span className="kbd">⌘K</span>
      </button>
      <Notifications />
      <Button variant="ghost" icon={CircleHelp} aria-label="Help for this page" onClick={() => setHelp(true)} />
      <ThemeToggle />
      <Button variant="ghost" icon={Settings} aria-label="Settings" onClick={() => setSettings(true)} />
      {palette && <CommandPalette onClose={() => setPalette(false)} />}
      {help && (
        <Modal title={`Help: ${item?.label || 'Overview'}`} onClose={() => setHelp(false)}>
          <p className="secondary">{HELP[page ?? ''] || HELP['']}</p>
          <hr className="divider" style={{ margin: '14px 0' }} />
          <p className="small muted">Workflow: Upload → Validate → Profile → Clean → Analyze → Model → Visualize → Export. Completed stages are marked in the progress rail; you can revisit any stage at any time.</p>
        </Modal>
      )}
      {settings && <SettingsModal onClose={() => setSettings(false)} />}
    </header>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useUi()
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia?.('(prefers-color-scheme: dark)').matches)
  return <Button variant="ghost" icon={dark ? Sun : Moon} aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} onClick={() => setTheme(dark ? 'light' : 'dark')} />
}

function Notifications() {
  const { notifications, setNotifications } = useUi()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const navigate = useNavigate()
  const unread = notifications.filter(n => !n.read).length
  useEffect(() => {
    if (!open) return
    const close = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <Button variant="ghost" icon={Bell} aria-label={`Notifications${unread ? ` (${unread} unread)` : ''}`} aria-expanded={open}
        onClick={() => { setOpen(o => !o); setNotifications(n => n.map(x => ({ ...x, read: true }))) }} />
      {unread > 0 && <span aria-hidden style={{ position: 'absolute', top: 6, right: 6, width: 8, height: 8, borderRadius: 4, background: 'var(--accent)' }} />}
      {open && (
        <div className="tooltip" style={{ position: 'absolute', right: 0, top: 40, width: 320, padding: 0, zIndex: 40 }}>
          <div className="row-between" style={{ padding: '10px 12px', borderBottom: '1px solid var(--line)' }}>
            <strong>Notifications</strong>
            {notifications.length > 0 && <Button size="sm" variant="ghost" onClick={() => setNotifications([])}>Clear</Button>}
          </div>
          <ul className="list-plain" style={{ maxHeight: 320, overflowY: 'auto' }}>
            {notifications.length === 0 && <li className="muted small" style={{ padding: 14 }}>Background jobs and exports will show up here.</li>}
            {notifications.map(n => (
              <li key={n.id}>
                <button type="button" className="palette-item" style={{ width: '100%', border: 0, background: 'none', textAlign: 'left' }}
                  onClick={() => { if (n.link) navigate(n.link); setOpen(false) }}>
                  <span className="grow small">{n.message}<br /><span className="tiny muted">{timeAgo(n.at / 1000)}</span></span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function CommandPalette({ onClose }) {
  const { id, ws } = useWorkspace()
  const { setTheme, theme } = useUi()
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const [sel, setSel] = useState(0)
  const [cols, setCols] = useState([])
  useEffect(() => { api.schema(id).then(s => setCols(s.columns)).catch(() => {}) }, [id])
  const items = useMemo(() => {
    const pages = ALL.map(p => ({ kind: 'Page', label: p.label, icon: p.icon, run: () => navigate(`/w/${id}${p.to ? '/' + p.to : ''}`) }))
    const columns = cols.map(c => ({ kind: `Column · ${c.logical_type}`, label: c.name, icon: Database, run: () => navigate(`/w/${id}/profile?column=${encodeURIComponent(c.name)}`) }))
    const actions = [
      { kind: 'Action', label: 'Upload a new dataset', icon: Upload, run: () => navigate('/') },
      { kind: 'Action', label: 'Download cleaned data (CSV)', icon: Database, run: () => api.downloadData(id, { version: 'working', format: 'csv' }) },
      { kind: 'Action', label: 'Ask the copilot', icon: Bot, run: () => navigate(`/w/${id}/copilot`) },
      { kind: 'Action', label: `Use ${theme === 'dark' ? 'light' : 'dark'} theme`, icon: Moon, run: () => setTheme(theme === 'dark' ? 'light' : 'dark') },
    ]
    const all = [...pages, ...actions, ...columns]
    const s = q.trim().toLowerCase()
    return (s ? all.filter(i => i.label.toLowerCase().includes(s)) : all).slice(0, 40)
  }, [q, cols, id, navigate, setTheme, theme, ws])
  useEffect(() => setSel(0), [q])
  const go = it => { onClose(); it?.run() }
  return (
    <div className="palette" onMouseDown={e => e.target === e.currentTarget && onClose()} role="dialog" aria-modal="true" aria-label="Search">
      <div className="palette-box">
        <input autoFocus placeholder="Search pages, columns and actions" value={q} onChange={e => setQ(e.target.value)}
          aria-label="Search" aria-controls="palette-list" aria-activedescendant={`pi-${sel}`}
          onKeyDown={e => {
            if (e.key === 'Escape') onClose()
            if (e.key === 'ArrowDown') { e.preventDefault(); setSel(s => Math.min(s + 1, items.length - 1)) }
            if (e.key === 'ArrowUp') { e.preventDefault(); setSel(s => Math.max(s - 1, 0)) }
            if (e.key === 'Enter') go(items[sel])
          }} />
        <div className="palette-list" role="listbox" id="palette-list">
          {items.length === 0 && <div className="muted small" style={{ padding: 12 }}>No matches.</div>}
          {items.map((it, i) => (
            <div key={it.kind + it.label} id={`pi-${i}`} role="option" aria-selected={i === sel} className="palette-item"
              onMouseEnter={() => setSel(i)} onClick={() => go(it)}>
              <it.icon size={16} aria-hidden /><span className="grow truncate">{it.label}</span><span className="tiny muted">{it.kind}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function SettingsModal({ onClose }) {
  const { theme, setTheme, collapsed, setCollapsed, notify, forgetWorkspace } = useUi()
  const { ws, id, refresh } = useWorkspace()
  const navigate = useNavigate()
  const [name, setName] = useState(ws?.name || '')
  const [cfg, setCfg] = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api.config().then(setCfg).catch(() => {}) }, [])
  const save = async () => {
    try { await api.patchWorkspace(id, { name }); await refresh(); notify('Workspace renamed', { kind: 'success' }) } catch (e) { notify(e.message, { kind: 'error' }) }
  }
  const remove = async () => {
    setBusy(true)
    try { await api.deleteWorkspace(id); forgetWorkspace(id); notify('Workspace deleted', { kind: 'success' }); navigate('/') } catch (e) { notify(e.message, { kind: 'error' }) } finally { setBusy(false) }
  }
  return (
    <Modal title="Settings" onClose={onClose}>
      <div className="stack">
        <section className="stack-sm">
          <h3>Appearance</h3>
          <Segmented label="Theme" value={theme} onChange={setTheme} options={[{ value: 'system', label: 'System' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
          <CheckBox label="Collapse the sidebar" checked={collapsed} onChange={setCollapsed} />
        </section>
        <hr className="divider" />
        <section className="stack-sm">
          <h3>Workspace</h3>
          <div className="form-row">
            <Input label="Name" value={name} onChange={e => setName(e.target.value)} maxLength={120} />
            <Button onClick={save} disabled={!name.trim() || name === ws?.name}>Rename</Button>
          </div>
          {ws && <p className="small muted">Kept until {new Date(ws.expires_at * 1000).toLocaleDateString()} ({cfg?.workspace_ttl_days ?? '…'} days after the last change). Original files, cleaned data, models and results are stored on this server only.</p>}
          <div><Button variant="danger-ghost" onClick={() => setConfirmDelete(true)}>Delete workspace…</Button></div>
        </section>
        {cfg && (
          <>
            <hr className="divider" />
            <section className="stack-sm">
              <h3 className="row"><ShieldCheck size={16} aria-hidden />Server</h3>
              <dl className="kv small">
                <dt>Version</dt><dd>{cfg.version}</dd>
                <dt>Upload limit</dt><dd>{cfg.max_upload_mb} MB, {cfg.max_rows.toLocaleString()} rows, {cfg.max_columns} columns</dd>
                <dt>Training</dt><dd>up to {cfg.training_max_rows.toLocaleString()} rows, {cfg.training_time_budget_s}s budget</dd>
                <dt>Optional models</dt><dd>{Object.entries(cfg.models).filter(([, v]) => v).map(([k]) => k.replace(/_/g, ' ')).join(', ')}</dd>
                <dt>AI explanations</dt><dd>{cfg.llm_enabled ? <Badge tone="good"><Check size={12} />Enabled ({cfg.llm_model})</Badge> : <span>Off — set ANTHROPIC_API_KEY on the server to enable</span>}</dd>
              </dl>
            </section>
          </>
        )}
      </div>
      {confirmDelete && (
        <Confirm title="Delete this workspace?" danger confirmLabel="Delete permanently" busy={busy} onConfirm={remove} onCancel={() => setConfirmDelete(false)}>
          <p>This permanently deletes the uploaded files, cleaned data, models, predictions and reports in “{ws?.name}”. It cannot be undone.</p>
        </Confirm>
      )}
    </Modal>
  )
}

export function WorkflowRail() {
  const { stages, id } = useWorkspace()
  const page = useCurrentPage()
  const here = PAGE_STAGE[page]
  const done = Object.fromEntries((stages || []).map(s => [s.id, s.completed]))
  return (
    <nav className="rail" aria-label="Analysis workflow">
      {STAGES.map(s => (
        <Link key={s.id} to={`/w/${id}/${s.page}`} className={`rail-step ${done[s.id] ? 'done' : ''} ${here === s.id ? 'here' : ''}`}
          aria-current={here === s.id ? 'step' : undefined}>
          <span className="rail-name">
            <span className="rail-dot" aria-hidden>{done[s.id] && <Check size={11} strokeWidth={3} />}</span>{s.label}
          </span>
          <span className="rail-state">{done[s.id] ? 'Done' : here === s.id ? 'You are here' : 'Not started'}</span>
        </Link>
      ))}
    </nav>
  )
}

export function PageHeader({ title, description, actions, rail = true }) {
  return (
    <>
      {rail && <WorkflowRail />}
      <div className="page-head">
        <div className="grow">
          <h1>{title}</h1>
          {description && <p>{description}</p>}
        </div>
        {actions && <div className="row">{actions}</div>}
      </div>
    </>
  )
}
