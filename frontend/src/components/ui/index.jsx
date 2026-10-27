import { useEffect, useId, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, CircleHelp, Info, Loader2, X, XCircle } from 'lucide-react'
import { fmtCell } from '../../lib/format'

export function Button({ variant = 'secondary', size, icon: Icon, busy, children, className = '', ...rest }) {
  const cls = ['btn', `btn-${variant}`, size && `btn-${size}`, !children && Icon && 'btn-icon', className].filter(Boolean).join(' ')
  return (
    <button type="button" className={cls} disabled={busy || rest.disabled} aria-busy={busy || undefined} {...rest}>
      {busy ? <Loader2 size={15} className="spin" aria-hidden /> : Icon && <Icon size={15} aria-hidden />}
      {children}
    </button>
  )
}

export function Panel({ title, subtitle, actions, children, pad = true, className = '', ...rest }) {
  return (
    <section className={`panel ${className}`} {...rest}>
      {(title || actions) && (
        <div className="panel-head">
          <div className="grow">
            {title && <h3>{title}</h3>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </div>
      )}
      <div className={pad ? 'panel-body' : ''}>{children}</div>
    </section>
  )
}

export function Badge({ tone, children, title, className = '' }) {
  return <span className={`badge ${tone ? `badge-${tone}` : ''} ${className}`} title={title}>{children}</span>
}

const CALLOUT_ICON = { info: Info, warn: AlertTriangle, bad: XCircle, good: CheckCircle2 }
export function Callout({ tone = 'info', title, children, action }) {
  const Icon = CALLOUT_ICON[tone]
  return (
    <div className={`callout callout-${tone}`} role={tone === 'bad' ? 'alert' : 'status'}>
      <Icon size={16} className={`callout-icon-${tone}`} aria-hidden />
      <div className="grow">
        {title && <div style={{ fontWeight: 600 }}>{title}</div>}
        <div>{children}</div>
      </div>
      {action}
    </div>
  )
}

export function ErrorState({ error, onRetry, compact }) {
  if (!error) return null
  const msg = error.message || String(error)
  return (
    <Callout tone="bad" title={compact ? null : 'Something went wrong'} action={onRetry && <Button size="sm" onClick={onRetry}>Retry</Button>}>
      {msg}
      {error.requestId && <span className="tiny muted"> (request {error.requestId})</span>}
    </Callout>
  )
}

export function Empty({ icon: Icon = Info, title, children, action }) {
  return (
    <div className="empty">
      <div className="empty-icon"><Icon size={22} aria-hidden /></div>
      {title && <h3>{title}</h3>}
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function Skeleton({ h = 16, w = '100%', style }) {
  return <div className="skel" style={{ height: h, width: w, ...style }} aria-hidden />
}
export function LoadingBlock({ rows = 3, h = 120 }) {
  return (
    <div className="stack" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => <Skeleton key={i} h={h} />)}
    </div>
  )
}

export function Metric({ label, value, sub, help, tone }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}{help && <Help text={help} />}</div>
      <div className="metric-value num" style={tone ? { color: `var(--${tone})` } : undefined}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

export function Help({ text, label = 'More information' }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const close = e => { if (!ref.current?.contains(e.target)) setOpen(false) }
    const esc = e => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc) }
  }, [open])
  return (
    <span ref={ref} style={{ position: 'relative', display: 'inline-flex' }}>
      <button type="button" className="btn btn-ghost btn-sm btn-icon" style={{ width: 20, height: 20 }} aria-label={label}
        aria-expanded={open} onClick={() => setOpen(o => !o)}>
        <CircleHelp size={14} aria-hidden />
      </button>
      {open && (
        <span role="tooltip" className="tooltip" style={{ position: 'absolute', top: 24, left: -8, width: 280, zIndex: 30, fontWeight: 400, lineHeight: 1.45 }}>
          {text}
        </span>
      )}
    </span>
  )
}

export function Field({ label, hint, children, error, htmlFor }) {
  const id = useId()
  const forId = htmlFor || id
  const child = typeof children === 'function' ? children(forId) : children
  return (
    <div className="field">
      {label && <label htmlFor={forId}>{label}</label>}
      {child}
      {hint && !error && <span className="hint">{hint}</span>}
      {error && <span className="hint" style={{ color: 'var(--bad)' }}>{error}</span>}
    </div>
  )
}

export function Select({ label, hint, value, onChange, options, placeholder, ...rest }) {
  return (
    <Field label={label} hint={hint}>
      {id => (
        <select id={id} className="select" value={value ?? ''} onChange={e => onChange(e.target.value || null)} {...rest}>
          {placeholder !== undefined && <option value="">{placeholder}</option>}
          {options.map(o => {
            const opt = typeof o === 'string' ? { value: o, label: o } : o
            return <option key={opt.value} value={opt.value} disabled={opt.disabled}>{opt.label}</option>
          })}
        </select>
      )}
    </Field>
  )
}

export function Input({ label, hint, error, ...rest }) {
  return <Field label={label} hint={hint} error={error}>{id => <input id={id} className="input" {...rest} />}</Field>
}

export function Check({ label, checked, onChange, ...rest }) {
  return (
    <label className="check">
      <input type="checkbox" checked={!!checked} onChange={e => onChange(e.target.checked)} {...rest} />
      {label}
    </label>
  )
}

export function Tabs({ tabs, value, onChange, label = 'Sections' }) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map(t => (
        <button key={t.id} type="button" role="tab" className="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          onKeyDown={e => {
            const i = tabs.findIndex(x => x.id === value)
            if (e.key === 'ArrowRight') onChange(tabs[(i + 1) % tabs.length].id)
            if (e.key === 'ArrowLeft') onChange(tabs[(i - 1 + tabs.length) % tabs.length].id)
          }}>
          {t.icon && <t.icon size={15} aria-hidden />}{t.label}{t.badge !== undefined && <Badge>{t.badge}</Badge>}
        </button>
      ))}
    </div>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(o => {
        const opt = typeof o === 'string' ? { value: o, label: o } : o
        return <button key={opt.value} type="button" aria-pressed={value === opt.value} onClick={() => onChange(opt.value)}>{opt.label}</button>
      })}
    </div>
  )
}

export function Modal({ title, onClose, children, footer, size }) {
  const ref = useRef(null)
  useEffect(() => {
    const prev = document.activeElement
    ref.current?.querySelector('button, input, select, textarea, [tabindex]')?.focus()
    const onKey = e => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'Tab') {
        const f = [...ref.current.querySelectorAll('button:not(:disabled), input, select, textarea, a[href]')]
        if (!f.length) return
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus() }
        else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus() }
      }
    }
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); prev?.focus?.() }
  }, [onClose])
  return (
    <div className="modal-backdrop" onMouseDown={e => e.target === e.currentTarget && onClose()}>
      <div ref={ref} className={`modal ${size === 'lg' ? 'modal-lg' : ''}`} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : 'Dialog'}>
        <div className="modal-head">
          <h2 style={{ fontSize: '1.05rem' }}>{title}</h2>
          <Button variant="ghost" icon={X} aria-label="Close" onClick={onClose} />
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>
  )
}

export function Confirm({ title, children, confirmLabel = 'Confirm', danger, busy, onConfirm, onCancel }) {
  return (
    <Modal title={title} onClose={onCancel} footer={<>
      <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      <Button variant={danger ? 'danger' : 'primary'} busy={busy} onClick={onConfirm}>{confirmLabel}</Button>
    </>}>
      {children}
    </Modal>
  )
}

export function Progress({ value, label }) {
  return (
    <div className="progress" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((value || 0) * 100)}>
      <span style={{ width: `${Math.max(2, (value || 0) * 100)}%` }} />
    </div>
  )
}

/** Simple table for evidence/records. columns: [{key, label, numeric, render}] */
export function DataTable({ columns, rows, maxHeight = 420, empty = 'No rows', caption, rowKey }) {
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table className="table">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>{columns.map(c => <th key={c.key} className={c.numeric ? 'n' : ''} scope="col">{c.label ?? c.key}</th>)}</tr>
        </thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={columns.length} className="muted" style={{ textAlign: 'center', padding: 20 }}>{empty}</td></tr>}
          {rows.map((r, i) => (
            <tr key={rowKey ? rowKey(r, i) : i}>
              {columns.map(c => {
                const v = c.render ? c.render(r[c.key], r) : r[c.key]
                const shown = c.render ? v : fmtCell(v)
                return (
                  <td key={c.key} className={c.numeric ? 'n' : ''}>
                    {shown === null || shown === undefined ? <span className="null">—</span> : <span className={c.clip === false ? '' : 'cell-clip'} title={typeof shown === 'string' ? shown : undefined}>{shown}</span>}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Tiny, safe markdown renderer (bold, italic, lists, headings). Never renders HTML. */
export function Markdown({ text }) {
  if (!text) return null
  const lines = String(text).split('\n')
  const blocks = []
  let list = null
  const inline = (s, k) => {
    const parts = s.split(/(\*\*[^*]+\*\*|_[^_]+_|\*[^*]+\*)/g)
    return parts.map((p, i) => {
      if (/^\*\*[^*]+\*\*$/.test(p)) return <strong key={`${k}-${i}`}>{p.slice(2, -2)}</strong>
      if (/^(_[^_]+_|\*[^*]+\*)$/.test(p)) return <em key={`${k}-${i}`}>{p.slice(1, -1)}</em>
      return p
    })
  }
  lines.forEach((ln, i) => {
    const ul = /^\s*[-•]\s+(.*)/.exec(ln)
    const ol = /^\s*\d+\.\s+(.*)/.exec(ln)
    if (ul || ol) {
      const type = ul ? 'ul' : 'ol'
      if (!list || list.type !== type) { list = { type, items: [] }; blocks.push(list) }
      list.items.push((ul || ol)[1])
      return
    }
    list = null
    if (!ln.trim()) return
    const h = /^#{1,4}\s+(.*)/.exec(ln)
    blocks.push(h ? { type: 'h', text: h[1], i } : { type: 'p', text: ln, i })
  })
  return (
    <div className="md">
      {blocks.map((b, i) => b.type === 'ul' ? <ul key={i}>{b.items.map((t, j) => <li key={j}>{inline(t, `${i}-${j}`)}</li>)}</ul>
        : b.type === 'ol' ? <ol key={i}>{b.items.map((t, j) => <li key={j}>{inline(t, `${i}-${j}`)}</li>)}</ol>
          : b.type === 'h' ? <h3 key={i}>{inline(b.text, i)}</h3> : <p key={i}>{inline(b.text, i)}</p>)}
    </div>
  )
}

export function Toasts({ toasts, onDismiss }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.kind}`} role={t.kind === 'error' ? 'alert' : 'status'}>
          <div className="grow">{t.message}</div>
          <Button variant="ghost" size="sm" icon={X} aria-label="Dismiss" onClick={() => onDismiss(t.id)} />
        </div>
      ))}
    </div>
  )
}
