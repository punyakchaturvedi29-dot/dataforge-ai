import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Bot, Calculator, Eraser, Send, Sparkles } from 'lucide-react'
import { api } from '../api/endpoints'
import { useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { ChartBody } from '../components/charts/ChartView'
import { Badge, Button, DataTable, ErrorState, Markdown } from '../components/ui'
import { fmtNum } from '../lib/format'

const STARTERS = [
  'What are the most important patterns in this dataset?',
  'Which columns have the greatest number of missing values?',
  'Identify unusual observations',
  'Generate a management summary',
  'Recommend the next analysis to perform',
]

export default function Copilot() {
  const { id, ws, dataKey } = useWorkspace()
  const { notify } = useUi()
  const { data: hist, error } = useApi(() => api.chatHistory(id), [id])
  const [msgs, setMsgs] = useState([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef(null)
  const input = useRef(null)
  useEffect(() => { if (hist) setMsgs(hist.messages.map(m => ({ role: m.role, content: m.content, payload: m.payload }))) }, [hist])
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [msgs, busy])
  const send = async q => {
    const message = (q ?? text).trim()
    if (!message || busy) return
    setText('')
    setMsgs(m => [...m, { role: 'user', content: message }])
    setBusy(true)
    try {
      const r = await api.ask(id, message)
      setMsgs(m => [...m, { role: 'assistant', content: r.answer, payload: r }])
    } catch (e) {
      setMsgs(m => [...m, { role: 'assistant', content: `I couldn't answer that: ${e.message}`, payload: { computed: false, error: true } }])
    } finally {
      setBusy(false)
      input.current?.focus()
    }
  }
  const clear = async () => { await api.clearChat(id); setMsgs([]); notify('Conversation cleared') }
  return (
    <div className="page" style={{ paddingBottom: 16 }}>
      <PageHeader rail={false} title="Copilot" description="Ask about the active dataset in plain language. Numbers come from computed analyses and show their evidence; the copilot never runs code."
        actions={msgs.length > 0 && <Button variant="ghost" icon={Eraser} onClick={clear}>Clear conversation</Button>} />
      {error && <ErrorState error={error} />}
      <div className="chat">
        <div className="chat-log" role="log" aria-live="polite" aria-label="Conversation">
          {msgs.length === 0 && (
            <div className="empty" style={{ marginTop: 20 }}>
              <div className="empty-icon"><Bot size={22} /></div>
              <h3>Ask about “{ws.active_dataset.name}”</h3>
              <p>Try a question below, or ask to compare a measure across a category, forecast a metric, or build a model.</p>
            </div>
          )}
          {msgs.map((m, i) => m.role === 'user'
            ? <div key={i} className="msg-user">{m.content}</div>
            : <AssistantMessage key={i} m={m} onAsk={send} dataKey={dataKey} />)}
          {busy && <div className="msg-ai"><div className="bubble muted small" aria-label="Thinking">Computing…</div></div>}
          <div ref={end} />
        </div>
        {msgs.length < 2 && (
          <div className="row" style={{ marginBottom: 10 }}>
            {STARTERS.map(s => <button key={s} type="button" className="pill-btn" onClick={() => send(s)}>{s}</button>)}
          </div>
        )}
        <form className="row" onSubmit={e => { e.preventDefault(); send() }} style={{ alignItems: 'stretch' }}>
          <textarea ref={input} className="textarea grow" rows={2} maxLength={2000} placeholder="e.g. Compare revenue across customer segments" aria-label="Question"
            value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} />
          <Button type="submit" variant="primary" icon={Send} busy={busy} disabled={!text.trim()} style={{ height: 'auto' }}>Ask</Button>
        </form>
      </div>
    </div>
  )
}

function AssistantMessage({ m, onAsk }) {
  const p = m.payload || {}
  const navigate = useNavigate()
  const { id } = useWorkspace()
  const { notify } = useUi()
  const act = async a => {
    if (a.kind === 'navigate') {
      const q = new URLSearchParams()
      if (a.payload.tab) q.set('tab', a.payload.tab)
      if (a.payload.target) q.set('target', a.payload.target)
      navigate(`/w/${id}/${a.payload.page}${q.toString() ? '?' + q : ''}`)
    }
    if (a.kind === 'train') {
      try {
        await api.train(id, { target: a.payload.target })
        notify(`Training started for “${a.payload.target}”. Follow progress on the Model page.`, { kind: 'success' })
        navigate(`/w/${id}/model?target=${encodeURIComponent(a.payload.target)}`)
      } catch (e) { notify(e.message, { kind: 'error' }) }
    }
  }
  return (
    <div className="msg-ai">
      <div className="bubble">
        <Markdown text={m.content} />
        {p.llm_explanation && (
          <div className="panel-flat small" style={{ marginTop: 10 }}>
            <div className="row tiny muted" style={{ marginBottom: 4 }}><Sparkles size={12} /> AI explanation — its numbers were checked against the computed results</div>
            {p.llm_explanation}
          </div>
        )}
        {p.clarification && (
          <div className="row" style={{ marginTop: 10 }}>
            {p.clarification.options.map(o => <button key={o.label} type="button" className="pill-btn" onClick={() => onAsk(o.message)}>{o.label}</button>)}
          </div>
        )}
        {p.actions?.length > 0 && (
          <div className="row" style={{ marginTop: 10 }}>
            {p.actions.map((a, i) => <Button key={i} size="sm" variant={a.kind === 'train' ? 'primary' : 'secondary'} onClick={() => act(a)}>{a.label}</Button>)}
          </div>
        )}
      </div>
      {(p.evidence || []).map((ev, i) => <Evidence key={i} ev={ev} />)}
      <div className="row tiny muted">
        {p.computed ? <Badge tone="good"><Calculator size={11} />computed from data</Badge> : p.tool && !p.error ? <Badge>guidance</Badge> : null}
        {p.tool && p.tool !== 'clarify' && p.tool !== 'help' && <span>tool: {p.tool.replace(/_/g, ' ')}</span>}
        {p.routed_by === 'llm' && <span>question routed by AI</span>}
      </div>
      {p.suggestions?.length > 0 && <div className="row">{p.suggestions.map(s => <button key={s} type="button" className="pill-btn" onClick={() => onAsk(s)}>{s}</button>)}</div>}
    </div>
  )
}

function Evidence({ ev }) {
  const { id } = useWorkspace()
  const [chart, setChart] = useState(null)
  useEffect(() => { if (ev.type === 'chart') api.chart(id, ev.chart).then(setChart).catch(() => setChart(null)) }, [ev, id])
  if (ev.type === 'table') {
    return (
      <div>
        {ev.title && <div className="tiny muted" style={{ marginBottom: 4 }}>Evidence: {ev.title}</div>}
        <DataTable columns={ev.columns.map(c => ({ key: c, label: c.replace(/_/g, ' '), numeric: typeof ev.rows[0]?.[c] === 'number' }))} rows={ev.rows} maxHeight={260} />
      </div>
    )
  }
  if (ev.type === 'metrics') {
    return (
      <div className="row">
        {ev.items.map(it => <span key={it.label} className="panel-flat small num" style={{ padding: '6px 10px' }}><span className="muted">{it.label}: </span>{typeof it.value === 'number' ? fmtNum(it.value) : String(it.value ?? '—')}</span>)}
      </div>
    )
  }
  if (ev.type === 'chart' && chart) return <div className="chart-card"><div className="chart-title">{chart.title}</div><ChartBody chart={chart} height={Math.min(260, 60 + (chart.data?.length || 6) * 26)} /></div>
  return null
}
