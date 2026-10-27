import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Badge } from './ui'

const CLAIM = {
  observation: { label: 'Observation', cls: 'claim-observation', help: 'A computed fact about the data.' },
  association: { label: 'Association', cls: 'claim-association', help: 'A statistical relationship. It does not show cause and effect.' },
  hypothesis: { label: 'Hypothesis', cls: 'claim-hypothesis', help: 'A plausible explanation that needs validation.' },
  'potential driver': { label: 'Potential driver', cls: 'claim-driver', help: 'An association that survived the available checks. Still not proof of causation.' },
  'diagnostic hypothesis': { label: 'Diagnostic hypothesis', cls: 'claim-diagnostic', help: 'A plausible explanation; the checks could not rule out alternatives.' },
}
const CONF = { high: 'good', medium: 'warn', low: undefined }

export function ClaimBadge({ type }) {
  const c = CLAIM[type] || { label: type, cls: '' }
  return <span className={`badge ${c.cls}`} title={c.help}>{c.label}</span>
}

export function FindingCard({ f, defaultOpen = false, onChart, rank }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <article className="finding" aria-labelledby={`f-${f.id}`}>
      <div className="finding-head">
        <div className="grow">
          <h3 id={`f-${f.id}`} style={{ fontSize: '0.98rem' }}>{rank ? `${rank}. ` : ''}{f.title}</h3>
          <p className="secondary" style={{ marginTop: 4 }}>{f.observation}</p>
        </div>
        <div className="row" style={{ flexShrink: 0 }}>
          <ClaimBadge type={f.claim_type} />
          {f.confidence && <Badge tone={CONF[f.confidence]} title="Confidence from sample size and statistical evidence">{f.confidence} confidence</Badge>}
        </div>
      </div>
      <p className="small"><span className="muted">Evidence: </span><span className="num">{f.evidence_text}</span></p>
      <div className="row">
        <button type="button" className="btn btn-ghost btn-sm" aria-expanded={open} onClick={() => setOpen(o => !o)}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}{open ? 'Hide details' : 'Why it matters & next steps'}
        </button>
        {f.chart && onChart && <button type="button" className="btn btn-ghost btn-sm" onClick={() => onChart(f.chart)}>Show chart</button>}
        <span className="tiny muted grow" style={{ textAlign: 'right' }}>priority score {f.score}</span>
      </div>
      {open && (
        <dl>
          <dt>Why it may matter</dt><dd>{f.why_it_matters}</dd>
          <dt>Possible interpretations</dt><dd><ul style={{ margin: 0, paddingLeft: 18 }}>{f.interpretations.map((x, i) => <li key={i}>{x}</li>)}</ul></dd>
          <dt>Follow-up analysis</dt><dd><ul style={{ margin: 0, paddingLeft: 18 }}>{f.follow_up.map((x, i) => <li key={i}>{x}</li>)}</ul></dd>
        </dl>
      )}
    </article>
  )
}

/** Diagnostics finding (RCA) */
export function DiagnosticCard({ f }) {
  return (
    <article className="finding">
      <div className="finding-head">
        <h3 className="grow" style={{ fontSize: '0.98rem' }}>{f.title}</h3>
        <ClaimBadge type={f.label} />
      </div>
      <dl>
        <dt>Metric</dt><dd>{f.metric}</dd>
        <dt>Comparison</dt><dd>{f.comparison}</dd>
        <dt>Evidence</dt><dd className="num">{f.evidence}</dd>
        {f.uncertainty && Array.isArray(f.uncertainty) && <><dt>95% interval</dt><dd className="num">{f.uncertainty.map(v => (v === null ? '—' : Number(v).toPrecision(3))).join(' to ')}</dd></>}
        {f.confounders?.length > 0 && <><dt>Potential confounders</dt><dd>{f.confounders.join(', ')}</dd></>}
        <dt>Alternative explanations</dt><dd><ul style={{ margin: 0, paddingLeft: 18 }}>{f.alternatives.map((x, i) => <li key={i}>{x}</li>)}</ul></dd>
        <dt>How to validate</dt><dd><ul style={{ margin: 0, paddingLeft: 18 }}>{f.validation.map((x, i) => <li key={i}>{x}</li>)}</ul></dd>
      </dl>
    </article>
  )
}
