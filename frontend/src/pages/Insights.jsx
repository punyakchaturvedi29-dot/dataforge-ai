import { useState } from 'react'
import { Download, Sparkles } from 'lucide-react'
import { api } from '../api/endpoints'
import { useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { FindingCard } from '../components/Findings'
import { ChartCard } from '../components/charts/ChartView'
import { Button, Callout, Empty, ErrorState, LoadingBlock, Modal, Segmented } from '../components/ui'

const CATS = [
  { value: 'all', label: 'All' }, { value: 'segment', label: 'Segments' }, { value: 'relationship', label: 'Relationships' },
  { value: 'trend', label: 'Trends' }, { value: 'distribution', label: 'Distributions' }, { value: 'anomaly', label: 'Outliers' },
  { value: 'data_quality', label: 'Data quality' },
]

export default function Insights() {
  const { id, dataKey, ws } = useWorkspace()
  const { notify } = useUi()
  const [cat, setCat] = useState('all')
  const [chart, setChart] = useState(null)
  const { data, error, loading, reload } = useApi(() => api.insights(id), [id, dataKey])
  const showChart = async spec => {
    try { setChart(await api.chart(id, spec)) } catch (e) { notify(e.message, { kind: 'error' }) }
  }
  const list = (data?.findings || []).filter(f => cat === 'all' || f.category === cat || (cat === 'distribution' && f.category === 'concentration'))
  return (
    <div className="page">
      <PageHeader title="Insights" description="Statistically supported findings, ranked by evidence strength and practical impact. Each states what was observed, the evidence, and how to follow up."
        actions={<Button icon={Download} onClick={() => api.insightsCsv(id).catch(e => notify(e.message, { kind: 'error' }))}>Findings CSV</Button>} />
      {data?.target && <div style={{ marginBottom: 12 }}><Callout tone="info">Findings involving the target “{data.target}” are ranked higher. Change the target in Profile → Target analysis.</Callout></div>}
      <div className="row" style={{ marginBottom: 14 }}><Segmented label="Filter by kind" value={cat} onChange={setCat} options={CATS} /></div>
      {error && <ErrorState error={error} onRetry={reload} />}
      {!data && loading && <LoadingBlock rows={4} h={110} />}
      {data && list.length === 0 && <Empty icon={Sparkles} title="Nothing to report here">No finding of this kind passed the statistical checks (multiple-testing corrected). That is a result too: it means no strong pattern is supported by this data.</Empty>}
      <div className="stack-sm">{list.map((f, i) => <FindingCard key={f.id} f={f} rank={cat === 'all' ? i + 1 : undefined} onChart={showChart} />)}</div>
      {data && <p className="small muted" style={{ marginTop: 16 }}>{data.notes.join(' ')}</p>}
      {chart && <Modal size="lg" title={chart.title} onClose={() => setChart(null)}><ChartCard chart={chart} title="" height={300} /></Modal>}
    </div>
  )
}
