import { useEffect, useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Download, FileWarning, Search } from 'lucide-react'
import { api } from '../api/endpoints'
import { useApi } from '../hooks/useApi'
import { useUi } from '../state/ui'
import { useWorkspace } from '../state/workspace'
import { PageHeader } from '../components/layout/Shell'
import { Badge, Button, Callout, Check, DataTable, ErrorState, LoadingBlock, Panel, Segmented, Select, Tabs } from '../components/ui'
import { bytes, fmtCell, fmtNum, plural } from '../lib/format'

export default function DataPage() {
  const { ws } = useWorkspace()
  const [tab, setTab] = useState('rows')
  const ds = ws.active_dataset
  return (
    <div className="page">
      <PageHeader title="Data" description="Every row, paged on the server. The original upload is kept unchanged; cleaning works on a separate copy." actions={<DownloadMenu />} />
      <DatasetSwitcher />
      <Tabs value={tab} onChange={setTab} tabs={[
        { id: 'rows', label: 'Rows' }, { id: 'schema', label: 'Schema', badge: ds.columns },
        { id: 'ingestion', label: 'Validation report', badge: ds.ingestion?.warnings?.length || undefined },
      ]} />
      {tab === 'rows' && <RowsView />}
      {tab === 'schema' && <SchemaView />}
      {tab === 'ingestion' && <IngestionReport ing={ds.ingestion} ds={ds} />}
    </div>
  )
}

function DatasetSwitcher() {
  const { ws, id, dataChanged } = useWorkspace()
  const { notify } = useUi()
  if ((ws.datasets || []).length < 2) return null
  return (
    <div className="row" style={{ marginBottom: 14 }}>
      <Select label="Active dataset" value={ws.active_dataset.id} onChange={async v => {
        try { await api.activateDataset(id, v); await dataChanged(); notify('Active dataset changed', { kind: 'success' }) } catch (e) { notify(e.message, { kind: 'error' }) }
      }} options={ws.datasets.map(d => ({ value: d.id, label: `${d.name} (${fmtNum(d.rows)} rows${d.source !== 'upload' ? `, ${d.source}` : ''})` }))} />
    </div>
  )
}

function DownloadMenu() {
  const { id, ws } = useWorkspace()
  const { notify } = useUi()
  const dl = async (version, format) => {
    try { const name = await api.downloadData(id, { version, format }); notify(`Downloaded ${name}`, { kind: 'success' }) } catch (e) { notify(e.message, { kind: 'error' }) }
  }
  return (
    <div className="row">
      <Button icon={Download} onClick={() => dl('working', 'csv')}>{ws.active_dataset.n_ops ? 'Cleaned CSV' : 'CSV'}</Button>
      <Button variant="ghost" onClick={() => dl('working', 'xlsx')}>Excel</Button>
      {ws.active_dataset.n_ops > 0 && <Button variant="ghost" onClick={() => dl('original', 'csv')}>Original CSV</Button>}
    </div>
  )
}

function RowsView() {
  const { id, dataKey, ws } = useWorkspace()
  const [q, setQ] = useState({ page: 1, page_size: 50, sort: null, desc: false, search: '', version: 'working', missing_only: false, duplicates_only: false })
  const [searchText, setSearchText] = useState('')
  useEffect(() => { const t = setTimeout(() => setQ(s => ({ ...s, search: searchText, page: 1 })), 300); return () => clearTimeout(t) }, [searchText])
  const params = useMemo(() => Object.fromEntries(Object.entries(q).filter(([, v]) => v !== null && v !== '' && v !== false)), [q])
  const { data, error, loading, reload } = useApi(() => api.data(id, params), [id, dataKey, JSON.stringify(params)])
  const setSort = col => setQ(s => ({ ...s, sort: col, desc: s.sort === col ? !s.desc : false, page: 1 }))
  return (
    <div className="stack">
      <div className="row">
        <div style={{ position: 'relative', flex: '1 1 260px', maxWidth: 360 }}>
          <Search size={15} aria-hidden style={{ position: 'absolute', left: 10, top: 10, color: 'var(--ink-3)' }} />
          <input className="input" style={{ paddingLeft: 32 }} placeholder="Search text columns" aria-label="Search rows" value={searchText} onChange={e => setSearchText(e.target.value)} />
        </div>
        {ws.active_dataset.n_ops > 0 && <Segmented label="Version" value={q.version} onChange={v => setQ(s => ({ ...s, version: v, page: 1 }))}
          options={[{ value: 'working', label: 'Cleaned' }, { value: 'original', label: 'Original' }]} />}
        <Check label="Rows with missing values" checked={q.missing_only} onChange={v => setQ(s => ({ ...s, missing_only: v, page: 1 }))} />
        <Check label="Duplicate rows" checked={q.duplicates_only} onChange={v => setQ(s => ({ ...s, duplicates_only: v, page: 1 }))} />
        <span className="grow" />
        {data && <span className="small muted num">{fmtNum(data.matched_rows)} of {fmtNum(data.total_rows)} rows</span>}
      </div>
      {error && <ErrorState error={error} onRetry={reload} />}
      {!data && loading && <LoadingBlock rows={1} h={360} />}
      {data && (
        <div className="table-wrap" style={{ maxHeight: 'calc(100vh - 330px)', minHeight: 240, opacity: loading ? 0.6 : 1 }}>
          <table className="table" aria-label="Dataset rows" aria-busy={loading}>
            <thead>
              <tr>
                <th className="n" scope="col" title="Row number in the original upload">#</th>
                {data.columns.map(c => (
                  <th key={c.name} scope="col" aria-sort={q.sort === c.name ? (q.desc ? 'descending' : 'ascending') : 'none'}>
                    <button type="button" className="th-sort" onClick={() => setSort(c.name)} title={`${c.name} (${c.dtype}) — sort`}>
                      {c.name}{q.sort === c.name ? (q.desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />) : <ArrowUpDown size={12} style={{ opacity: 0.35 }} />}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.length === 0 && <tr><td colSpan={data.columns.length + 1} className="muted" style={{ textAlign: 'center', padding: 28 }}>No rows match these filters.</td></tr>}
              {data.rows.map((r, i) => (
                <tr key={data.row_ids[i]}>
                  <td className="n muted">{Number(data.row_ids[i]) + 1}</td>
                  {data.columns.map(c => {
                    const v = r[c.name]
                    const shown = fmtCell(v, { raw: true })
                    return <td key={c.name} className={typeof v === 'number' ? 'n' : ''}>{shown === null ? <span className="null">missing</span> : <span className="cell-clip" title={String(shown)}>{shown}</span>}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && (
        <div className="row-between">
          <Select label="Rows per page" value={String(q.page_size)} onChange={v => setQ(s => ({ ...s, page_size: Number(v), page: 1 }))} options={['25', '50', '100', '250']} />
          <div className="row" role="navigation" aria-label="Pagination">
            <Button size="sm" icon={ChevronLeft} aria-label="Previous page" disabled={q.page <= 1} onClick={() => setQ(s => ({ ...s, page: s.page - 1 }))} />
            <span className="small num">Page {data.page} of {data.pages}</span>
            <Button size="sm" icon={ChevronRight} aria-label="Next page" disabled={q.page >= data.pages} onClick={() => setQ(s => ({ ...s, page: s.page + 1 }))} />
          </div>
        </div>
      )}
    </div>
  )
}

const TYPE_TONE = { numeric: 'info', categorical: 'accent', datetime: 'good', boolean: 'accent', identifier: undefined, text: undefined, constant: 'warn', empty: 'bad' }

function SchemaView() {
  const { id, dataKey } = useWorkspace()
  const { data, error, loading, reload } = useApi(() => api.schema(id), [id, dataKey])
  if (error) return <ErrorState error={error} onRetry={reload} />
  if (!data) return <LoadingBlock rows={1} h={300} />
  return (
    <Panel title="Column types" subtitle="Logical types are inferred from the values; they decide how each column is analysed and modelled." pad={false}>
      <div style={{ padding: '12px 20px 20px' }}>
        <DataTable rowKey={r => r.name} columns={[
          { key: 'name', label: 'Column' },
          { key: 'logical_type', label: 'Analysed as', render: v => <Badge tone={TYPE_TONE[v]}>{v}</Badge> },
          { key: 'semantic_type', label: 'Meaning (from name)' },
          { key: 'dtype', label: 'Storage type' },
          { key: 'missing_pct', label: 'Missing %', numeric: true, render: v => v.toFixed(1) },
          { key: 'n_unique', label: 'Distinct', numeric: true, render: v => fmtNum(v) },
          { key: 'hints', label: 'Notes', render: v => v?.length ? v.map(h => <Badge key={h} tone="warn">{h.replace(/_/g, ' ')}</Badge>) : '' },
        ]} rows={data.columns} maxHeight={600} />
      </div>
    </Panel>
  )
}

export function IngestionReport({ ing, ds }) {
  if (!ing) return null
  const mal = ing.malformed_rows || {}
  return (
    <div className="stack">
      {ing.warnings?.length > 0
        ? <Callout tone="warn" title="Warnings"><ul style={{ margin: 0, paddingLeft: 18 }}>{ing.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Callout>
        : <Callout tone="good">The file was read without warnings.</Callout>}
      <div className="grid-2">
        <Panel title="File">
          <dl className="kv">
            <dt>Name</dt><dd>{ing.filename}</dd>
            <dt>Format</dt><dd>{String(ing.format || '').toUpperCase()}</dd>
            {ing.size_bytes != null && <><dt>Size</dt><dd>{bytes(ing.size_bytes)}</dd></>}
            {ing.encoding && <><dt>Encoding</dt><dd>{ing.encoding}</dd></>}
            {ing.delimiter && <><dt>Delimiter</dt><dd>{ing.delimiter}</dd></>}
            {ing.sheet && <><dt>Sheet</dt><dd>{ing.sheet}{ing.sheets?.length > 1 ? ` (of ${ing.sheets.length})` : ''}</dd></>}
            <dt>Rows read</dt><dd>{fmtNum(ds.original_rows)}</dd>
            <dt>Columns</dt><dd>{ds.original_columns}</dd>
            {ing.sha256 && <><dt>SHA-256</dt><dd className="truncate" title={ing.sha256}>{ing.sha256.slice(0, 24)}…</dd></>}
          </dl>
        </Panel>
        <Panel title="Changes made while reading" subtitle="Only lossless conversions happen automatically.">
          <div className="stack-sm small">
            {(ing.type_conversions || []).map(t => <div key={t.column}><Badge tone="good">{t.from} → {t.to}</Badge> <strong>{t.column}</strong> <span className="muted">{t.note}</span></div>)}
            {(ing.column_changes || []).map((c, i) => <div key={i}><Badge tone="warn">renamed</Badge> column {c.position}: “{c.original || '(blank)'}” → <strong>{c.new}</strong> <span className="muted">({c.reason})</span></div>)}
            {!(ing.type_conversions || []).length && !(ing.column_changes || []).length && <span className="muted">None.</span>}
          </div>
        </Panel>
      </div>
      {mal.count > 0 && (
        <Panel title={<span className="row"><FileWarning size={16} /> {plural(mal.count, 'malformed row')} skipped</span>}
          subtitle={`These rows had more fields than the header (${mal.expected_fields ?? '?'}). Fix them in the source file if they matter.`}>
          <DataTable columns={[{ key: 'line', label: 'Line', numeric: true }, { key: 'fields', label: 'Fields', numeric: true }, { key: 'values', label: 'Values', render: v => v.join(' | ') }]}
            rows={(mal.samples || []).map(s => (Array.isArray(s) ? { line: '—', fields: s.length, values: s } : s))} />
        </Panel>
      )}
    </div>
  )
}
