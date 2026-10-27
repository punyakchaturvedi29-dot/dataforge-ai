import { del, download, get, patch, post, upload } from './client'

const w = id => `/workspaces/${id}`

export const api = {
  health: () => get('/health'),
  config: () => get('/config'),

  // workspaces & data
  listWorkspaces: () => get('/workspaces'),
  inspectFile: file => upload('/workspaces/inspect', file),
  uploadFile: (file, fields, onProgress) => upload('/workspaces/upload', file, fields, onProgress),
  loadSample: () => post('/workspaces/sample'),
  workspace: id => get(w(id)),
  patchWorkspace: (id, body) => patch(w(id), body),
  deleteWorkspace: id => del(w(id)),
  markStage: (id, stage) => post(`${w(id)}/stages/${stage}`),
  overview: id => get(`${w(id)}/overview`),
  data: (id, params) => get(`${w(id)}/data`, params),
  schema: id => get(`${w(id)}/schema`),
  activateDataset: (id, did) => post(`${w(id)}/datasets/${did}/activate`),
  deleteDataset: (id, did) => del(`${w(id)}/datasets/${did}`),
  downloadData: (id, params) => download(`${w(id)}/download`, { params }),

  // profiling, insights, charts
  profile: id => get(`${w(id)}/profile`),
  profileCsv: id => download(`${w(id)}/profile/export`),
  correlations: id => get(`${w(id)}/correlations`),
  targetAnalysis: (id, target) => get(`${w(id)}/target-analysis`, { target }),
  insights: (id, target) => get(`${w(id)}/insights`, target ? { target } : {}),
  insightsCsv: id => download(`${w(id)}/insights/export`),
  recommendedCharts: id => get(`${w(id)}/charts/recommended`),
  chart: (id, spec) => post(`${w(id)}/charts`, spec),
  dashboard: id => get(`${w(id)}/dashboard`),

  // cleaning & rules
  suggestions: id => get(`${w(id)}/cleaning/suggestions`),
  operations: id => get(`${w(id)}/cleaning/operations`),
  previewOp: (id, body) => post(`${w(id)}/cleaning/preview`, body),
  applyOp: (id, body) => post(`${w(id)}/cleaning/apply`, body),
  applySafe: (id, body = {}) => post(`${w(id)}/cleaning/apply-safe`, body),
  undo: id => post(`${w(id)}/cleaning/undo`),
  revert: (id, body) => post(`${w(id)}/cleaning/revert`, body),
  history: id => get(`${w(id)}/cleaning/history`),
  rules: id => get(`${w(id)}/rules`),
  addRule: (id, rule) => post(`${w(id)}/rules`, { rule }),
  deleteRule: (id, rid) => del(`${w(id)}/rules/${rid}`),
  evaluateRules: id => post(`${w(id)}/rules/evaluate`),

  // ML
  mlOptions: id => get(`${w(id)}/ml/options`),
  train: (id, body) => post(`${w(id)}/ml/train`, body),
  job: jid => get(`/jobs/${jid}`),
  cancelJob: jid => post(`/jobs/${jid}/cancel`),
  jobs: id => get(`${w(id)}/jobs`),
  experiments: id => get(`${w(id)}/experiments`),
  experiment: (id, eid) => get(`${w(id)}/experiments/${eid}`),
  experimentCsv: (id, eid) => download(`${w(id)}/experiments/${eid}/export`),
  deleteExperiment: (id, eid) => del(`${w(id)}/experiments/${eid}`),
  models: id => get(`${w(id)}/models`),
  model: (id, mid) => get(`${w(id)}/models/${mid}`),
  predict: (id, mid, body) => post(`${w(id)}/models/${mid}/predict`, body),
  predictBatch: (id, mid, file) => upload(`${w(id)}/models/${mid}/predict/batch`, file),
  downloadPredictions: (id, pid) => download(`${w(id)}/predictions/${pid}/download`),
  cluster: (id, body) => post(`${w(id)}/ml/cluster`, body),
  anomaly: (id, body) => post(`${w(id)}/ml/anomaly`, body),
  latest: (id, kind) => get(`${w(id)}/analyses/${kind}`),

  // forecasting & diagnostics
  forecastOptions: id => get(`${w(id)}/forecast/options`),
  forecast: (id, body) => post(`${w(id)}/forecast`, body),
  diagnostics: (id, body) => post(`${w(id)}/diagnostics`, body),

  // multi-file
  addSource: (id, file, fields) => upload(`${w(id)}/sources`, file, fields),
  compareSources: (id, ids) => get(`${w(id)}/sources/compare`, ids ? { ids } : {}),
  joinCandidates: (id, left, right) => get(`${w(id)}/sources/join-candidates`, { left, right }),
  joinPreview: (id, body) => post(`${w(id)}/sources/join/preview`, body),
  join: (id, body) => post(`${w(id)}/sources/join`, body),
  stack: (id, body) => post(`${w(id)}/sources/stack`, body),

  // copilot
  ask: (id, message) => post(`${w(id)}/copilot`, { message }),
  chatHistory: id => get(`${w(id)}/copilot/history`),
  clearChat: id => del(`${w(id)}/copilot/history`),

  // reports
  summary: id => get(`${w(id)}/summary`),
  recommendations: id => get(`${w(id)}/recommendations`),
  reportPreview: (id, body) => post(`${w(id)}/report/preview`, body),
  reportDownload: (id, body) => download(`${w(id)}/report`, { method: 'post', body }),
}
