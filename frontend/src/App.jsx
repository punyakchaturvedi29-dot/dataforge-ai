import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useParams } from 'react-router-dom'
import { UiProvider, useUi } from './state/ui'
import { WorkspaceProvider, useWorkspace } from './state/workspace'
import { Shell } from './components/layout/Shell'
import { Button, Empty, ErrorState, LoadingBlock, Toasts } from './components/ui'
import Home from './pages/Home'
import { SearchX } from 'lucide-react'

const Overview = lazy(() => import('./pages/Overview'))
const DataPage = lazy(() => import('./pages/DataPage'))
const Profile = lazy(() => import('./pages/Profile'))
const Clean = lazy(() => import('./pages/Clean'))
const Sources = lazy(() => import('./pages/Sources'))
const Insights = lazy(() => import('./pages/Insights'))
const Visualize = lazy(() => import('./pages/Visualize'))
const Diagnose = lazy(() => import('./pages/Diagnose'))
const Forecast = lazy(() => import('./pages/Forecast'))
const Model = lazy(() => import('./pages/Model'))
const Predict = lazy(() => import('./pages/Predict'))
const Copilot = lazy(() => import('./pages/Copilot'))
const Report = lazy(() => import('./pages/Report'))

function WorkspaceGate({ children }) {
  const { ws, error, loading, refresh } = useWorkspace()
  if (loading && !ws) return <div className="page"><LoadingBlock rows={3} /></div>
  if (error && !ws) {
    return (
      <div className="page">
        {error.status === 404
          ? <Empty icon={SearchX} title="Workspace not found" action={<Button variant="primary" onClick={() => (window.location.href = '/')}>Go to workspaces</Button>}>
              It may have been deleted or expired after the retention period.
            </Empty>
          : <ErrorState error={error} onRetry={refresh} />}
      </div>
    )
  }
  return children
}

function WorkspaceRoutes() {
  const { id } = useParams()
  return (
    <WorkspaceProvider id={id}>
      <Shell>
        <WorkspaceGate>
          <Suspense fallback={<div className="page"><LoadingBlock rows={2} /></div>}>
            <Routes>
              <Route index element={<Overview />} />
              <Route path="data" element={<DataPage />} />
              <Route path="profile" element={<Profile />} />
              <Route path="clean" element={<Clean />} />
              <Route path="sources" element={<Sources />} />
              <Route path="insights" element={<Insights />} />
              <Route path="visualize" element={<Visualize />} />
              <Route path="diagnose" element={<Diagnose />} />
              <Route path="forecast" element={<Forecast />} />
              <Route path="model" element={<Model />} />
              <Route path="predict" element={<Predict />} />
              <Route path="copilot" element={<Copilot />} />
              <Route path="report" element={<Report />} />
              <Route path="*" element={<Navigate to="" replace />} />
            </Routes>
          </Suspense>
        </WorkspaceGate>
      </Shell>
    </WorkspaceProvider>
  )
}

function ToastHost() {
  const { toasts, dismissToast } = useUi()
  return <Toasts toasts={toasts} onDismiss={dismissToast} />
}

export default function App() {
  return (
    <UiProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/w/:id/*" element={<WorkspaceRoutes />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
      <ToastHost />
    </UiProvider>
  )
}
