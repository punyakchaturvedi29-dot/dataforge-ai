import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api } from '../api/endpoints'
import { useUi } from './ui'

const WsCtx = createContext(null)

/** Current workspace (from the URL) shared by all pages. `version` bumps whenever data changes. */
export function WorkspaceProvider({ id, children }) {
  const [ws, setWs] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [version, setVersion] = useState(0)
  const { rememberWorkspace } = useUi()

  const refresh = useCallback(async () => {
    try {
      const data = await api.workspace(id)
      setWs(data)
      setError(null)
      rememberWorkspace(data)
      return data
    } catch (e) {
      setError(e)
    } finally {
      setLoading(false)
    }
  }, [id, rememberWorkspace])

  useEffect(() => { setLoading(true); setWs(null); refresh() }, [refresh])

  /** Call after anything that changes the active data (cleaning, joins, activation). */
  const dataChanged = useCallback(async () => {
    await refresh()
    setVersion(v => v + 1)
  }, [refresh])

  const value = useMemo(() => ({
    id, ws, error, loading, refresh, dataChanged, version,
    dataset: ws?.active_dataset, stages: ws?.stages || [],
    dataKey: `${ws?.active_dataset?.id}:${ws?.active_dataset?.version}:${version}`,
  }), [id, ws, error, loading, refresh, dataChanged, version])
  return <WsCtx.Provider value={value}>{children}</WsCtx.Provider>
}

export const useWorkspace = () => useContext(WsCtx)
