import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const UiCtx = createContext(null)

function read(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v) } catch { return fallback }
}
function write(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* private mode */ }
}

export function UiProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    try { return localStorage.getItem('df.theme') || 'system' } catch { return 'system' }
  })
  const [collapsed, setCollapsed] = useState(() => read('df.sidebar', false))
  const [toasts, setToasts] = useState([])
  const [notifications, setNotifications] = useState(() => read('df.notifications', []))
  const [recent, setRecent] = useState(() => read('df.recent', []))

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'light' || theme === 'dark') root.dataset.theme = theme
    else delete root.dataset.theme
    try { localStorage.setItem('df.theme', theme) } catch { /* ignore */ }
  }, [theme])
  useEffect(() => write('df.sidebar', collapsed), [collapsed])
  useEffect(() => write('df.notifications', notifications.slice(0, 30)), [notifications])
  useEffect(() => write('df.recent', recent.slice(0, 12)), [recent])

  const notify = useCallback((message, { kind = 'info', persist = false, link } = {}) => {
    const id = Math.random().toString(36).slice(2)
    setToasts(t => [...t, { id, message, kind }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), kind === 'error' ? 8000 : 4500)
    if (persist) setNotifications(n => [{ id, message, kind, link, at: Date.now(), read: false }, ...n].slice(0, 30))
  }, [])
  const dismissToast = id => setToasts(t => t.filter(x => x.id !== id))
  const rememberWorkspace = useCallback((ws) => {
    if (!ws?.id) return
    setRecent(r => [{ id: ws.id, name: ws.name, rows: ws.active_dataset?.rows, at: Date.now() }, ...r.filter(x => x.id !== ws.id)])
  }, [])
  const forgetWorkspace = useCallback(id => setRecent(r => r.filter(x => x.id !== id)), [])

  const value = useMemo(() => ({
    theme, setTheme: setThemeState, collapsed, setCollapsed, toasts, notify, dismissToast,
    notifications, setNotifications, recent, rememberWorkspace, forgetWorkspace,
  }), [theme, collapsed, toasts, notify, notifications, recent, rememberWorkspace, forgetWorkspace])
  return <UiCtx.Provider value={value}>{children}</UiCtx.Provider>
}

export const useUi = () => useContext(UiCtx)
