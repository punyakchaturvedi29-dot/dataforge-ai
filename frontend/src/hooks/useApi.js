import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Minimal data-fetching hook: { data, error, loading, reload, setData }.
 * `deps` re-runs the request; stale responses from earlier calls are ignored.
 */
export function useApi(fn, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({ data: undefined, error: null, loading: enabled })
  const seq = useRef(0)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const run = useCallback(async () => {
    const my = ++seq.current
    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const data = await fnRef.current()
      if (my === seq.current) setState({ data, error: null, loading: false })
      return data
    } catch (error) {
      if (my === seq.current) setState(s => ({ ...s, error, loading: false }))
      return undefined
    }
  }, [])

  useEffect(() => {
    if (enabled) run()
    else setState(s => ({ ...s, loading: false }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled])

  const setData = useCallback(updater => setState(s => ({ ...s, data: typeof updater === 'function' ? updater(s.data) : updater })), [])
  return { ...state, reload: run, setData }
}

/** Wraps an async action with busy/error state. */
export function useAction(fn) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const run = useCallback(async (...args) => {
    setBusy(true)
    setError(null)
    try {
      return await fnRef.current(...args)
    } catch (e) {
      setError(e)
      throw e
    } finally {
      setBusy(false)
    }
  }, [])
  return { run, busy, error, setError }
}

/** Poll a background job until it finishes. */
export function useJob(fetchJob) {
  const [job, setJob] = useState(null)
  const timer = useRef(null)
  const stop = () => { if (timer.current) clearTimeout(timer.current); timer.current = null }
  const start = useCallback((jobId, onDone) => {
    stop()
    const tick = async () => {
      try {
        const j = await fetchJob(jobId)
        setJob(j)
        if (['queued', 'running'].includes(j.status)) timer.current = setTimeout(tick, 700)
        else onDone?.(j)
      } catch (e) {
        setJob(j => ({ ...(j || {}), status: 'failed', error: { message: e.message } }))
      }
    }
    tick()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => stop, [])
  return { job, start, setJob }
}
