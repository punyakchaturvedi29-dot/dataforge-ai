import axios from 'axios'

export const API_BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '')

export const http = axios.create({ baseURL: API_BASE, timeout: 180000 })

/** Normalised error: { status, code, message, details, requestId } */
export class ApiError extends Error {
  constructor({ status, code, message, details, requestId }) {
    super(message)
    this.status = status
    this.code = code
    this.details = details || {}
    this.requestId = requestId
  }
}

export function toApiError(err) {
  if (err instanceof ApiError) return err
  if (axios.isCancel?.(err) || err?.name === 'CanceledError') return new ApiError({ status: 0, code: 'cancelled', message: 'Request cancelled' })
  const res = err?.response
  if (!res) {
    return new ApiError({ status: 0, code: 'network', message: 'Cannot reach the DataForge server. Check that the backend is running on port 8000.' })
  }
  const e = res.data?.error
  return new ApiError({
    status: res.status,
    code: e?.code || 'http_error',
    message: e?.message || res.data?.detail || `Request failed (${res.status})`,
    details: e?.details,
    requestId: e?.request_id,
  })
}

async function call(p) {
  try { return (await p).data } catch (err) { throw toApiError(err) }
}

export const get = (url, params, opts = {}) => call(http.get(url, { params, ...opts }))
export const post = (url, body, opts = {}) => call(http.post(url, body, opts))
export const patch = (url, body) => call(http.patch(url, body))
export const del = (url, opts = {}) => call(http.delete(url, opts))

export async function upload(url, file, fields = {}, onProgress) {
  const form = new FormData()
  form.append('file', file)
  Object.entries(fields).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') form.append(k, v) })
  return call(http.post(url, form, { onUploadProgress: e => onProgress?.(e.total ? e.loaded / e.total : 0) }))
}

/** Download a binary response (PDF/CSV) and save it with the server-provided file name. */
export async function download(url, { method = 'get', body, params, fallbackName = 'download' } = {}) {
  try {
    const res = await http.request({ url, method, data: body, params, responseType: 'blob' })
    const cd = res.headers['content-disposition'] || ''
    const m = /filename\*=UTF-8''([^;]+)/i.exec(cd) || /filename="?([^";]+)"?/i.exec(cd)
    const name = m ? decodeURIComponent(m[1]) : fallbackName
    const href = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = href
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(href), 2000)
    return name
  } catch (err) {
    // Blob error bodies need decoding to get the JSON message
    if (err?.response?.data instanceof Blob) {
      try { err.response.data = JSON.parse(await err.response.data.text()) } catch { /* ignore */ }
    }
    throw toApiError(err)
  }
}
