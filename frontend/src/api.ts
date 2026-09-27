import type { ApiState, AppConfig, Job, Layers, Meta, ProbeResponse, StationResult, Summary } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

function post<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export const api = {
  config: () => request<AppConfig>('/api/config'),
  state: () => request<ApiState>('/api/state'),
  meta: () => request<Meta>('/api/result/meta'),
  layers: () => request<Layers>('/api/result/layers'),
  probe: (lat: number, lon: number) => request<ProbeResponse>(`/api/probe?lat=${lat}&lon=${lon}`),
  fetch: (body: object) => post<{ job_id: string }>('/api/fetch', body),
  train: (body: object) => post<{ job_id: string }>('/api/train', body),
  apply: (body: object) => post<{ job_id: string }>('/api/apply', body),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cities: () => request<{ default: string; count: number; cities: import('./types').CityItem[] }>('/api/cities'),
  validation: () => request<import('./types').ValidationPayload>('/api/validation'),
  forecast: (city?: string) =>
    request<import('./types').ForecastResponse>(`/api/forecast?city=${encodeURIComponent(city || '')}`),
  validateStations: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<StationResult>('/api/validate/stations', { method: 'POST', body: form })
  },
}

export async function pollJob(id: string, onTick: (job: Job) => void): Promise<Job> {
  for (;;) {
    const job = await api.job(id)
    onTick(job)
    if (job.status === 'done' || job.status === 'error') return job
    await sleep(600)
  }
}

export type { Summary }
