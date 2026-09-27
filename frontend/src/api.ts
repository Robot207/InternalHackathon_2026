import type { ApiState, AppConfig, Job, Layers, Meta, StationResult, Summary } from './types'

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
  fetch: (body: object) => post<{ job_id: string }>('/api/fetch', body),
  train: (body: object) => post<{ job_id: string }>('/api/train', body),
  apply: (body: object) => post<{ job_id: string }>('/api/apply', body),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  validateStations: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<StationResult>('/api/validate/stations', { method: 'POST', body: form })
  },
  getBenchmarkStations: (preset = 'mumbai') =>
    request<{ preset: string; region: string; stations: import('./types').StationBenchmarkItem[]; landmarks: import('./types').LandmarkItem[] }>(
      `/api/stations/benchmark?preset=${encodeURIComponent(preset)}`,
    ),
  validateBenchmarkStations: (preset = 'mumbai') =>
    post<import('./types').StationBenchmarkResponse>(
      `/api/stations/benchmark/validate?preset=${encodeURIComponent(preset)}`,
      {},
    ),
  benchmarkModels: (split = 'spatiotemporal', conserve = true) =>
    post<{ job_id: string }>(
      `/api/benchmark/models?split=${encodeURIComponent(split)}&conserve=${conserve}`,
      {},
    ),
  /**
   * Point inspector. `t` is the timeline frame the map is painting: 0..n-1 is
   * that hour, `n` (the slider's end stop) is the period-mean frame. Omitting it
   * falls back to the latest hour.
   */
  inspectPoint: (lat: number, lon: number, t?: number) =>
    request<import('./types').PointInspectResult>(
      `/api/point/inspect?lat=${lat.toFixed(5)}&lon=${lon.toFixed(5)}${
        t === undefined ? '' : `&t=${Math.round(t)}`
      }`,
    ),
  cities: () => request<import('./types').CityListResponse>('/api/cities'),
  loso: (preset?: string) =>
    request<import('./types').Validation>(
      `/api/validation/loso${preset ? `?preset=${encodeURIComponent(preset)}` : ''}`,
    ),
  forecast: (preset: string) =>
    request<import('./types').Forecast>(
      `/api/forecast?preset=${encodeURIComponent(preset)}`,
    ),
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
