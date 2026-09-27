import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import type { LayerGroup, LatLngBoundsExpression, Polyline } from 'leaflet'
import './style.css'

import { api, pollJob } from './api'
import { frameAt, frameToUrl } from './gridImage'
import { createPane, syncMaps, type Pane } from './map'
import type { AppConfig, Job, Layers, Meta, Summary } from './types'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

const LAYER_MODE: Record<string, 'seq' | 'div'> = {
  coarse: 'seq',
  coarse_bilinear: 'seq',
  reference: 'seq',
  prediction: 'seq',
  cloud_gap: 'seq',
  residual: 'div',
}

let config: AppConfig
let summary: Summary | null = null
let meta: Meta | null = null
let layers: Layers | null = null
let hasModel = false
let jobRunning = false
let timeIdx = 0
let playTimer: number | null = null
let leftPane: Pane
let rightPane: Pane

function status(text: string, isError = false): void {
  const el = $('jobStatus')
  el.textContent = text
  el.classList.toggle('error', isError)
}

function setProgress(p: number): void {
  $('progressBar').style.width = `${Math.round(p * 100)}%`
}

async function runJob(
  start: Promise<{ job_id: string }>,
  done: () => Promise<void>,
  label: string,
): Promise<void> {
  jobRunning = true
  updateButtons()
  setProgress(0.03)
  try {
    const { job_id } = await start
    const job: Job = await pollJob(job_id, (j) => {
      setProgress(j.progress)
      status(`${label}: ${j.stage} ${Math.round(j.progress * 100)}%`)
    })
    if (job.status === 'error') throw new Error(job.error || 'job failed')
    setProgress(1)
    status(`${label} ✓`)
    await done()
  } catch (err) {
    status(err instanceof Error ? err.message : String(err), true)
    setProgress(0)
  } finally {
    jobRunning = false
    updateButtons()
    window.setTimeout(() => setProgress(0), 1500)
  }
}

function updateButtons(): void {
  $<HTMLButtonElement>('btnFetch').disabled = jobRunning
  $<HTMLButtonElement>('btnTrain').disabled = jobRunning || !summary || !summary.has_reference
  $<HTMLButtonElement>('btnApply').disabled = jobRunning || !summary || !hasModel
  $('btnFetch').textContent = jobRunning ? '⏳ working…' : '⬇ Fetch coarse data'
}

function fmtDay(d: Date): string {
  return d.toISOString().slice(0, 10)
}

function renderDataInfo(): void {
  const el = $('dataInfo')
  if (!summary) {
    el.textContent = 'no dataset loaded'
    return
  }
  const src = summary.sources || {}
  el.innerHTML = [
    `<b>${summary.preset_label}</b> · ${summary.start_date} → ${summary.end_date}`,
    `grid ${summary.n_lat}×${summary.n_lon} @ ${summary.fine_step}° · ${summary.n_times} hours`,
    `cloud gap: <b>${Math.round(summary.gap_fraction * 100)}%</b> of coarse pixels repaired`,
    `fine reference: ${summary.has_reference ? 'yes (independent 0.1° product)' : 'none — transfer mode'}`,
    src.roads ? 'features: NO₂ + met + DEM + OSM roads' : 'features: NO₂ + met + DEM (OSM roads unavailable)',
  ].join('<br>')
}

function renderWarnings(): void {
  const el = $('warnings')
  const msgs: string[] = []
  if (summary) {
    for (const w of summary.warnings || []) msgs.push(w)
    if (meta && meta.summary_key && meta.summary_key !== summary.key) {
      msgs.push('current results belong to a previous dataset — retrain for the loaded one')
    }
  }
  if (meta) {
    for (const w of meta.warnings || []) if (!msgs.includes(w)) msgs.push(w)
  }
  el.innerHTML = msgs.map((m) => `⚠ ${m}`).join('<br>')
}

function card(kind: string, key: string, value: string, base?: string): string {
  return `<div class="card ${kind}"><div class="k">${key}</div><div class="v">${value}</div>${
    base ? `<div class="b">${base}</div>` : ''
  }</div>`
}

function renderMetrics(): void {
  const el = $('metrics')
  if (!meta) {
    el.textContent = 'run training to see metrics'
    renderScatter(null)
    renderImportance(null)
    return
  }
  const m = meta.metrics ?? (meta.mode === 'transfer' ? meta.benchmark?.metrics : undefined)
  if (!m) {
    el.textContent = 'no held-out metrics for this result'
    renderScatter(null)
    renderImportance(meta.importances ?? meta.benchmark?.importances ?? null)
    return
  }
  const splitLabel = meta.holdout_description ?? meta.benchmark?.holdout_description
  const nHold = meta.n_test ?? meta.benchmark?.n_test ?? m.n
  el.innerHTML = [
    card(
      m.skill_vs_baseline >= 0 ? 'good' : 'warn',
      'skill vs bilinear',
      `${m.skill_vs_baseline >= 0 ? '+' : ''}${(m.skill_vs_baseline * 100).toFixed(1)}%`,
      'RMSE reduction vs blurry baseline',
    ),
    card('', 'RMSE', `${m.rmse}`, `baseline ${m.baseline_rmse}`),
    card('', 'MAE', `${m.mae}`, `baseline ${m.baseline_mae}`),
    card(m.pattern_r2 >= 0.5 ? 'good' : 'warn', 'pattern r²', `${m.pattern_r2}`, `r = ${m.pearson}`),
    card('', 'sample n', String(nHold), splitLabel || ''),
    card('', 'cloud gap', `${Math.round((meta.gap_fraction ?? 0) * 100)}%`, 'coarse pixels repaired'),
  ].join('')
  renderScatter(meta)
  renderImportance(meta.importances ?? meta.benchmark?.importances ?? null)
}

function renderScatter(m: Meta | null): void {
  const svg = $('scatter')
  svg.innerHTML = ''
  if (!m || !layers) return
  const pred = frameAt(layers.layers.prediction, -1)
  const ref = frameAt(layers.layers.reference, -1)
  if (!pred || !ref) return
  const pts: [number, number][] = []
  for (let r = 0; r < pred.length; r++) {
    for (let c = 0; c < pred[0].length; c++) {
      const a = pred[r][c]
      const b = ref[r][c]
      if (typeof a === 'number' && typeof b === 'number' && isFinite(a) && isFinite(b)) pts.push([b, a])
    }
  }
  if (!pts.length) return
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const xmin = Math.min(...xs)
  const xmax = Math.max(...xs)
  const ymin = Math.min(...ys)
  const ymax = Math.max(...ys)
  const pad = 26
  const sx = (v: number) => pad + ((v - xmin) / (xmax - xmin || 1)) * (210 - pad * 2)
  const sy = (v: number) => 210 - pad - ((v - ymin) / (ymax - ymin || 1)) * (210 - pad * 2)
  const ns = 'http://www.w3.org/2000/svg'
  const line = document.createElementNS(ns, 'line')
  line.setAttribute('x1', String(sx(xmin)))
  line.setAttribute('y1', String(sy(xmin)))
  line.setAttribute('x2', String(sx(xmax)))
  line.setAttribute('y2', String(sy(xmax)))
  line.setAttribute('stroke', '#4a5768')
  line.setAttribute('stroke-dasharray', '4 3')
  svg.appendChild(line)
  for (const [x, y] of pts) {
    const circle = document.createElementNS(ns, 'circle')
    circle.setAttribute('cx', String(sx(x)))
    circle.setAttribute('cy', String(sy(y)))
    circle.setAttribute('r', '2.4')
    circle.setAttribute('fill', '#35d0c0')
    circle.setAttribute('fill-opacity', '0.55')
    svg.appendChild(circle)
  }
  const label = document.createElementNS(ns, 'text')
  label.setAttribute('x', '30')
  label.setAttribute('y', '14')
  label.setAttribute('fill', '#8b98a9')
  label.setAttribute('font-size', '9')
  label.textContent = 'x = reference, y = ML (dashed = perfect)'
  svg.appendChild(label)
}

function renderImportance(items: { feature: string; importance: number }[] | null): void {
  const el = $('importance')
  if (!items || !items.length) {
    el.innerHTML = '<div class="info">n/a for transfer runs</div>'
    return
  }
  const top = items.slice(0, 10)
  const max = top[0].importance || 1
  el.innerHTML = top
    .map(
      (it) =>
        `<div class="imp-row"><span class="lbl" title="${it.feature}">${it.feature}</span><span class="bar" style="width:${Math.max(
          4,
          (it.importance / max) * 90,
        )}px"></span><span>${(it.importance * 100).toFixed(1)}%</span></div>`,
    )
    .join('')
}

function layerBounds(ls: Layers): LatLngBoundsExpression {
  const dlat = ls.lats.length > 1 ? ls.lats[1] - ls.lats[0] : 0.05
  const dlon = ls.lons.length > 1 ? ls.lons[1] - ls.lons[0] : 0.05
  return L.latLngBounds([
    [ls.lats[0] - dlat / 2, ls.lons[0] - dlon / 2],
    [ls.lats[ls.lats.length - 1] + dlat / 2, ls.lons[ls.lons.length - 1] + dlon / 2],
  ])
}

function layerStyle(name: string): { vmin: number; vmax: number; mode: 'seq' | 'div' } {
  const mode = LAYER_MODE[name] || 'seq'
  const r = layers?.ranges?.[name]
  if (!r) return { vmin: 0, vmax: 1, mode }
  if (mode === 'div') {
    const m = Math.max(Math.abs(r[0]), Math.abs(r[1]), 0.5)
    return { vmin: -m, vmax: m, mode }
  }
  return { vmin: r[0], vmax: r[1], mode }
}

function renderOverlays(): void {
  if (!layers) return
  const bounds = layerBounds(layers)
  const leftName = ($('layerLeft') as HTMLSelectElement).value
  const rightName = ($('layerRight') as HTMLSelectElement).value
  const ls = layerStyle(leftName)
  leftPane.setOverlay(frameToUrl(frameAt(layers.layers[leftName], timeIdx), ls.vmin, ls.vmax, ls.mode), bounds, leftName === 'cloud_gap' ? 0.85 : 0.75)
  const rs = layerStyle(rightName)
  rightPane.setOverlay(frameToUrl(frameAt(layers.layers[rightName], timeIdx), rs.vmin, rs.vmax, rs.mode), bounds, rightName === 'residual' ? 0.8 : 0.75)
  drawLegend(rs)
  updateLegendLabels(rs)
}

function drawLegend(style: { vmin: number; vmax: number; mode: 'seq' | 'div' }): void {
  const canvas = $<HTMLCanvasElement>('legendCanvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  for (let x = 0; x < canvas.width; x++) {
    const v = style.vmin + ((style.vmax - style.vmin) * x) / (canvas.width - 1)
    const t = (v - style.vmin) / (style.vmax - style.vmin || 1)
    ctx.fillStyle =
      style.mode === 'div'
        ? `rgb(${divColor(t)})`
        : `rgb(${seqColor(t)})`
    ctx.fillRect(x, 0, 1, canvas.height)
  }
}

function seqColor(t: number): string {
  const anchors = [
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [109, 205, 89],
    [180, 222, 44],
    [253, 231, 37],
  ]
  return ramp(anchors, t)
}

function divColor(t: number): string {
  const anchors = [
    [59, 76, 192],
    [122, 161, 233],
    [201, 221, 237],
    [221, 221, 221],
    [245, 179, 100],
    [228, 108, 48],
    [180, 4, 38],
  ]
  return ramp(anchors, t)
}

function ramp(anchors: number[][], tRaw: number): string {
  const t = Math.min(1, Math.max(0, tRaw)) * (anchors.length - 1)
  const i = Math.min(anchors.length - 2, Math.floor(t))
  const f = t - i
  const a = anchors[i]
  const b = anchors[i + 1]
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f))
  return `${c[0]},${c[1]},${c[2]}`
}

function updateLegendLabels(style: { vmin: number; vmax: number }): void {
  $('legendLabels').textContent = ''
  const span = document.createElement('span')
  span.textContent = style.vmin.toFixed(1)
  const span2 = document.createElement('span')
  span2.textContent = `${style.vmax.toFixed(1)} µg/m³`
  $('legendLabels').appendChild(span)
  $('legendLabels').appendChild(span2)
}

const gridLayers: { coarse: LayerGroup | null; fine: LayerGroup | null }[] = [
  { coarse: null, fine: null },
  { coarse: null, fine: null },
]

function makeEdges(lo: number, hi: number, step: number): number[] {
  const out = [lo]
  let x = lo
  while (x + step < hi - 1e-9) {
    x += step
    out.push(x)
  }
  out.push(hi)
  return out
}

function drawGridlines(): void {
  const panes = [leftPane, rightPane]
  panes.forEach((pane, i) => {
    const store = gridLayers[i]
    if (store.coarse) {
      pane.map.removeLayer(store.coarse)
      store.coarse = null
    }
    if (store.fine) {
      pane.map.removeLayer(store.fine)
      store.fine = null
    }
    if (!summary) return
    if (!pane.map.getPane('gridlinePane')) {
      pane.map.createPane('gridlinePane').style.zIndex = '410'
    }
    const [lonMin, latMin, lonMax, latMax] = summary.bbox
    const mk = (
      latEdges: number[],
      lonEdges: number[],
      color: string,
      weight: number,
      opacity: number,
    ): LayerGroup => {
      const lines: Polyline[] = []
      for (const la of latEdges) {
        lines.push(
          L.polyline(
            [
              [la, lonMin],
              [la, lonMax],
            ],
            { pane: 'gridlinePane', color, weight, opacity, interactive: false },
          ),
        )
      }
      for (const lo of lonEdges) {
        lines.push(
          L.polyline(
            [
              [latMin, lo],
              [latMax, lo],
            ],
            { pane: 'gridlinePane', color, weight, opacity, interactive: false },
          ),
        )
      }
      return L.layerGroup(lines)
    }
    if ($<HTMLInputElement>('gridCoarse').checked) {
      const cs = Number(summary.coarse_step) || 0.25
      store.coarse = mk(makeEdges(latMin, latMax, cs), makeEdges(lonMin, lonMax, cs), '#ffb454', 1.6, 0.9)
      store.coarse.addTo(pane.map)
    }
    if (i === 1 && $<HTMLInputElement>('gridFine').checked) {
      const fs = Number(summary.fine_step) || 0.05
      store.fine = mk(makeEdges(latMin, latMax, fs), makeEdges(lonMin, lonMax, fs), '#ffffff', 0.6, 0.35)
      store.fine.addTo(pane.map)
    }
  })
}

function updateTimeLabel(): void {  if (!layers) {
    $('timeLabel').textContent = '–'
    return
  }
  if (timeIdx >= layers.t_len) {
    $('timeLabel').textContent = 'period mean'
    return
  }
  $('timeLabel').textContent = layers.times[timeIdx].replace('T', ' ').replace(':00Z', 'Z')
}

async function loadLayers(): Promise<boolean> {
  try {
    layers = await api.layers()
    timeIdx = layers.t_len
    const slider = $<HTMLInputElement>('timeSlider')
    slider.max = String(layers.t_len)
    slider.value = String(timeIdx)
    leftPane.fit(layerBounds(layers))
    renderOverlays()
    updateTimeLabel()
    renderMetrics()
    return true
  } catch {
    layers = null
    return false
  }
}

async function refreshState(): Promise<void> {
  const st = await api.state()
  summary = st.summary
  meta = st.meta
  hasModel = st.has_model
  const s = summary
  if (s) {
    const sel = $<HTMLSelectElement>('preset')
    if (Array.from(sel.options).some((o) => o.value === s.preset)) {
      sel.value = s.preset
    }
  }
  renderDataInfo()
  renderWarnings()
  renderMetrics()
  drawGridlines()
  updateButtons()
}

function setPresetDefaults(): void {
  const id = ($('preset') as HTMLSelectElement).value
  const p = config.presets.find((x) => x.id === id)
  if (!p) return
  const bounds = layerBoundsFor(p.bbox)
  if (layers === null) leftPane.fit(bounds)
}

function layerBoundsFor(bbox: number[]): LatLngBoundsExpression {
  return L.latLngBounds([
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ])
}

async function init(): Promise<void> {
  leftPane = createPane($('mapLeft'))
  rightPane = createPane($('mapRight'))
  syncMaps(leftPane.map, rightPane.map)

  config = await api.config()
  const presetSel = $<HTMLSelectElement>('preset')
  for (const p of config.presets) {
    const opt = document.createElement('option')
    opt.value = p.id
    opt.textContent = p.label
    presetSel.appendChild(opt)
  }
  presetSel.value = config.defaults.preset
  const modelSel = $<HTMLSelectElement>('model')
  for (const m of config.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.label
    modelSel.appendChild(opt)
  }
  modelSel.value = config.defaults.model
  const splitSel = $<HTMLSelectElement>('split')
  for (const s of config.splits) {
    const opt = document.createElement('option')
    opt.value = s.id
    opt.textContent = s.label
    splitSel.appendChild(opt)
  }
  splitSel.value = config.defaults.split

  const end = new Date()
  end.setUTCDate(end.getUTCDate() - 1)
  const start = new Date(end)
  start.setUTCDate(start.getUTCDate() - 6)
  $<HTMLInputElement>('endDate').value = fmtDay(end)
  $<HTMLInputElement>('startDate').value = fmtDay(start)

  await refreshState()
  const loaded = summary ? await loadLayers() : false
  if (!loaded) leftPane.fit(layerBoundsFor(config.presets.find((p) => p.id === presetSel.value)!.bbox))
  status(summary ? 'previous results loaded' : 'ready')

  $('btnFetch').addEventListener('click', () => {
    void runJob(
      api.fetch({
        preset: presetSel.value,
        start_date: $<HTMLInputElement>('startDate').value,
        end_date: $<HTMLInputElement>('endDate').value,
        cloud_threshold: Number($<HTMLInputElement>('cloudThreshold').value) || 60,
      }),
      async () => {
        await refreshState()
        renderOverlays()
      },
      'fetch',
    )
  })

  $('btnTrain').addEventListener('click', () => {
    void runJob(
      api.train({
        model: modelSel.value,
        split: splitSel.value,
        conserve: $<HTMLInputElement>('conserve').checked,
      }),
      async () => {
        await refreshState()
        await loadLayers()
      },
      'train',
    )
  })

  $('btnApply').addEventListener('click', () => {
    void runJob(
      api.apply({ conserve: $<HTMLInputElement>('conserve').checked }),
      async () => {
        await refreshState()
        await loadLayers()
      },
      'apply',
    )
  })

  presetSel.addEventListener('change', () => {
    setPresetDefaults()
    updateButtons()
  })
  ;($('layerLeft') as HTMLSelectElement).addEventListener('change', renderOverlays)
  ;($('layerRight') as HTMLSelectElement).addEventListener('change', renderOverlays)
  ;($('gridCoarse') as HTMLInputElement).addEventListener('change', drawGridlines)
  ;($('gridFine') as HTMLInputElement).addEventListener('change', drawGridlines)

  const slider = $<HTMLInputElement>('timeSlider')
  slider.addEventListener('input', () => {
    timeIdx = Number(slider.value)
    renderOverlays()
    updateTimeLabel()
  })

  $('btnPlay').addEventListener('click', () => {
    const btn = $<HTMLButtonElement>('btnPlay')
    if (playTimer !== null) {
      window.clearInterval(playTimer)
      playTimer = null
      btn.textContent = '▶'
      return
    }
    if (!layers) return
    btn.textContent = '⏸'
    playTimer = window.setInterval(() => {
      if (!layers) return
      timeIdx = timeIdx >= layers.t_len ? 0 : (timeIdx + 1) % layers.t_len
      slider.value = String(timeIdx)
      renderOverlays()
      updateTimeLabel()
    }, 450)
  })

  $<HTMLInputElement>('csvFile').addEventListener('change', (ev) => {
    const input = ev.target as HTMLInputElement
    const file = input.files?.[0]
    const out = $('stationResult')
    if (!file) return
    out.textContent = 'validating…'
    api
      .validateStations(file)
      .then((r) => {
        const m = r.metrics
        out.innerHTML = `n=${r.n_matched} · RMSE <b>${m.rmse}</b> vs baseline ${m.baseline_rmse} · r² ${
          m.pattern_r2
        } · skill ${(m.skill_vs_baseline * 100).toFixed(1)}%<br><span class="hint">${r.note}</span>`
      })
      .catch((err: unknown) => {
        out.textContent = err instanceof Error ? err.message : String(err)
      })
      .finally(() => {
        input.value = ''
      })
  })
}

init().catch((err: unknown) => {
  status(err instanceof Error ? err.message : String(err), true)
})
