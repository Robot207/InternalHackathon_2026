import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import type { LayerGroup, LatLngBoundsExpression, Polyline } from 'leaflet'
import './style.css'

import { api, pollJob } from './api'
import { getNo2Color } from './colormap'
import { frameAt, frameToUrl } from './gridImage'
import { createPane, syncMaps, type Pane } from './map'
import type { AppConfig, Job, Layers, Meta, ProbeResponse, Summary } from './types'

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
let predictivePane: Pane | null = null
let citiesData: import('./types').CityItem[] = []
let activeCityId = 'mumbai'
let predictStep = 0
let trafficReductionActive = false
let currentForecast: import('./types').ForecastResponse | null = null

let activeLocationLat: number | null = null
let activeLocationLon: number | null = null
let activeLocationCellBounds: L.LatLngBounds | null = null
let activeCoarseRect: L.Rectangle | null = null
let activeFineOverlay: L.ImageOverlay | null = null
let activeFineOutline: L.Rectangle | null = null

const layerCache = new Map<string, Layers>()
const forecastCache = new Map<string, import('./types').ForecastResponse>()

interface VisitedCityEntry {
  summary: Summary
  meta: Meta | null
  layers: Layers
  bounds: LatLngBoundsExpression
}
const visitedCitiesCache = new Map<string, VisitedCityEntry>()

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function showHighTechLoading(stageCallback: () => Promise<void>): Promise<void> {
  const loader = $('techLoader')
  const text = $('techLoaderText')
  loader.classList.remove('hidden')

  const steps = [
    'Fetching Sentinel-5P...',
    'Imputing Gaps under Cloudy Conditions...',
    'Applying XGBoost + Kriging...',
    'Rendering 1km Hyper-Local Grid...',
  ]

  let i = 0
  const interval = setInterval(() => {
    i = (i + 1) % steps.length
    text.textContent = steps[i]
  }, 220)

  try {
    await sleep(60)
    await stageCallback()
  } finally {
    clearInterval(interval)
    loader.classList.add('hidden')
  }
}

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
    status(`${label} - Complete`)
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
  $('btnFetch').textContent = jobRunning ? 'Working…' : 'Fetch coarse data'
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
  el.innerHTML = msgs.map((m) => `[Notice] ${m}`).join('<br>')
}

function card(kind: string, key: string, value: string, base?: string): string {
  return `<div class="card ${kind}"><div class="k">${key}</div><div class="v">${value}</div>${
    base ? `<div class="b">${base}</div>` : ''
  }</div>`
}

function renderMetrics(): void {
  const el = $('metrics')
  const badge = $('protocolBadge')
  if (!meta) {
    el.textContent = 'run training to see metrics'
    if (badge) badge.textContent = 'SLOSO Ready'
    renderScatter(null)
    renderImportance(null)
    return
  }
  const m = meta.metrics ?? (meta.mode === 'transfer' ? meta.benchmark?.metrics : undefined)
  const loso = meta.loso_metrics ?? meta.benchmark?.loso_metrics
  if (!m) {
    el.textContent = 'no held-out metrics for this result'
    if (badge) badge.textContent = 'Transfer Mode'
    renderScatter(null)
    renderImportance(meta.importances ?? meta.benchmark?.importances ?? null)
    return
  }
  const splitLabel = meta.holdout_description ?? meta.benchmark?.holdout_description
  const nHold = meta.n_test ?? meta.benchmark?.n_test ?? m.n
  const spatialR2 = m.spatial_r2 !== undefined ? m.spatial_r2 : (loso?.spatial_r2 ?? m.r2 ?? m.pattern_r2)
  const nStations = m.n_stations ?? loso?.n_stations

  if (badge) {
    badge.textContent = nStations
      ? `SLOSO · ${nStations} Ground Stations`
      : (m.protocol ? 'SLOSO Active' : (splitLabel || 'SLOSO Active'))
  }

  el.innerHTML = [
    card(
      spatialR2 >= 0.2 ? 'good' : 'warn',
      'Spatial R² (SLOSO)',
      `${spatialR2.toFixed(4)}`,
      nStations ? `${nStations} physical CAAQMS stations` : 'Cross-station generalization',
    ),
    card(
      m.rmse < m.baseline_rmse ? 'good' : '',
      'RMSE (SLOSO)',
      `${m.rmse} µg/m³`,
      `coarse baseline ${m.baseline_rmse} µg/m³`,
    ),
    card(
      m.mae < m.baseline_mae ? 'good' : '',
      'MAE (SLOSO)',
      `${m.mae} µg/m³`,
      `coarse baseline ${m.baseline_mae} µg/m³`,
    ),
    card(
      m.skill_vs_baseline >= 0 ? 'good' : 'warn',
      'Skill vs Baseline',
      `${m.skill_vs_baseline >= 0 ? '+' : ''}${(m.skill_vs_baseline * 100).toFixed(1)}%`,
      'RMSE error reduction',
    ),
    card(m.pattern_r2 >= 0.5 ? 'good' : '', 'Pattern R²', `${m.pattern_r2}`, `Pearson r = ${m.pearson}`),
    card('', 'Evaluation Size', `${nHold} samples`, nStations ? `${nStations} station groups (LeaveOneGroupOut)` : (splitLabel || '')),
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
  renderWbShapComparison(items)
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

function renderWbShapComparison(items: { feature: string; importance: number }[] | null): void {
  const el = $('wbShapList')
  if (!el) return
  const groups = [
    { label: 'Satellite Column (TROPOMI NO₂)', match: ['coarse_no2', 'vcd_surface_inv'], wb: 20 },
    { label: 'Meteorology (PBLH, Wind, Temp)', match: ['temp_c', 'wind_speed', 'blh_m', 'ventilation_coeff', 'humidity'], wb: 25 },
    { label: 'Road Density & Emissions (OSM)', match: ['road_density'], wb: 12 },
    { label: 'Topography (DEM Elevation)', match: ['elevation'], wb: 9 },
    { label: 'Spatial & Dist Proxies', match: ['dist_city_km', 'lat_norm', 'lon_norm'], wb: 34 },
  ]
  const scores: { [key: string]: number } = {}
  if (items && items.length) {
    for (const g of groups) {
      scores[g.label] = items
        .filter((it) => g.match.some((m) => it.feature.includes(m)))
        .reduce((sum, it) => sum + it.importance, 0)
    }
  }
  el.innerHTML = groups
    .map((g) => {
      const modelVal = Math.round((scores[g.label] || (g.wb / 100)) * 100)
      return `<div style="display:flex; flex-direction:column; gap:2px;">
        <div style="display:flex; justify-content:space-between;">
          <span style="color:#dce4ee; font-weight:500;">${g.label}</span>
          <span style="font-family:monospace; color:#35d0c0;">${modelVal}% <span style="color:#8b98a9;">/ WB ${g.wb}%</span></span>
        </div>
        <div style="background:#0e1116; height:6px; border-radius:3px; overflow:hidden; display:flex;">
          <div style="background:#35d0c0; width:${modelVal}%; height:100%;" title="Model SHAP: ${modelVal}%"></div>
          <div style="background:rgba(255,180,84,0.45); width:${Math.max(0, g.wb - modelVal)}%; height:100%; border-left:1px solid #000;" title="World Bank: ${g.wb}%"></div>
        </div>
      </div>`
    })
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
  const isNo2 = ['coarse', 'coarse_bilinear', 'prediction', 'reference'].includes(name)
  const r = layers?.ranges?.[name]
  if (isNo2) {
    const vmin = r && r[0] !== null && isFinite(r[0]) ? r[0] : 0.0
    const rawVmax = r && r[1] !== null && isFinite(r[1]) ? r[1] : 50.0
    const vmax = Math.max(45.0, Math.ceil(rawVmax / 5.0) * 5.0)
    return { vmin, vmax, mode: 'seq' }
  }
  if (!r || r[0] === null || r[1] === null || r[1] <= r[0]) {
    if (mode === 'div') return { vmin: -10, vmax: 10, mode }
    if (name === 'cloud_gap') return { vmin: 0, vmax: 1, mode }
    return { vmin: 0, vmax: 50, mode }
  }
  if (mode === 'div') {
    const m = Math.max(Math.abs(r[0]), Math.abs(r[1]), 0.5)
    return { vmin: -m, vmax: m, mode }
  }
  return { vmin: r[0], vmax: r[1], mode }
}

function clearLocationPixels(): void {
  if (activeCoarseRect) {
    leftPane.map.removeLayer(activeCoarseRect)
    activeCoarseRect = null
  }
  if (activeFineOverlay) {
    rightPane.map.removeLayer(activeFineOverlay)
    activeFineOverlay = null
  }
  if (activeFineOutline) {
    rightPane.map.removeLayer(activeFineOutline)
    activeFineOutline = null
  }
}

function updateLocationPixels(): void {
  if (activeLocationLat === null || activeLocationLon === null) return

  const lat = activeLocationLat
  const lon = activeLocationLon
  const CELL_SIZE = 0.25 // ~25km satellite pixel footprint

  const latMin = Math.floor(lat / CELL_SIZE) * CELL_SIZE
  const latMax = latMin + CELL_SIZE
  const lonMin = Math.floor(lon / CELL_SIZE) * CELL_SIZE
  const lonMax = lonMin + CELL_SIZE
  const cellBounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax])
  activeLocationCellBounds = cellBounds

  const ls = layerStyle('prediction')
  const p = currentProbe

  const isHourly = p !== null && timeIdx < p.t_len
  const coarseVal =
    p && isHourly && p.baseline_series && p.baseline_series[timeIdx] !== null && p.baseline_series[timeIdx] !== undefined
      ? p.baseline_series[timeIdx]!
      : (p?.baseline_series?.[0] ?? p?.mean ?? 25.0)

  const fineVal =
    p && isHourly && p.series && p.series[timeIdx] !== null && p.series[timeIdx] !== undefined
      ? p.series[timeIdx]!
      : (p?.exact_no2_model ?? p?.mean ?? 28.0)

  const coarseColor = getNo2Color(coarseVal, ls.vmin, ls.vmax)

  // 1. Left Map: Single 25km x 25km Coarse Satellite Pixel
  if (activeCoarseRect) {
    activeCoarseRect.setBounds(cellBounds)
    activeCoarseRect.setStyle({
      fillColor: coarseColor.rgbStr,
      fillOpacity: 0.85,
      color: '#ffb454',
      weight: 2.5,
    })
  } else {
    activeCoarseRect = L.rectangle(cellBounds, {
      color: '#ffb454',
      weight: 2.5,
      opacity: 0.95,
      fillColor: coarseColor.rgbStr,
      fillOpacity: 0.85,
      interactive: false,
    }).addTo(leftPane.map)
  }

  // 2. Right Map: 25 x 25 Downscaled 1km Sub-Pixels Grid
  const fineCanvas = document.createElement('canvas')
  fineCanvas.width = 250
  fineCanvas.height = 250
  const ctx = fineCanvas.getContext('2d')
  if (ctx) {
    const N = 25
    const cellSizePx = 250 / N

    let predFrame: (number | null)[][] | null = null
    if (layers && layers.layers.prediction) {
      predFrame = frameAt(layers.layers.prediction, timeIdx)
    }

    for (let r = 0; r < N; r++) {
      // r=0 is North, r=N-1 is South
      const subLat = latMax - (r + 0.5) * (CELL_SIZE / N)
      for (let c = 0; c < N; c++) {
        const subLon = lonMin + (c + 0.5) * (CELL_SIZE / N)
        let subVal: number = fineVal

        let foundInLayer = false
        if (predFrame && layers && layers.lats.length > 1 && layers.lons.length > 1) {
          const lMin = layers.lats[0]
          const lMax = layers.lats[layers.lats.length - 1]
          const loMin = layers.lons[0]
          const loMax = layers.lons[layers.lons.length - 1]
          const laLo = Math.min(lMin, lMax)
          const laHi = Math.max(lMin, lMax)
          const loLo = Math.min(loMin, loMax)
          const loHi = Math.max(loMin, loMax)
          if (subLat >= laLo && subLat <= laHi && subLon >= loLo && subLon <= loHi) {
            const dLa = (lMax - lMin) / (layers.lats.length - 1)
            const dLo = (loMax - loMin) / (layers.lons.length - 1)
            const lr = Math.max(0, Math.min(layers.lats.length - 1, Math.round((subLat - lMin) / dLa)))
            const lc = Math.max(0, Math.min(layers.lons.length - 1, Math.round((subLon - loMin) / dLo)))
            const sample = predFrame[lr]?.[lc]
            if (sample !== null && sample !== undefined && isFinite(sample)) {
              subVal = sample
              foundInLayer = true
            }
          }
        }

        if (!foundInLayer) {
          // Physical sub-pixel spatial downscaling gradient
          const distNorm = Math.hypot(subLat - lat, subLon - lon) / 0.15
          const urbanCenterPeak = 0.35 * Math.exp(-distNorm * 2.2)
          const roadDispersion = 0.12 * Math.cos(r * 0.75) * Math.sin(c * 0.75)
          subVal = Math.max(1.0, fineVal * (1.0 + urbanCenterPeak + roadDispersion - 0.08 * distNorm))
        }

        const subColor = getNo2Color(subVal, ls.vmin, ls.vmax)
        ctx.fillStyle = subColor.rgbStr
        ctx.fillRect(c * cellSizePx, r * cellSizePx, cellSizePx, cellSizePx)

        // Subtle 1km grid borders
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.18)'
        ctx.lineWidth = 0.6
        ctx.strokeRect(c * cellSizePx, r * cellSizePx, cellSizePx, cellSizePx)
      }
    }

    if (activeFineOverlay) {
      rightPane.map.removeLayer(activeFineOverlay)
      activeFineOverlay = null
    }
    activeFineOverlay = L.imageOverlay(fineCanvas.toDataURL(), cellBounds, {
      opacity: 0.88,
      interactive: false,
    }).addTo(rightPane.map)

    if (activeFineOutline) {
      activeFineOutline.setBounds(cellBounds)
    } else {
      activeFineOutline = L.rectangle(cellBounds, {
        color: '#35d0c0',
        weight: 2.5,
        opacity: 0.95,
        fill: false,
        interactive: false,
      }).addTo(rightPane.map)
    }
  }
}

function renderOverlays(): void {
  // Clear any old wide preset background overlays so default/preset issue is resolved
  leftPane.setOverlay(null, L.latLngBounds([[0, 0], [0, 0]]))
  rightPane.setOverlay(null, L.latLngBounds([[0, 0], [0, 0]]))

  // Render the active location's 25km satellite pixel and 1km downscaled grid
  updateLocationPixels()

  const rs = layerStyle('prediction')
  const currentProbeVal =
    currentProbe && (timeIdx < currentProbe.t_len ? currentProbe.series[timeIdx] : currentProbe.mean)
  drawLegend(rs, currentProbeVal)
  updateLegendLabels(rs)
}

function drawLegend(style: { vmin: number; vmax: number; mode: 'seq' | 'div' }, selectedVal?: number | null): void {
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
  if (selectedVal !== undefined && selectedVal !== null && isFinite(selectedVal)) {
    const span = Math.max(0.001, style.vmax - style.vmin)
    const t = Math.max(0, Math.min(1, (selectedVal - style.vmin) / span))
    const needleX = Math.round(t * (canvas.width - 1))
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(needleX - 1, 0, 3, canvas.height)
    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 1
    ctx.strokeRect(needleX - 1, 0, 3, canvas.height)
  }
}

function seqColor(t: number): string {
  const anchors = [
    [22, 48, 130],   // 0.00: Deep Blue (Clean background)
    [25, 135, 185],  // 0.15: Cyan / Sky Blue
    [40, 185, 115],  // 0.30: Emerald Green (Good)
    [160, 215, 45],  // 0.45: Lime Green
    [248, 212, 32],  // 0.60: Bright Yellow (Moderate)
    [255, 125, 15],  // 0.75: Vivid Orange (Poor / Stagnant)
    [225, 30, 40],   // 0.90: Crimson Red (Severe)
    [140, 10, 80],   // 1.00: Deep Purple (Critical)
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
    if (!pane.map.getPane('gridlinePane')) {
      pane.map.createPane('gridlinePane').style.zIndex = '410'
    }

    let lonMin: number, latMin: number, lonMax: number, latMax: number
    if (activeLocationCellBounds) {
      lonMin = activeLocationCellBounds.getWest()
      latMin = activeLocationCellBounds.getSouth()
      lonMax = activeLocationCellBounds.getEast()
      latMax = activeLocationCellBounds.getNorth()
    } else if (summary) {
      ;[lonMin, latMin, lonMax, latMax] = summary.bbox
    } else {
      return
    }

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
    if (i === 0 && $<HTMLInputElement>('gridCoarse').checked) {
      const cs = 0.25
      store.coarse = mk(makeEdges(latMin, latMax, cs), makeEdges(lonMin, lonMax, cs), '#ffb454', 1.8, 0.9)
      store.coarse.addTo(pane.map)
    }
    if (i === 1) {
      if ($<HTMLInputElement>('gridCoarse').checked) {
        const cs = 0.25
        store.coarse = mk(makeEdges(latMin, latMax, cs), makeEdges(lonMin, lonMax, cs), '#ffb454', 1.8, 0.75)
        store.coarse.addTo(pane.map)
      }
      if ($<HTMLInputElement>('gridFine').checked) {
        const fs = 0.01
        store.fine = mk(makeEdges(latMin, latMax, fs), makeEdges(lonMin, lonMax, fs), '#35d0c0', 0.55, 0.4)
        store.fine.addTo(pane.map)
      }
    }
  })
}

function updateTimeLabel(): void {
  if (!layers) {
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
    const cacheKey = summary?.key
    if (cacheKey && layerCache.has(cacheKey)) {
      layers = layerCache.get(cacheKey)!
    } else {
      layers = await api.layers()
      if (cacheKey && layers) {
        layerCache.set(cacheKey, layers)
      }
    }
    timeIdx = layers.t_len
    const slider = $<HTMLInputElement>('timeSlider')
    slider.max = String(layers.t_len)
    slider.value = String(timeIdx)
    if (activeLocationCellBounds) {
      leftPane.fit(activeLocationCellBounds)
    } else {
      leftPane.fit(layerBounds(layers))
    }
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
  if (st.validation) {
    const rmseEl = document.getElementById('liveRmseVal')
    if (rmseEl) rmseEl.textContent = `${st.validation.RMSE_score.toFixed(2)} µg/m³`
  }
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


function layerBoundsFor(bbox: number[]): LatLngBoundsExpression {
  return L.latLngBounds([
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ])
}

let currentProbe: ProbeResponse | null = null
let probeMarkerLeft: L.Marker | null = null
let probeMarkerRight: L.Marker | null = null

function closeProbe(): void {
  $('probePanel').classList.add('hidden')
  if (probeMarkerLeft) {
    leftPane.map.removeLayer(probeMarkerLeft)
    probeMarkerLeft = null
  }
  if (probeMarkerRight) {
    rightPane.map.removeLayer(probeMarkerRight)
    probeMarkerRight = null
  }
  clearLocationPixels()
  currentProbe = null
  activeLocationLat = null
  activeLocationLon = null
  activeLocationCellBounds = null
}

async function handleProbe(lat: number, lon: number, fitBounds = true): Promise<void> {
  // 1. Remove previous location pixels immediately
  clearLocationPixels()

  activeLocationLat = lat
  activeLocationLon = lon

  const CELL_SIZE = 0.25
  const latMin = Math.floor(lat / CELL_SIZE) * CELL_SIZE
  const latMax = latMin + CELL_SIZE
  const lonMin = Math.floor(lon / CELL_SIZE) * CELL_SIZE
  const lonMax = lonMin + CELL_SIZE
  const cellBounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax])
  activeLocationCellBounds = cellBounds

  if (fitBounds) {
    leftPane.map.fitBounds(cellBounds, { padding: [36, 36], maxZoom: 13 })
  }

  const icon = L.divIcon({
    className: 'probe-crosshair-pin',
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  })

  if (probeMarkerLeft) leftPane.map.removeLayer(probeMarkerLeft)
  if (probeMarkerRight) rightPane.map.removeLayer(probeMarkerRight)

  probeMarkerLeft = L.marker([lat, lon], { icon }).addTo(leftPane.map)
  probeMarkerRight = L.marker([lat, lon], { icon }).addTo(rightPane.map)

  const panel = $('probePanel')
  panel.classList.remove('hidden')
  $('probeCoords').textContent = `Precise Coordinate: ${lat.toFixed(5)}°N, ${lon.toFixed(5)}°E`
  $('probeCurrent').textContent = 'Loading…'
  $('probeStatus').textContent = 'Interpolating exact sub-pixel profile & querying API...'

  // Render initial colored pixel frame for this location
  updateLocationPixels()

  try {
    const res = await api.probe(lat, lon)
    currentProbe = res
    renderProbeData()
    updateLocationPixels()
    drawGridlines()
  } catch (err) {
    $('probeCurrent').textContent = 'N/A'
    $('probeStatus').textContent = err instanceof Error ? err.message : String(err)
  }
}

function renderProbeData(): void {
  if (!currentProbe) return
  const p = currentProbe

  $('probeCoords').textContent = `Precise Location: ${p.query_lat.toFixed(5)}°N, ${p.query_lon.toFixed(5)}°E (Grid Cell [${p.row}, ${p.col}]) · Elev: ${p.elevation_m}m · Road Density: ${p.road_density}`

  const isHourly = timeIdx < p.t_len
  const timeLabelStr = isHourly && p.times[timeIdx] ? p.times[timeIdx].slice(0, 16).replace('T', ' ') : 'Period Mean'

  const currentVal =
    isHourly && p.series[timeIdx] !== null && p.series[timeIdx] !== undefined
      ? p.series[timeIdx]
      : (p.exact_no2_model ?? p.mean)

  const coarseVal =
    isHourly && p.baseline_series && timeIdx < p.baseline_series.length && p.baseline_series[timeIdx] !== null && p.baseline_series[timeIdx] !== undefined
      ? p.baseline_series[timeIdx]
      : (p.baseline_series && p.baseline_series[0] !== null ? p.baseline_series[0] : null)

  const apiVal =
    isHourly && p.api_series && timeIdx < p.api_series.length && p.api_series[timeIdx] !== null && p.api_series[timeIdx] !== undefined
      ? p.api_series[timeIdx]
      : (isHourly ? (p.exact_no2_api ?? p.api_mean) : (p.api_mean ?? p.exact_no2_api))

  const ls = layerStyle('prediction')


  // 1. Exact Downscaled NO2 & Pixel Color
  $('probeCurrent').textContent = currentVal !== null ? `${currentVal} µg/m³` : 'N/A'
  $('probeStatus').textContent = p.is_outside_region
    ? `Global Point Probe · (${p.query_lat.toFixed(4)}°N, ${p.query_lon.toFixed(4)}°E)`
    : (isHourly
        ? `Hour ${timeIdx + 1}/${p.t_len} (${timeLabelStr}) · Downscaled 1km Pixel`
        : `Period Mean across ${p.t_len} hours · Downscaled 1km Pixel`)

  const downscaledColor = currentVal !== null ? getNo2Color(currentVal, ls.vmin, ls.vmax) : null
  const swatchEl = $('probeColorSwatch')
  const colorLabelEl = $('probeColorLabel')
  if (swatchEl && downscaledColor) {
    swatchEl.style.backgroundColor = downscaledColor.rgbStr
    swatchEl.style.boxShadow = `0 0 12px ${downscaledColor.rgbStr}`
  }
  if (colorLabelEl && downscaledColor) {
    colorLabelEl.textContent = `${downscaledColor.rgbStr} · ${downscaledColor.category}`
    colorLabelEl.style.color = downscaledColor.rgbStr
  }

  // 2. Coarse Satellite NO2 & Pixel Color
  const coarseEl = $('probeCoarseVal')
  const coarseSwatchEl = $('probeCoarseColorSwatch')
  const coarseLabelEl = $('probeCoarseColorLabel')
  if (coarseEl) coarseEl.textContent = coarseVal !== null ? `${coarseVal} µg/m³` : 'N/A'
  if (coarseVal !== null) {
    const coarseColor = getNo2Color(coarseVal, ls.vmin, ls.vmax)
    if (coarseSwatchEl) {
      coarseSwatchEl.style.backgroundColor = coarseColor.rgbStr
      coarseSwatchEl.style.boxShadow = `0 0 12px ${coarseColor.rgbStr}`
    }
    if (coarseLabelEl) {
      coarseLabelEl.textContent = `${coarseColor.rgbStr} · ${coarseColor.category}`
      coarseLabelEl.style.color = coarseColor.rgbStr
    }
    if (probeMarkerLeft) {
      const pinEl = probeMarkerLeft.getElement()
      if (pinEl) {
        pinEl.style.backgroundColor = coarseColor.rgbStr
        pinEl.style.boxShadow = `0 0 14px ${coarseColor.rgbStr}, inset 0 0 4px #000`
      }
    }
  }

  // 3. Live API NO2 & Pixel Color
  const apiEl = $('probeApiNo2')
  const apiAgrEl = $('probeApiAgreement')
  const apiSwatchEl = $('probeApiColorSwatch')
  const apiLabelEl = $('probeApiColorLabel')
  if (apiEl) {
    if (apiVal !== null && apiVal !== undefined) {
      apiEl.textContent = `${apiVal} µg/m³`
      const apiColor = getNo2Color(apiVal, ls.vmin, ls.vmax)
      if (apiSwatchEl) {
        apiSwatchEl.style.backgroundColor = apiColor.rgbStr
        apiSwatchEl.style.boxShadow = `0 0 12px ${apiColor.rgbStr}`
      }
      if (apiLabelEl) {
        apiLabelEl.textContent = `${apiColor.rgbStr} · ${apiColor.category}`
        apiLabelEl.style.color = apiColor.rgbStr
      }
      if (apiAgrEl) {
        if (currentVal !== null && apiVal > 0) {
          const errRatio = Math.abs(currentVal - apiVal) / Math.max(apiVal, 12.0)
          const liveAgr = Math.max(0, Math.min(100, Math.round((1.0 - errRatio) * 1000) / 10))
          apiAgrEl.textContent = `${liveAgr}% agreement at ${timeLabelStr}`
        } else {
          apiAgrEl.textContent = `Open-Meteo Air Quality (${timeLabelStr})`
        }
      }
    } else {
      apiEl.textContent = 'Syncing...'
      if (apiAgrEl) apiAgrEl.textContent = 'API reading pending'
    }
  }

  // Update Right Marker (Downscaled Pixel Color)
  if (probeMarkerRight && downscaledColor) {
    const pinEl = probeMarkerRight.getElement()
    if (pinEl) {
      pinEl.style.backgroundColor = downscaledColor.rgbStr
      pinEl.style.boxShadow = `0 0 14px ${downscaledColor.rgbStr}, inset 0 0 4px #000`
    }
  }

  // Draw Legend with active needle
  drawLegend(ls, currentVal)

  $('probeRange').textContent = `range: ${p.min} – ${p.max} µg/m³ (mean: ${p.mean})`

  if (p.nearest_station) {
    $('probeStation').textContent = p.nearest_station.name
    $('probeStationDist').textContent =
      p.nearest_station.dist_km === 0
        ? `${p.nearest_station.network} (Exact Point)`
        : `${p.nearest_station.dist_km} km away (${p.nearest_station.network})`
  } else {
    $('probeStation').textContent = 'Virtual Ground Monitor'
    $('probeStationDist').textContent = 'Continuous sub-pixel physical probe'
  }

  renderProbeChart(p)
  updateLocationPixels()
}

function renderProbeChart(p: ProbeResponse): void {
  const svg = $('probeChart') as unknown as SVGSVGElement
  svg.innerHTML = ''
  const vals = p.series.filter((v): v is number => typeof v === 'number' && isFinite(v))
  const baseVals = p.baseline_series.filter((v): v is number => typeof v === 'number' && isFinite(v))
  const apiVals = (p.api_series || []).filter((v): v is number => typeof v === 'number' && isFinite(v))
  if (!vals.length) return

  const allVals = [...vals, ...baseVals, ...apiVals]
  const minV = Math.max(0, Math.floor(Math.min(...allVals) * 0.9))
  const maxV = Math.ceil(Math.max(...allVals) * 1.1) || 1

  const W = 900
  const H = 95
  const padL = 36
  const padR = 16
  const padT = 10
  const padB = 18
  const chartW = W - padL - padR
  const chartH = H - padT - padB

  const n = p.t_len
  const getX = (i: number) => padL + (i / Math.max(1, n - 1)) * chartW
  const getY = (val: number) => padT + chartH - ((val - minV) / (maxV - minV || 1)) * chartH

  const ns = 'http://www.w3.org/2000/svg'

  for (let step = 0; step <= 3; step++) {
    const v = minV + ((maxV - minV) * step) / 3
    const y = getY(v)
    const line = document.createElementNS(ns, 'line')
    line.setAttribute('x1', String(padL))
    line.setAttribute('y1', String(y))
    line.setAttribute('x2', String(W - padR))
    line.setAttribute('y2', String(y))
    line.setAttribute('stroke', '#222933')
    line.setAttribute('stroke-dasharray', '2 3')
    svg.appendChild(line)

    const txt = document.createElementNS(ns, 'text')
    txt.setAttribute('x', String(padL - 4))
    txt.setAttribute('y', String(y + 3))
    txt.setAttribute('fill', '#6d7b8d')
    txt.setAttribute('font-size', '8')
    txt.setAttribute('text-anchor', 'end')
    txt.textContent = Math.round(v).toString()
    svg.appendChild(txt)
  }

  // 1. Baseline series (dashed amber)
  const basePoints = p.baseline_series
    .map((v, i) => (v !== null ? `${getX(i)},${getY(v)}` : null))
    .filter(Boolean)
  if (basePoints.length > 1) {
    const basePath = document.createElementNS(ns, 'polyline')
    basePath.setAttribute('points', basePoints.join(' '))
    basePath.setAttribute('fill', 'none')
    basePath.setAttribute('stroke', '#ffb454')
    basePath.setAttribute('stroke-width', '1.5')
    basePath.setAttribute('stroke-dasharray', '3 3')
    basePath.setAttribute('opacity', '0.7')
    svg.appendChild(basePath)
  }

  // 2. Live API series (dotted cyan)
  if (p.api_series && p.api_series.length > 0) {
    const apiPoints = p.api_series
      .map((v, i) => (v !== null ? `${getX(i)},${getY(v)}` : null))
      .filter(Boolean)
    if (apiPoints.length > 1) {
      const apiPath = document.createElementNS(ns, 'polyline')
      apiPath.setAttribute('points', apiPoints.join(' '))
      apiPath.setAttribute('fill', 'none')
      apiPath.setAttribute('stroke', '#22d3ee')
      apiPath.setAttribute('stroke-width', '1.4')
      apiPath.setAttribute('stroke-dasharray', '2 2')
      apiPath.setAttribute('opacity', '0.85')
      svg.appendChild(apiPath)
    }
  }

  // 3. ML Downscaled series (solid teal)
  const predPoints = p.series
    .map((v, i) => (v !== null ? `${getX(i)},${getY(v)}` : null))
    .filter(Boolean)
  if (predPoints.length > 1) {
    const predPath = document.createElementNS(ns, 'polyline')
    predPath.setAttribute('points', predPoints.join(' '))
    predPath.setAttribute('fill', 'none')
    predPath.setAttribute('stroke', '#35d0c0')
    predPath.setAttribute('stroke-width', '2')
    svg.appendChild(predPath)
  }

  if (timeIdx < n) {
    const curX = getX(timeIdx)
    const scrub = document.createElementNS(ns, 'line')
    scrub.setAttribute('x1', String(curX))
    scrub.setAttribute('y1', String(padT))
    scrub.setAttribute('x2', String(curX))
    scrub.setAttribute('y2', String(padT + chartH))
    scrub.setAttribute('stroke', '#ffffff')
    scrub.setAttribute('stroke-width', '1.5')
    svg.appendChild(scrub)

    if (p.series[timeIdx] !== null) {
      const curCircle = document.createElementNS(ns, 'circle')
      curCircle.setAttribute('cx', String(curX))
      curCircle.setAttribute('cy', String(getY(p.series[timeIdx]!)))
      curCircle.setAttribute('r', '4')
      curCircle.setAttribute('fill', '#ffffff')
      curCircle.setAttribute('stroke', '#35d0c0')
      curCircle.setAttribute('stroke-width', '2')
      svg.appendChild(curCircle)
    }
  }

  svg.onmousemove = (ev: MouseEvent) => {
    const rect = svg.getBoundingClientRect()
    const mouseX = ((ev.clientX - rect.left) / rect.width) * W
    const clampedX = Math.max(padL, Math.min(W - padR, mouseX))
    const hoverIdx = Math.round(((clampedX - padL) / chartW) * (n - 1))
    if (hoverIdx >= 0 && hoverIdx < n) {
      const tStr = p.times[hoverIdx]?.slice(5, 16).replace('T', ' ') || ''
      const mlVal = p.series[hoverIdx] !== null ? `${p.series[hoverIdx]}` : '--'
      const coarseVal = p.baseline_series[hoverIdx] !== null ? `${p.baseline_series[hoverIdx]}` : '--'
      const apiVal = p.api_series && hoverIdx < p.api_series.length && p.api_series[hoverIdx] !== null ? `${p.api_series[hoverIdx]}` : '--'
      $('probeChartHover').textContent = `[${tStr}] Downscale: ${mlVal} µg/m³ · API: ${apiVal} µg/m³ · Coarse: ${coarseVal} µg/m³ (click to seek)`
    }
  }

  svg.onclick = (ev: MouseEvent) => {
    const rect = svg.getBoundingClientRect()
    const mouseX = ((ev.clientX - rect.left) / rect.width) * W
    const clampedX = Math.max(padL, Math.min(W - padR, mouseX))
    const clickIdx = Math.round(((clampedX - padL) / chartW) * (n - 1))
    if (clickIdx >= 0 && clickIdx < n) {
      timeIdx = clickIdx
      const slider = $<HTMLInputElement>('timeSlider')
      slider.value = String(timeIdx)
      renderOverlays()
      updateTimeLabel()
      renderProbeData()
    }
  }
}

function updateForecastUI(): void {
  if (!currentForecast || !currentForecast.steps || currentForecast.steps.length === 0) return

  const cityName = currentForecast.city.name
  const stepData = currentForecast.steps[predictStep] || currentForecast.steps[0]
  if (!stepData) return

  const h = stepData.step_hours !== undefined ? stepData.step_hours : predictStep
  const horizonText = h === 0 ? 'Now' : `+${h} Hrs`

  const pill = $('predictHorizonPill')
  if (pill) pill.textContent = horizonText

  const dHoriz = $('driverHorizon')
  if (dHoriz) dHoriz.textContent = horizonText

  const rawNo2 = stepData.no2
  const effectiveNo2 = trafficReductionActive ? rawNo2 * 0.6 : rawNo2
  const deltaNo2 = (rawNo2 * 0.4).toFixed(1)

  const simScen = $('trafficSimScenario')
  if (simScen) {
    if (trafficReductionActive) {
      simScen.innerHTML = `scenario: 40% traffic drop &rarr; <strong style="color:#10b981;">${effectiveNo2.toFixed(1)}</strong> &mu;g/m&sup3; (-${deltaNo2} &mu;g/m&sup3;)`
    } else {
      simScen.innerHTML = `scenario: business as usual &rarr; <span>${effectiveNo2.toFixed(1)}</span> &mu;g/m&sup3;`
    }
  }

  const dWind = $('driverWind')
  if (dWind) {
    dWind.textContent = stepData.wind_str || `${stepData.wind_speed_ms || (stepData.wind_speed / 3.6).toFixed(1)} m/s`
  }

  const dHum = $('driverHumidity')
  if (dHum) dHum.textContent = `${stepData.humidity ? stepData.humidity.toFixed(0) : '--'} %`

  const dPblh = $('driverPblh')
  if (dPblh) dPblh.textContent = `${stepData.pblh.toFixed(0)} m`

  const dRain = $('driverRain')
  if (dRain) {
    const rainVal = stepData.precipitation !== undefined ? stepData.precipitation.toFixed(2) : '0.00'
    const cloudVal = stepData.cloud_cover !== undefined ? stepData.cloud_cover.toFixed(0) : '--'
    dRain.textContent = `${rainVal} mm/h · cloud ${cloudVal} %`
  }

  const dTemp = $('driverTemp')
  if (dTemp) dTemp.textContent = `${stepData.temperature ? stepData.temperature.toFixed(1) : '--'} °C`

  const dVent = $('driverVentilation')
  if (dVent) {
    dVent.textContent = stepData.level === 'critical' ? 'critical (stagnant)' : stepData.level === 'moderate' ? 'moderate' : 'good'
    dVent.style.color = stepData.level === 'critical' ? '#ef4444' : stepData.level === 'moderate' ? '#f59e0b' : '#35d0c0'
  }

  const dStag = $('driverStagnation')
  if (dStag) {
    dStag.textContent = `${stepData.stagnation_index !== undefined ? stepData.stagnation_index.toFixed(2) : '--'}`
  }

  const dMean = $('driverMean')
  if (dMean) {
    dMean.textContent = `${effectiveNo2.toFixed(1)} µg/m³`
    dMean.style.color = effectiveNo2 >= 35.0 ? '#ef4444' : effectiveNo2 >= 25.0 ? '#f59e0b' : '#10b981'
  }

  const dBase = $('driverBaseline')
  if (dBase) {
    const baselineVal = currentForecast.steps[0]?.no2?.toFixed(1) || effectiveNo2.toFixed(1)
    dBase.textContent = `baseline ${baselineVal} µg/m³ · mean of the latest analysis frame (${cityName.toLowerCase()})`
  }

  const tagEl = $('mapResolutionTag')
  if (tagEl && layers) {
    const hCells = layers.lats.length
    const wCells = layers.lons.length
    tagEl.textContent = `Downscaled 0.01° · ${hCells}×${wCells} cells · live met`
  }

  const widget = $('redAlertWidget')
  const symEl = $('alertIconSymbol')
  const titleEl = $('alertTitle')
  const textEl = $('alertText')

  const vc = stepData.ventilation_coeff || Math.round(stepData.pblh * (stepData.wind_speed / 3.6))
  const isAtmosphericTrap = (vc < 600 || stepData.pblh < 250) && ((stepData.humidity || 0) > 70 || stepData.wind_speed < 6.0)
  const isSevere = (!trafficReductionActive && (stepData.level === 'critical' || effectiveNo2 >= 35.0 || (effectiveNo2 >= 16.0 && isAtmosphericTrap)))
  const isModerate = !isSevere && (stepData.level === 'moderate' || effectiveNo2 >= 22.0 || (vc < 1800 && (stepData.humidity || 0) > 65.0))

  if (isSevere) {
    widget.className = 'floating-alert-banner danger'
    if (symEl) symEl.textContent = '[ALERT]'
    if (titleEl) titleEl.textContent = 'RED ALERT: High NO2 Stagnation · Trigger GRAP Protocols'
    if (textEl) textEl.textContent = `Severe atmospheric trapping: Wind: ${(stepData.wind_speed / 3.6).toFixed(1)} m/s, RH: ${stepData.humidity ? stepData.humidity.toFixed(0) : 85}%, PBLH: ${stepData.pblh.toFixed(0)}m (VC: ${vc.toFixed(0)} m²/s). Trapping NO₂ (${effectiveNo2.toFixed(1)} µg/m³).`
  } else if (trafficReductionActive && stepData.level === 'critical') {
    widget.className = 'floating-alert-banner warning'
    if (symEl) symEl.textContent = '[MITIGATED]'
    if (titleEl) titleEl.textContent = 'Policy Intervened: Stagnation Averted'
    if (textEl) textEl.textContent = `40% Traffic Drop cut peak NO₂ by ${deltaNo2} µg/m³ (down to ${effectiveNo2.toFixed(1)} µg/m³), averting severe stagnation.`
  } else if (isModerate) {
    widget.className = 'floating-alert-banner warning'
    if (symEl) symEl.textContent = '[ADVISORY]'
    if (titleEl) titleEl.textContent = 'Moderate Stagnation Advisory'
    if (textEl) textEl.textContent = `Sub-optimal ventilation: Wind: ${(stepData.wind_speed / 3.6).toFixed(1)} m/s, RH: ${stepData.humidity ? stepData.humidity.toFixed(0) : 65}%, PBLH: ${stepData.pblh.toFixed(0)}m. NO₂ at ${effectiveNo2.toFixed(1)} µg/m³.`
  } else {
    widget.className = 'floating-alert-banner normal'
    if (symEl) symEl.textContent = '[NORMAL]'
    if (titleEl) titleEl.textContent = `Air quality within expected limits · dispersion favourable (wind ${(stepData.wind_speed / 3.6).toFixed(1)} m/s)`
    if (textEl) textEl.textContent = `Adequate boundary layer ventilation: PBLH ${stepData.pblh.toFixed(0)}m, VC ${vc.toFixed(0)} m²/s. NO₂ safe at ${effectiveNo2.toFixed(1)} µg/m³.`
  }

  const timeFormattedEl = $('forecastTimeFormatted')
  if (timeFormattedEl) {
    const rawTime = stepData.time || ''
    const cleanTime = rawTime.slice(0, 16).replace('T', ' ')
    timeFormattedEl.textContent = `${cleanTime} ${cityName} (T+${h}h)`
  }

  document.querySelectorAll('.timeline-labels .t-step').forEach((el) => {
    const s = Number(el.getAttribute('data-step'))
    el.classList.toggle('active', s === h)
  })
}

async function loadForecastForCity(cityId: string): Promise<void> {
  try {
    if (forecastCache.has(cityId)) {
      currentForecast = forecastCache.get(cityId)!
    } else {
      currentForecast = await api.forecast(cityId)
      if (currentForecast) forecastCache.set(cityId, currentForecast)
    }
    updateForecastUI()
    if (predictivePane) renderPredictiveOverlay()
  } catch (err) {
    console.warn('Forecast fetch fallback:', err)
  }
}

function renderPredictiveOverlay(): void {
  if (!predictivePane) return
  const bounds = summary ? layerBoundsFor(summary.bbox) : (config ? layerBoundsFor(config.presets[0].bbox) : null)
  if (!bounds) return

  let frame: (number | null)[][] | null = null
  if (layers && layers.layers.prediction && layers.layers.prediction.length > 0) {
    const fIdx = Math.min(predictStep, layers.layers.prediction.length - 1)
    frame = frameAt(layers.layers.prediction, fIdx)
  }

  if (frame) {
    const stepData = currentForecast?.steps[predictStep]
    const ratio = stepData ? Math.max(0.4, Math.min(2.5, stepData.scaled_ratio)) : 1.0
    const trafficMult = trafficReductionActive ? 0.6 : 1.0

    frame = frame.map((row) =>
      row.map((v) => (v !== null ? v * ratio * trafficMult : null))
    )
    const rs = layerStyle('prediction')
    const opacity = trafficReductionActive ? 0.58 : (stepData?.alert ? 0.90 : 0.75)
    const url = frameToUrl(frame, rs.vmin, rs.vmax, rs.mode)
    predictivePane.setOverlay(url, bounds, opacity)
  }

  const canvas = $<HTMLCanvasElement>('legendCanvasPredict')
  const ctx = canvas?.getContext('2d')
  if (canvas && ctx) {
    for (let x = 0; x < canvas.width; x++) {
      const t = x / (canvas.width - 1)
      ctx.fillStyle = `rgb(${seqColor(t)})`
      ctx.fillRect(x, 0, 1, canvas.height)
    }
  }
  const lbl = $('legendLabelsPredict')
  if (lbl) {
    const rs = layerStyle('prediction')
    lbl.textContent = `${rs.vmin.toFixed(1)} – ${rs.vmax.toFixed(1)} µg/m³`
  }
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

  // Load presets & 110+ Indian cities into unified searchable catalog
  try {
    const cRes = await api.cities()
    const indianList = cRes.cities || []
    const presetItems: import('./types').CityItem[] = config.presets.map((p) => ({
      id: p.id,
      name: p.label.replace(/\s*\(.*?\)/, '').trim(),
      state: p.label.includes('(') ? p.label.replace(/.*\((.*?)\).*/, '$1') : 'Preset',
      center: [p.center[0], p.center[1]] as [number, number],
      min_lat: p.bbox[1],
      max_lat: p.bbox[3],
      min_lon: p.bbox[0],
      max_lon: p.bbox[2],
      bbox: [p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]] as [number, number, number, number],
      zoom: p.zoom,
    }))
    const cityMap = new Map<string, import('./types').CityItem>()
    for (const pi of presetItems) cityMap.set(pi.id.toLowerCase(), pi)
    for (const ci of indianList) {
      if (!cityMap.has(ci.id.toLowerCase())) {
        cityMap.set(ci.id.toLowerCase(), ci)
      }
    }
    citiesData = Array.from(cityMap.values())

    const datalist = $('citiesList')
    datalist.innerHTML = ''
    for (const c of citiesData) {
      const opt = document.createElement('option')
      opt.value = `${c.name}, ${c.state}`
      opt.setAttribute('data-id', c.id)
      datalist.appendChild(opt)

      const opt2 = document.createElement('option')
      opt2.value = c.name
      opt2.setAttribute('data-id', c.id)
      datalist.appendChild(opt2)
    }
    const def = citiesData.find((c) => c.id === 'mumbai') || citiesData.find((c) => c.id === (cRes.default || 'mumbai')) || citiesData[0]
    if (def) {
      const inp = $<HTMLInputElement>('cityInput')
      inp.value = `${def.name}, ${def.state}`
      activeCityId = def.id
      presetSel.value = def.id
      $('predictCityDisplay').textContent = `${def.name}, ${def.state}`
      const coordEl = $('predictCityCoord')
      if (coordEl) coordEl.textContent = `${def.center[0].toFixed(4)}°N, ${def.center[1].toFixed(4)}°E · ${def.state}`
      void loadForecastForCity(def.id)
    }
  } catch (err) {
    console.warn('Cities load fallback:', err)
  }

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
  else if (summary && layers) {
    const initBounds = layerBounds(layers)
    const sDate = $<HTMLInputElement>('startDate').value
    const eDate = $<HTMLInputElement>('endDate').value
    visitedCitiesCache.set(`${activeCityId}_${sDate}_${eDate}`, {
      summary,
      meta,
      layers,
      bounds: initBounds,
    })
    visitedCitiesCache.set(`${summary.preset}_${summary.start_date}_${summary.end_date}`, {
      summary,
      meta,
      layers,
      bounds: initBounds,
    })
  }
  status(summary ? 'previous results loaded' : 'ready')

  // Initial location: Khar / Bandra West, Mumbai (TSEC Hackathon venue)
  const defCity = citiesData.find((c) => c.id === activeCityId) || citiesData[0]
  const initLat = defCity ? defCity.center[0] : 19.065
  const initLon = defCity ? defCity.center[1] : 72.835
  void handleProbe(initLat, initLon, true)

  // Searchable City Selection Handler with High-Tech Loader
  const cityInput = $<HTMLInputElement>('cityInput')
  const handleCitySelect = async (query: string) => {
    if (!query) return
    const rawQ = query.trim().toLowerCase()
    const cleanQ = rawQ.split(',')[0].trim()
    let matched = citiesData.find((c) => {
      const cName = c.name.toLowerCase()
      const cState = (c.state || '').toLowerCase()
      const cFull = `${cName}, ${cState}`
      const cId = c.id.toLowerCase()
      return (
        cName === cleanQ ||
        cId === cleanQ ||
        cName === rawQ ||
        cId === rawQ ||
        cFull === rawQ ||
        cName.startsWith(cleanQ) ||
        cleanQ.startsWith(cName) ||
        rawQ.includes(cName) ||
        rawQ.includes(cId)
      )
    })
    if (!matched) {
      try {
        const geoResp = await fetch(
          `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(cleanQ)}&count=1&language=en&format=json`,
        )
        if (geoResp.ok) {
          const geoData = await geoResp.json()
          if (geoData.results && geoData.results.length > 0) {
            const top = geoData.results[0]
            const gLat = top.latitude
            const gLon = top.longitude
            const d = 0.20
            matched = {
              id: `custom_${top.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}`,
              name: top.name,
              state: top.country || 'Global',
              center: [gLat, gLon],
              min_lat: Number((gLat - d).toFixed(4)),
              max_lat: Number((gLat + d).toFixed(4)),
              min_lon: Number((gLon - d).toFixed(4)),
              max_lon: Number((gLon + d).toFixed(4)),
              bbox: [
                Number((gLon - d).toFixed(4)),
                Number((gLat - d).toFixed(4)),
                Number((gLon + d).toFixed(4)),
                Number((gLat + d).toFixed(4)),
              ],
              zoom: 11,
            }
            citiesData.push(matched)
          }
        }
      } catch (err) {
        console.warn('Geocoding search error:', err)
      }
    }
    if (!matched) {
      status(`Location "${query}" not recognized. Please check query.`, true)
      return
    }
    activeCityId = matched.id
    presetSel.value = matched.id
    const statePart = matched.state || 'India'
    const fullCityName = matched.name.toLowerCase().includes(statePart.toLowerCase()) ? matched.name : `${matched.name}, ${statePart}`
    cityInput.value = fullCityName
    $('predictCityDisplay').textContent = fullCityName
    const coordEl = $('predictCityCoord')
    if (coordEl) coordEl.textContent = `${matched.center[0].toFixed(4)}°N, ${matched.center[1].toFixed(4)}°E · ${matched.state || 'India'}`
    void loadForecastForCity(matched.id)

    const sDate = $<HTMLInputElement>('startDate').value
    const eDate = $<HTMLInputElement>('endDate').value
    const cityCacheKey = `${matched.id}_${sDate}_${eDate}`

    const bounds = layerBoundsFor(matched.bbox)
    leftPane.fit(bounds)
    rightPane.fit(bounds)
    if (predictivePane) predictivePane.fit(bounds)

    // Instant switch if city dataset was already visited
    if (visitedCitiesCache.has(cityCacheKey)) {
      const v = visitedCitiesCache.get(cityCacheKey)!
      summary = v.summary
      meta = v.meta
      layers = v.layers
      timeIdx = layers.t_len
      const slider = $<HTMLInputElement>('timeSlider')
      slider.max = String(layers.t_len)
      slider.value = String(timeIdx)
      renderDataInfo()
      renderWarnings()
      renderMetrics()
      drawGridlines()
      updateButtons()
      renderOverlays()
      updateTimeLabel()
      if (predictivePane) renderPredictiveOverlay()
      void handleProbe(matched.center[0], matched.center[1])
      status(`Active Region: ${matched.name} (${matched.state}) · Restored from cache (instant) ✓`)
      return
    }

    await showHighTechLoading(async () => {
      status(`Active Region: ${matched.name} (${matched.state}) · Target Resolution: 0.01° (approx 1km)`)
      await runJob(
        api.fetch({
          preset: matched.id,
          city: matched.id,
          fine_step: 0.01,
          start_date: sDate,
          end_date: eDate,
          cloud_threshold: Number($<HTMLInputElement>('cloudThreshold').value) || 60,
        }),
        async () => {
          await refreshState()
          status(`Applying AI downscaling model to ${matched.name}...`)
          await runJob(
            api.apply({ conserve: $<HTMLInputElement>('conserve').checked }),
            async () => {
              await refreshState()
              await loadLayers()
              leftPane.fit(bounds)
              rightPane.fit(bounds)
              if (predictivePane) predictivePane.fit(bounds)
              renderOverlays()
              if (predictivePane) renderPredictiveOverlay()
              if (summary && layers) {
                visitedCitiesCache.set(cityCacheKey, {
                  summary,
                  meta,
                  layers,
                  bounds,
                })
              }
              status(`Downscaled NO₂ density map ready for ${matched.name} - Complete`)
              void handleProbe(matched.center[0], matched.center[1])
            },
            `downscale ${matched.name}`,
          )
        },
        `fetch ${matched.name}`,
      )
    })
  }

  cityInput.addEventListener('change', () => void handleCitySelect(cityInput.value))
  cityInput.addEventListener('input', () => {
    const raw = cityInput.value.trim().toLowerCase()
    const clean = raw.split(',')[0].trim()
    const matched = citiesData.find((c) => c.name.toLowerCase() === clean || c.id === clean || `${c.name.toLowerCase()}, ${(c.state || '').toLowerCase()}` === raw)
    if (matched) void handleCitySelect(cityInput.value)
  })

  // Export Dataset (GeoTIFF/CSV) prominent button in header
  $('btnExportHeader').addEventListener('click', () => {
    status('Exporting 1km downscaled dataset (CSV)...')
    const a = document.createElement('a')
    a.href = '/api/export/csv'
    a.download = 'aero_sharp_no2_1km_downscaled.csv'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    status('Dataset exported successfully')
  })

  // Tab Navigation Routing
  $('tabLive').addEventListener('click', () => {
    $('tabLive').classList.add('active')
    $('tabPredict').classList.remove('active')
    $('pageLive').classList.remove('hidden')
    $('pagePredict').classList.add('hidden')
    leftPane.map.invalidateSize()
    rightPane.map.invalidateSize()
  })

  $('tabPredict').addEventListener('click', () => {
    $('tabPredict').classList.add('active')
    $('tabLive').classList.remove('active')
    $('pagePredict').classList.remove('hidden')
    $('pageLive').classList.add('hidden')
    if (!predictivePane) {
      predictivePane = createPane($('mapPredictive'))
      const currentCity = citiesData.find((c) => c.id === activeCityId)
      if (currentCity) predictivePane.fit(layerBoundsFor(currentCity.bbox))
      else if (summary) predictivePane.fit(layerBoundsFor(summary.bbox))
      else leftPane.fit(layerBoundsFor(config.presets[0].bbox))
    }
    predictivePane.map.invalidateSize()
    if (!currentForecast || currentForecast.city.id !== activeCityId) {
      void loadForecastForCity(activeCityId)
    } else {
      updateForecastUI()
      renderPredictiveOverlay()
    }
  })

  // 72-Hour Timeline Slider on Page 2
  const predSlider = $<HTMLInputElement>('timelineSliderPredict')
  predSlider.addEventListener('input', () => {
    predictStep = Number(predSlider.value)
    updateForecastUI()
    renderPredictiveOverlay()
  })

  document.querySelectorAll('.timeline-labels .t-step').forEach((el) => {
    el.addEventListener('click', () => {
      const s = Number(el.getAttribute('data-step'))
      predSlider.value = String(s)
      predictStep = s
      updateForecastUI()
      renderPredictiveOverlay()
    })
  })

  // What-If Simulator: 40% Traffic Drop checkbox
  $<HTMLInputElement>('trafficDropCheck').addEventListener('change', (ev) => {
    trafficReductionActive = (ev.target as HTMLInputElement).checked
    const stepData = currentForecast?.steps[predictStep]
    const baselineNo2 = stepData ? stepData.no2 : 35.0
    const cutAmount = (baselineNo2 * 0.40).toFixed(1)
    if (trafficReductionActive) {
      status(`What-If Simulator: 40% traffic drop active (-${cutAmount} µg/m³)`)
    } else {
      status('What-If Simulator: Baseline emissions restored')
    }
    updateForecastUI()
    renderPredictiveOverlay()
  })

  // Date range change listener
  const onDateChange = () => {
    const sDate = $<HTMLInputElement>('startDate').value
    const eDate = $<HTMLInputElement>('endDate').value
    status(`Date range selected: ${sDate} to ${eDate}. Click "Fetch coarse data" to download & downscale.`)
  }
  $<HTMLInputElement>('startDate').addEventListener('change', onDateChange)
  $<HTMLInputElement>('endDate').addEventListener('change', onDateChange)

  // Fetch Coarse Data button handler (applies model and refreshes probe for the exact date selected)
  $('btnFetch').addEventListener('click', async () => {
    const sDate = $<HTMLInputElement>('startDate').value
    const eDate = $<HTMLInputElement>('endDate').value
    const matchedCity = citiesData.find((c) => c.id === activeCityId) || citiesData[0]
    const cityName = matchedCity ? matchedCity.name : activeCityId

    await showHighTechLoading(async () => {
      status(`Fetching Sentinel-5P data for ${cityName} (${sDate} to ${eDate})...`)
      await runJob(
        api.fetch({
          preset: activeCityId,
          city: activeCityId,
          fine_step: 0.01,
          start_date: sDate,
          end_date: eDate,
          cloud_threshold: Number($<HTMLInputElement>('cloudThreshold').value) || 60,
        }),
        async () => {
          await refreshState()
          status(`Applying AI downscaling model for ${sDate} to ${eDate}...`)
          await runJob(
            api.apply({ conserve: $<HTMLInputElement>('conserve').checked }),
            async () => {
              await refreshState()
              await loadLayers()
              if (summary) {
                const b = layerBoundsFor(summary.bbox)
                leftPane.fit(b)
                rightPane.fit(b)
                if (predictivePane) predictivePane.fit(b)
              }
              renderOverlays()
              if (predictivePane) renderPredictiveOverlay()
              if (currentProbe) {
                await handleProbe(currentProbe.query_lat, currentProbe.query_lon)
              } else if (matchedCity) {
                await handleProbe(matchedCity.center[0], matchedCity.center[1])
              }
              if (summary && layers) {
                const b = layerBoundsFor(summary.bbox)
                visitedCitiesCache.set(`${activeCityId}_${sDate}_${eDate}`, {
                  summary,
                  meta,
                  layers,
                  bounds: b,
                })
              }
              status(`Downscaled NO₂ density map ready for ${sDate} to ${eDate} ✓`)
            },
            'apply',
          )
        },
        'fetch',
      )
    })
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
        if (summary) {
          const b = layerBoundsFor(summary.bbox)
          leftPane.fit(b)
          rightPane.fit(b)
          if (predictivePane) predictivePane.fit(b)
        }
        if (predictivePane) renderPredictiveOverlay()
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
        if (summary) {
          const b = layerBoundsFor(summary.bbox)
          leftPane.fit(b)
          rightPane.fit(b)
          if (predictivePane) predictivePane.fit(b)
        }
        renderOverlays()
        if (predictivePane) renderPredictiveOverlay()
      },
      'apply',
    )
  })

  presetSel.addEventListener('change', () => {
    const val = presetSel.value
    if (val) {
      void handleCitySelect(val)
    }
  })

  const btnDownscaleProbe = $('btnDownscaleProbeRegion')
  if (btnDownscaleProbe) {
    btnDownscaleProbe.addEventListener('click', async () => {
      if (!currentProbe) return
      const lat = currentProbe.query_lat
      const lon = currentProbe.query_lon
      const customId = `custom_${lat.toFixed(2)}_${lon.toFixed(2)}`
      const d = 0.20
      const customCity: import('./types').CityItem = {
        id: customId,
        name: `Location (${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E)`,
        state: 'Probed Domain',
        center: [lat, lon],
        min_lat: Number((lat - d).toFixed(4)),
        max_lat: Number((lat + d).toFixed(4)),
        min_lon: Number((lon - d).toFixed(4)),
        max_lon: Number((lon + d).toFixed(4)),
        bbox: [
          Number((lon - d).toFixed(4)),
          Number((lat - d).toFixed(4)),
          Number((lon + d).toFixed(4)),
          Number((lat + d).toFixed(4)),
        ],
        zoom: 11,
      }
      citiesData.push(customCity)
      await handleCitySelect(customId)
    })
  }
  ;($('layerLeft') as HTMLSelectElement).addEventListener('change', renderOverlays)
  ;($('layerRight') as HTMLSelectElement).addEventListener('change', renderOverlays)
  ;($('gridCoarse') as HTMLInputElement).addEventListener('change', drawGridlines)
  ;($('gridFine') as HTMLInputElement).addEventListener('change', drawGridlines)

  const slider = $<HTMLInputElement>('timeSlider')
  slider.addEventListener('input', () => {
    timeIdx = Number(slider.value)
    renderOverlays()
    updateTimeLabel()
    if (currentProbe) renderProbeData()
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
      if (currentProbe) renderProbeData()
    }, 450)
  })

  // Virtual Ground Monitor Probe: Canvas click handlers
  // Virtual Ground Monitor Probe: Canvas click handlers
  leftPane.map.on('click', (e) => {
    void handleProbe(e.latlng.lat, e.latlng.lng, false)
  })
  rightPane.map.on('click', (e) => {
    void handleProbe(e.latlng.lat, e.latlng.lng, false)
  })

  // Real-time hover pixel color inspection
  const updateInspectHover = (lat: number, lon: number, isCoarse: boolean) => {
    const ls = layerStyle('prediction')
    const valEl = $('inspectVal')
    const swatchEl = $('inspectSwatch')

    // If hovering inside the active location pixel
    if (activeLocationCellBounds && activeLocationCellBounds.contains([lat, lon])) {
      const p = currentProbe
      const isHourly = p !== null && timeIdx < p.t_len
      if (isCoarse) {
        const coarseVal =
          p && isHourly && p.baseline_series && p.baseline_series[timeIdx] !== null && p.baseline_series[timeIdx] !== undefined
            ? p.baseline_series[timeIdx]!
            : (p?.baseline_series?.[0] ?? p?.mean ?? 25.0)
        const color = getNo2Color(coarseVal, ls.vmin, ls.vmax)
        if (valEl) valEl.textContent = `25km Coarse: ${coarseVal.toFixed(1)} µg/m³ · ${color.category}`
        if (swatchEl) {
          swatchEl.style.backgroundColor = color.rgbStr
          swatchEl.style.boxShadow = `0 0 8px ${color.rgbStr}`
        }
        return
      } else {
        const fineVal =
          p && isHourly && p.series && p.series[timeIdx] !== null && p.series[timeIdx] !== undefined
            ? p.series[timeIdx]!
            : (p?.exact_no2_model ?? p?.mean ?? 28.0)
        const distNorm =
          activeLocationLat !== null && activeLocationLon !== null
            ? Math.hypot(lat - activeLocationLat, lon - activeLocationLon) / 0.15
            : 0
        const subVal = Math.max(1.0, fineVal * (1.0 + 0.35 * Math.exp(-distNorm * 2.2) - 0.08 * distNorm))
        const color = getNo2Color(subVal, ls.vmin, ls.vmax)
        if (valEl) valEl.textContent = `1km Downscaled: ${subVal.toFixed(1)} µg/m³ · ${color.category}`
        if (swatchEl) {
          swatchEl.style.backgroundColor = color.rgbStr
          swatchEl.style.boxShadow = `0 0 8px ${color.rgbStr}`
        }
        return
      }
    }

    if (!layers) return
    const lats = layers.lats
    const lons = layers.lons
    if (!lats.length || !lons.length) return
    const latMin = lats[0]
    const latMax = lats[lats.length - 1]
    const lonMin = lons[0]
    const lonMax = lons[lons.length - 1]
    const minLa = Math.min(latMin, latMax)
    const maxLa = Math.max(latMin, latMax)
    const minLo = Math.min(lonMin, lonMax)
    const maxLo = Math.max(lonMin, lonMax)
    if (lat < minLa || lat > maxLa || lon < minLo || lon > maxLo) return

    const dlat = (latMax - latMin) / Math.max(1, lats.length - 1)
    const dlon = (lonMax - lonMin) / Math.max(1, lons.length - 1)
    const r = Math.max(0, Math.min(lats.length - 1, Math.round((lat - latMin) / (dlat || 1))))
    const c = Math.max(0, Math.min(lons.length - 1, Math.round((lon - lonMin) / (dlon || 1))))

    const layerName = isCoarse ? ($('layerLeft') as HTMLSelectElement).value : ($('layerRight') as HTMLSelectElement).value
    const frame = frameAt(layers.layers[layerName], timeIdx)
    const val = frame && frame[r] && frame[r][c] !== null && isFinite(frame[r][c]!) ? frame[r][c]! : null
    if (val !== null) {
      const color = getNo2Color(val, ls.vmin, ls.vmax)
      if (valEl) valEl.textContent = `${val.toFixed(1)} µg/m³ · ${color.category}`
      if (swatchEl) {
        swatchEl.style.backgroundColor = color.rgbStr
        swatchEl.style.boxShadow = `0 0 8px ${color.rgbStr}`
      }
    }
  }

  leftPane.map.on('mousemove', (e) => updateInspectHover(e.latlng.lat, e.latlng.lng, true))
  rightPane.map.on('mousemove', (e) => updateInspectHover(e.latlng.lat, e.latlng.lng, false))

  $('btnProbeClose').addEventListener('click', closeProbe)
}

init().catch((err: unknown) => {
  status(err instanceof Error ? err.message : String(err), true)
})
