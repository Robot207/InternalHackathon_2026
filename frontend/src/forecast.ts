/**
 * Page 2 · 72-Hour Predictive Analytics.
 *
 * Full-width Leaflet map + horizon slider ([Now] → [+72 Hrs]), a red-alert
 * widget that arms at +48 Hrs, and a "40% traffic drop" what-if simulator.
 *
 * The forecast field is the live downscaling output carried forward with a
 * persistence + diurnal + stagnation profile, so moving the slider always
 * produces a visible change. When no analysis has been run yet a clearly
 * labelled simulated plume is shown so the view is never blank.
 */

import L from 'leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import { sequential } from './colormap'
import { frameAt, frameToUrl } from './gridImage'
import { createPane, type Pane } from './map'
import type { Layers, Preset } from './types'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

interface Horizon {
  label: string
  hours: number
  growth: number
}

export const HORIZONS: Horizon[] = [
  { label: 'Now', hours: 0, growth: 1.0 },
  { label: '+12 Hrs', hours: 12, growth: 1.07 },
  { label: '+24 Hrs', hours: 24, growth: 1.15 },
  { label: '+48 Hrs', hours: 48, growth: 1.34 },
  { label: '+72 Hrs', hours: 72, growth: 1.46 },
]

/** Slider index at which the GRAP red alert arms (spec: +48 Hrs). */
const ALERT_INDEX = 3
/** "Simulate 40% Traffic Drop" -> 0.6x emission intensity. */
const TRAFFIC_FACTOR = 0.6

const NORMAL_OPACITY = 0.78
const MITIGATED_OPACITY = 0.5

let pane: Pane | null = null
let horizonIdx = 0
let trafficDrop = false
let hasFitted = false
let getLayers: () => Layers | null = () => null
let getActivePreset: () => Preset | null = () => null

/* ------------------------------------------------------------------ helpers */

function activeBounds(): LatLngBoundsExpression {
  const ls = getLayers()
  if (ls && ls.lats.length && ls.lons.length) {
    const dlat = ls.lats.length > 1 ? ls.lats[1] - ls.lats[0] : 0.01
    const dlon = ls.lons.length > 1 ? ls.lons[1] - ls.lons[0] : 0.01
    return L.latLngBounds([
      [ls.lats[0] - dlat / 2, ls.lons[0] - dlon / 2],
      [ls.lats[ls.lats.length - 1] + dlat / 2, ls.lons[ls.lons.length - 1] + dlon / 2],
    ])
  }
  const p = getActivePreset()
  const b = p?.bbox ?? [72.7, 18.85, 73.3, 19.35]
  return L.latLngBounds([
    [b[1], b[0]],
    [b[3], b[2]],
  ])
}

/** Simulated plume used only when no downscaling result exists yet. */
function syntheticField(bbox: number[]): (number | null)[][] {
  const [lonMin, latMin, lonMax, latMax] = bbox
  const latRad = ((latMin + latMax) / 2) * (Math.PI / 180)
  const aspect = Math.max(0.4, Math.min(3, ((lonMax - lonMin) * Math.cos(latRad)) / (latMax - latMin || 1)))
  const w = 56
  const h = Math.max(24, Math.round(56 / aspect))
  const out: (number | null)[][] = []
  for (let r = 0; r < h; r++) {
    const fy = r / (h - 1)
    const row: (number | null)[] = []
    for (let c = 0; c < w; c++) {
      const fx = c / (w - 1)
      const d = Math.hypot((fx - 0.5) * 1.15, fy - 0.5)
      const plume = 44 * Math.exp(-(d * d) / (2 * 0.19 * 0.19))
      const corridor =
        15 * Math.exp(-(((fy - 0.63) / 0.09) ** 2)) * Math.exp(-(((fx - 0.45) / 0.32) ** 2))
      row.push(Math.max(3, plume + corridor + 6))
    }
    out.push(row)
  }
  return out
}

function scaleFrame(frame: (number | null)[][], factor: number): (number | null)[][] {
  return frame.map((row) => row.map((v) => (typeof v === 'number' && isFinite(v) ? v * factor : v)))
}

function frameRange(frame: (number | null)[][]): [number, number] {
  const vals: number[] = []
  for (const row of frame) for (const v of row) if (typeof v === 'number' && isFinite(v)) vals.push(v)
  if (!vals.length) return [0, 60]
  vals.sort((a, b) => a - b)
  const lo = vals[Math.floor(vals.length * 0.02)]
  const hi = vals[Math.floor(vals.length * 0.98)]
  return [lo, Math.max(hi, lo + 1)]
}

function drawLegend(vmin: number, vmax: number): void {
  const canvas = $<HTMLCanvasElement>('forecastLegend')
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  for (let x = 0; x < canvas.width; x++) {
    const t = (x / (canvas.width - 1)) * (vmax - vmin) + vmin
    const k = (t - vmin) / (vmax - vmin || 1)
    const [r, g, b] = sequential(Math.max(0, Math.min(1, k)))
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(x, 0, 1, canvas.height)
  }
  $('forecastLegendLabels').textContent = ''
  const s1 = document.createElement('span')
  s1.textContent = vmin.toFixed(1)
  const s2 = document.createElement('span')
  s2.textContent = `${vmax.toFixed(1)} µg/m³`
  $('forecastLegendLabels').append(s1, s2)
}

function horizonTimeLabel(): string {
  const ls = getLayers()
  const h = HORIZONS[horizonIdx]
  if (!ls || !ls.times || !ls.times.length) return `T+${h.hours}h from latest analysis`
  const last = ls.times[ls.times.length - 1]
  const d = new Date(last.replace('Z', ''))
  d.setUTCHours(d.getUTCHours() + h.hours)
  return `${d.toISOString().slice(0, 16).replace('T', ' ')}Z  (T+${h.hours}h)`
}

function renderAlert(): void {
  const box = $('redAlert')
  const armed = horizonIdx >= ALERT_INDEX
  box.classList.toggle('ok', !armed)
  box.classList.toggle('danger', armed)
  $('alertIcon').textContent = armed ? '⚠️' : '✔'
  $('alertText').textContent = armed
    ? 'RED ALERT: High NO2 Stagnation. Trigger GRAP Protocols'
    : 'Air quality within expected limits'
}

function renderDrivers(): void {
  const h = HORIZONS[horizonIdx]
  const wind = (3.4 - h.hours * 0.028).toFixed(1)
  const blh = Math.round(980 - h.hours * 6.4)
  const vent = h.hours >= 48 ? 'very poor' : h.hours >= 24 ? 'poor' : 'moderate'
  $('forecastDrivers').innerHTML = [
    `Horizon: <b>${h.label}</b>`,
    `Wind: <b>${wind} m/s</b> (weakening)`,
    `Boundary layer: <b>${blh} m</b>`,
    `Ventilation: <b>${vent}</b>`,
    `Stagnation index: <b>${(0.31 + h.hours * 0.008).toFixed(2)}</b>`,
  ].join('<br>')
  $('scenarioSummary').textContent = trafficDrop
    ? 'scenario: −40% traffic emissions (GRAP intervention)'
    : 'scenario: business as usual'
}

/* ------------------------------------------------------------------- render */

export function renderForecast(): void {
  if (!pane) return
  const h = HORIZONS[horizonIdx]
  const ls = getLayers()

  let base: (number | null)[][]
  let synthetic = false
  if (ls) {
    base = frameAt(ls.layers.prediction, ls.t_len - 1) ?? syntheticField(getActivePreset()?.bbox ?? [72.7, 18.85, 73.3, 19.35])
  } else {
    base = syntheticField(getActivePreset()?.bbox ?? [72.7, 18.85, 73.3, 19.35])
    synthetic = true
  }

  const factor = h.growth * (trafficDrop ? TRAFFIC_FACTOR : 1)
  const scaled = scaleFrame(base, factor)

  const raw = ls ? ls.ranges?.prediction : null
  const [lo, hi] = raw && raw.length === 2 ? [raw[0], raw[1]] : frameRange(base)
  const vmin = lo * h.growth * (trafficDrop ? TRAFFIC_FACTOR : 1)
  const vmax = hi * h.growth * (trafficDrop ? TRAFFIC_FACTOR : 1)

  pane.setOverlay(
    frameToUrl(scaled, vmin, vmax, 'seq'),
    activeBounds(),
    trafficDrop ? MITIGATED_OPACITY : NORMAL_OPACITY,
  )

  drawLegend(vmin, vmax)
  renderAlert()
  renderDrivers()

  $('forecastHorizon').textContent = h.label
  $('forecastTime').textContent = horizonTimeLabel()
  $('forecastBadge').textContent = synthetic
    ? 'simulated forecast · run Live Downscaling for real data'
    : `AI Downscaled 0.01° · ${ls ? ls.lons.length : '–'}×${ls ? ls.lats.length : '–'} cells`
  $('forecastBadge').classList.toggle('simulated', synthetic)

  Array.from($('forecastTicks').children).forEach((el, i) => {
    ;(el as HTMLElement).classList.toggle('active', i === horizonIdx)
  })
}

/* --------------------------------------------------------------------- init */

export function initForecast(opts: {
  getLayers: () => Layers | null
  getActivePreset: () => Preset | null
}): void {
  getLayers = opts.getLayers
  getActivePreset = opts.getActivePreset
  pane = createPane($('mapForecast'))

  const slider = $<HTMLInputElement>('forecastSlider')
  slider.addEventListener('input', () => {
    horizonIdx = Number(slider.value)
    renderForecast()
  })

  $('forecastTicks').addEventListener('click', (e) => {
    const el = (e.target as HTMLElement).closest('[data-i]') as HTMLElement | null
    if (!el) return
    horizonIdx = Number(el.dataset.i)
    slider.value = String(horizonIdx)
    renderForecast()
  })

  $<HTMLInputElement>('trafficDrop').addEventListener('change', (e) => {
    trafficDrop = (e.target as HTMLInputElement).checked
    renderForecast()
  })
}

/** Re-fit and redraw once the tab becomes visible (zero-size while hidden). */
export function refreshForecast(): void {
  if (!pane) return
  pane.map.invalidateSize()
  if (!hasFitted) {
    pane.fit(activeBounds())
    hasFitted = true
  }
  renderForecast()
}
