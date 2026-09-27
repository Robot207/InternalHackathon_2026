/**
 * Page 2 · 72-Hour Predictive Analytics.
 *
 * Full-width Leaflet map + horizon slider ([Now] → [+72 Hrs]), a red-alert
 * widget that arms when the forecast says so, and a "40% traffic drop"
 * what-if simulator.
 *
 * The projection is no longer a fixed set of growth numbers: `GET /api/forecast`
 * returns 73 hourly steps of live Open-Meteo meteorology for the active bbox
 * (wind, humidity, boundary-layer height, rain, cloud, temperature) together
 * with an emission x dispersion projection and a stagnation index, so the
 * drivers, the alert and the scaling of the field all move with the weather.
 * The fixed profile survives only as a labelled fallback when the API is
 * unreachable, and when no downscaling result exists yet a clearly labelled
 * simulated plume is shown so the view is never blank.
 */

import L from 'leaflet'
import type { LatLngBounds, LatLngBoundsExpression } from 'leaflet'
import { api } from './api'
import { sequential } from './colormap'
import { frameAt, frameToUrl } from './gridImage'
import { createPane, type Pane } from './map'
import type { Forecast, Layers, Preset } from './types'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

interface Horizon {
  label: string
  hours: number
  growth: number
}

/** Slider stops. `growth` is only the offline fallback when /api/forecast is down. */
export const HORIZONS: Horizon[] = [
  { label: 'Now', hours: 0, growth: 1.0 },
  { label: '+12 Hrs', hours: 12, growth: 1.07 },
  { label: '+24 Hrs', hours: 24, growth: 1.15 },
  { label: '+48 Hrs', hours: 48, growth: 1.34 },
  { label: '+72 Hrs', hours: 72, growth: 1.46 },
]

/** "Simulate 40% Traffic Drop" -> 0.6x traffic emission intensity. */
const TRAFFIC_FACTOR = 0.6

const NORMAL_OPACITY = 0.78
const MITIGATED_OPACITY = 0.5

const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

let pane: Pane | null = null
let horizonIdx = 0
let trafficDrop = false
let hasFitted = false
let getLayers: () => Layers | null = () => null
let getActivePreset: () => Preset | null = () => null
/** Cached /api/forecast payload for `fcPreset`. */
let fc: Forecast | null = null
let fcPreset = ''
let fcError = ''
let fcLoading = false

/* ------------------------------------------------ live forecast (API-backed) */

/** Index of the current slider stop inside the 73 hourly forecast steps. */
function hourIndex(): number | null {
  if (!fc || !fc.hours.length) return null
  return Math.min(HORIZONS[horizonIdx].hours, fc.hours.length - 1)
}

function num(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !isFinite(v) ? '–' : v.toFixed(digits)
}

/** Human ventilation class from the model's dispersion index (0..1). */
function ventilation(disp: number | null | undefined): string {
  const d = typeof disp === 'number' && isFinite(disp) ? disp : 0
  return d >= 0.7 ? 'good' : d >= 0.5 ? 'moderate' : d >= 0.3 ? 'poor' : 'very poor'
}

/** Fetch the live forecast for the selected city (backend caches it 30 min). */
async function loadForecast(force = false): Promise<void> {
  const id = getActivePreset()?.id ?? 'mumbai'
  if (!force && fc && fcPreset === id) return
  fcPreset = id
  fcLoading = true
  if (!fc) renderForecast()
  try {
    fc = await api.forecast(id)
    fcError = ''
  } catch (err) {
    fc = null
    fcError = err instanceof Error ? err.message : String(err)
  } finally {
    fcLoading = false
  }
  renderForecast()
}

/** Drop the cached forecast so the next render refetches (city change). */
export function resetForecast(): void {
  fc = null
  fcPreset = ''
  fcError = ''
  fcLoading = false
  // A new city (or a new analysis) must also re-frame Page 2 on its next
  // render — without this the map stayed parked on the previous city forever.
  hasFitted = false
}

/**
 * Redraw and make sure a forecast for the current city is in flight. Called
 * after `loadLayers()` too, so the projection never keeps showing the
 * synthetic plume while the real layers are still arriving.
 */
export function syncForecast(): void {
  renderForecast()
  void loadForecast()
}

/* ------------------------------------------------------------------ helpers */

/** Bounds of the city currently selected in the UI (null before config loads). */
function presetBounds(): LatLngBounds | null {
  const p = getActivePreset()
  if (!p) return null
  return L.latLngBounds([
    [p.bbox[1], p.bbox[0]],
    [p.bbox[3], p.bbox[2]],
  ])
}

/** Bounds of the loaded analysis grid, or null when no result is loaded. */
function layersBounds(): LatLngBounds | null {
  const ls = getLayers()
  if (!ls || !ls.lats.length || !ls.lons.length) return null
  const dlat = ls.lats.length > 1 ? ls.lats[1] - ls.lats[0] : 0.01
  const dlon = ls.lons.length > 1 ? ls.lons[1] - ls.lons[0] : 0.01
  return L.latLngBounds([
    [ls.lats[0] - dlat / 2, ls.lons[0] - dlon / 2],
    [ls.lats[ls.lats.length - 1] + dlat / 2, ls.lons[ls.lons.length - 1] + dlon / 2],
  ])
}

/** True when the loaded analysis actually covers the selected city. */
function layersInCity(): boolean {
  const lb = layersBounds()
  if (!lb) return false
  const pb = presetBounds()
  return pb === null || lb.intersects(pb)
}

/**
 * Domain Page 2 frames and draws: the analysis grid when it belongs to the
 * selected city, otherwise the selected city itself. Preferring the grid
 * unconditionally is what left the projection parked on the previous city.
 */
function activeBounds(): LatLngBoundsExpression {
  const lb = layersBounds()
  const pb = presetBounds()
  if (lb && (pb === null || lb.intersects(pb))) return lb
  if (pb) return pb
  return L.latLngBounds([
    [18.85, 72.7],
    [19.35, 73.3],
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
  const h = HORIZONS[horizonIdx]
  const i = hourIndex()
  if (fc && i !== null) {
    const stamp = fc.hours[i].replace('T', ' ')
    const zone = (fc.timezone.split('/').pop() ?? fc.timezone).replace(/_/g, ' ')
    return `${stamp} ${zone}  (T+${h.hours}h)`
  }
  const ls = getLayers()
  if (!ls || !ls.times || !ls.times.length) return `T+${h.hours}h from latest analysis`
  const last = ls.times[ls.times.length - 1]
  const d = new Date(last.replace('Z', ''))
  d.setUTCHours(d.getUTCHours() + h.hours)
  return `${d.toISOString().slice(0, 16).replace('T', ' ')}Z  (T+${h.hours}h)`
}

function renderAlert(): void {
  const box = $('redAlert')
  const i = hourIndex()
  const live = fc !== null && i !== null
  // Offline there is no stagnation/level to judge, so never claim a red alert:
  // it arms only from the forecast's own numbers (stagnation ≥ 0.70 or the
  // projected mean reaching the CPCB band).
  const armed = live ? fc!.model.alert[i!] : false
  const reason = live ? fc!.model.alert_reason[i!] : ''
  box.classList.toggle('ok', !armed)
  box.classList.toggle('danger', armed)
  $('alertIcon').textContent = armed ? '⚠️' : '✔'
  if (armed) {
    $('alertText').textContent = `RED ALERT: ${reason || 'High NO2 stagnation'} — trigger GRAP protocols`
    return
  }
  const wind = live ? num(fc!.met.wind_ms[i!]) : null
  $('alertText').textContent = wind
    ? `Air quality within expected limits · dispersion favourable (wind ${wind} m/s)`
    : 'Air quality within expected limits'
}

function renderDrivers(): void {
  const h = HORIZONS[horizonIdx]
  const i = hourIndex()
  if (fc && i !== null) {
    const { met, model } = fc
    const dir = met.wind_dir_deg[i]
    const from =
      dir === null || dir === undefined ? '' : ` from ${COMPASS[Math.round(dir / 22.5) % 16]}`
    $('forecastDrivers').innerHTML = [
      `Horizon: <b>${h.label}</b>`,
      `Wind: <b>${num(met.wind_ms[i])} m/s</b>${from}`,
      `Humidity: <b>${num(met.humidity_pct[i], 0)} %</b>`,
      `Boundary layer: <b>${num(met.blh_m[i], 0)} m</b>`,
      `Rain: <b>${num(met.precip_mm[i], 2)} mm/h</b> · cloud <b>${num(met.cloud_pct[i], 0)} %</b>`,
      `Temperature: <b>${num(met.temperature_c[i])} °C</b>`,
      `Ventilation: <b>${ventilation(model.dispersion_index[i])}</b>`,
      `Stagnation index: <b>${num(model.stagnation[i], 2)}</b>`,
      `Projected mean: <b>${num(model.projected_ugm3[i])} µg/m³</b>`,
      `<span class="hint">baseline ${num(fc.baseline_ugm3)} µg/m³ · ${fc.baseline_source}</span>`,
      `<span class="hint">met: ${fc.source}${fc.stale ? ' (cached)' : ''}</span>`,
    ].join('<br>')
    const bau = model.projected_ugm3[i]
    const mit = fc.baseline_ugm3 * model.factor_mitigated[i]
    $('scenarioSummary').textContent = trafficDrop
      ? `scenario: −40% traffic → ${mit.toFixed(1)} µg/m³ (−${Math.max(0, Math.round((1 - mit / (bau || 1)) * 100))}%)`
      : `scenario: business as usual → ${bau.toFixed(1)} µg/m³`
    return
  }
  // No forecast payload: keep the page honest about being offline.
  const wind = (3.4 - h.hours * 0.028).toFixed(1)
  const blh = Math.round(980 - h.hours * 6.4)
  const vent = h.hours >= 48 ? 'very poor' : h.hours >= 24 ? 'poor' : 'moderate'
  $('forecastDrivers').innerHTML = [
    `Horizon: <b>${h.label}</b>`,
    `Wind: <b>${wind} m/s</b> (offline estimate)`,
    `Boundary layer: <b>${blh} m</b>`,
    `Ventilation: <b>${vent}</b>`,
    `Stagnation index: <b>${(0.31 + h.hours * 0.008).toFixed(2)}</b>`,
    `<span class="hint">${
      fcLoading ? 'fetching live meteorology…' : `${fcError || 'live met unavailable'} — offline profile`
    }</span>`,
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
  const i = hourIndex()

  const cityBbox = getActivePreset()?.bbox ?? [72.7, 18.85, 73.3, 19.35]
  let base: (number | null)[][]
  let synthetic = false
  if (ls && layersInCity()) {
    base = frameAt(ls.layers.prediction, ls.t_len - 1) ?? syntheticField(cityBbox)
  } else {
    // No result for *this* city yet: show the labelled simulated plume instead
    // of stretching the previous city's grid over the newly selected one.
    base = syntheticField(cityBbox)
    synthetic = true
  }

  // Scaling comes from the live projection when it is available; the fixed
  // growth column is only the offline fallback.
  const growth =
    fc && i !== null
      ? trafficDrop
        ? fc.model.factor_mitigated[i]
        : fc.model.factor[i]
      : h.growth * (trafficDrop ? TRAFFIC_FACTOR : 1)
  const scaled = scaleFrame(base, growth)

  // Colour limits stay FIXED across horizons while only the field is scaled.
  // Scaling both made the ramp normalise the factor away, so every stop from
  // Now to +72 h rendered a pixel-identical image. The domain spans the
  // largest projected factor, so no horizon saturates the top of the ramp.
  const raw = ls && !synthetic ? ls.ranges?.prediction : null
  const [lo, hi] = raw && raw.length === 2 ? [raw[0], raw[1]] : frameRange(base)
  const maxFactor = fc
    ? Math.max(1, ...fc.model.factor, ...fc.model.factor_mitigated)
    : Math.max(1, ...HORIZONS.map((x) => x.growth))
  const vmin = lo
  const vmax = Math.max(hi, hi * maxFactor)

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
    : `AI Downscaled 0.01° · ${ls ? ls.lons.length : '–'}×${ls ? ls.lats.length : '–'} cells${
        fc ? ' · live met' : fcLoading ? ' · fetching met…' : ' · met fallback'
      }`
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

  // Warm the forecast as soon as the app boots so Page 2 opens instantly.
  syncForecast()
}

/** Re-fit and redraw once the tab becomes visible (zero-size while hidden). */
export function refreshForecast(): void {
  if (!pane) return
  pane.map.invalidateSize()
  // Read the container, never map.getSize(): Leaflet's invalidateSize() is a
  // no-op until the map has a view, and getSize() would then hand back the
  // 0×0 it cached while this tab was hidden — the fit would never run and the
  // map would never load (no tiles, no overlay) for the rest of the session.
  const el = pane.map.getContainer()
  if (!hasFitted && el.clientWidth > 0 && el.clientHeight > 0) {
    pane.fit(activeBounds())
    hasFitted = true
  }
  syncForecast()
}
