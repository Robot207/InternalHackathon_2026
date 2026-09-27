import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import type { LayerGroup, LatLngBoundsExpression, Polyline } from 'leaflet'
import './style.css'

import { api, pollJob } from './api'
import { initCitySelect, syncCitySelect } from './citySelect'
import { initExportMenu } from './exportMenu'
import { initForecast, refreshForecast, resetForecast } from './forecast'
import { frameAt, frameToUrl } from './gridImage'
import { runLoadingSequence } from './loadingOverlay'
import { createPane, syncMaps, type MarkerData, type Pane } from './map'
import { initTabs } from './tabs'
import { refreshValidation, renderValidation } from './validationPanel'
import type { AppConfig, BenchmarkArenaResponse, FrameValidation, Job, Layers, Meta, Metrics, Preset, StationBenchmarkResponse, Summary } from './types'

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
/** Last point clicked on the map — re-read whenever the timeline frame changes. */
let lastInspect: { lat: number; lon: number } | null = null
/** Monotonic id so a slow response can't overwrite a newer frame's values. */
let inspectSeq = 0
let inspectTimer: number | null = null
/**
 * Unseen-data metrics for the frame on screen. Panel 4 used to quote
 * `meta.metrics`, one number computed at train time, so scrubbing the timeline
 * changed nothing — this is re-fetched per frame instead.
 */
let frameVal: FrameValidation | null = null
let frameValSeq = 0
let frameValTimer: number | null = null
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
  done: (job: Job) => Promise<void>,
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
    await done(job)
    void refreshValidation()
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
  // Applying/training while another city is selected would run the backend on
  // the previous dataset and then drag the map back to it — block it instead.
  const mismatch = !!summary && !selectionMatchesResults()
  $<HTMLButtonElement>('btnFetch').disabled = jobRunning
  $<HTMLButtonElement>('btnTrain').disabled = jobRunning || !summary || !summary.has_reference || mismatch
  $<HTMLButtonElement>('btnApply').disabled = jobRunning || !summary || !hasModel || mismatch
  const btnArena = document.getElementById('btnArena') as HTMLButtonElement | null
  if (btnArena) {
    btnArena.disabled = jobRunning || !summary || !summary.has_reference || mismatch
  }
  const btnMumbai = document.getElementById('btnMumbaiStations') as HTMLButtonElement | null
  if (btnMumbai) {
    btnMumbai.disabled = jobRunning || !hasModel
  }
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
  const gapConf = summary.gap_recovery_confidence !== undefined
    ? ` (confidence <b>${Math.round((summary.gap_recovery_confidence as number) * 100)}%</b> · ${summary.gapfill_method || 'physics-informed'})`
    : ''
  el.innerHTML = [
    `<b>${summary.preset_label}</b> · ${summary.start_date} → ${summary.end_date}`,
    `grid ${summary.n_lat}×${summary.n_lon} @ ${summary.fine_step}° · ${summary.n_times} hours`,
    `cloud gap: <b>${Math.round(summary.gap_fraction * 100)}%</b> repaired${gapConf}`,
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
      msgs.push('model results are from a previous dataset — run Apply/Train for the loaded one')
    }
  }
  // The dropdown and the backend's active dataset can disagree (the city was
  // changed after fetching, or the fetch for it failed). Say so plainly and
  // point at the action that reconciles them instead of silently snapping back.
  const selected = activePreset()
  if (summary && selected && summary.preset !== selected.id) {
    msgs.push(
      `active dataset is <b>${summary.preset_label || summary.preset}</b> — run <b>⬇ Fetch coarse data</b> for <b>${selected.label}</b> before Apply/Train`,
    )
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

/**
 * One line of honesty under the cards: which frame was scored, against what,
 * or — when the region has no independent truth — what the shown numbers
 * actually are (they never silently pass for local validation).
 */
function metricsNote(fm: Metrics | undefined, fv: FrameValidation | null): string {
  if (fm && fv?.frame) {
    return [`⏱ ${fv.frame.label}`, fv.holdout_description, fv.source_label, fv.note]
      .filter(Boolean)
      .join(' · ')
  }
  if (fv && !fv.available && fv.source !== 'none') {
    // Independent truth exists here, just not on this frame (e.g. a temporal
    // holdout whose unseen hours are only the tail of the timeline).
    return [`⏱ ${fv.frame?.label ?? 'this frame'}: ${fv.reason ?? 'no unseen samples'}`, fv.note]
      .filter(Boolean)
      .join(' · ')
  }
  if (fv && fv.source === 'none') {
    const place = summary?.preset ? ` for ${summary.preset.replace(/_/g, ' ')}` : ''
    const shown =
      meta?.mode === 'transfer'
        ? "the benchmark region's holdout — a different region, not this city"
        : "this result's whole-period holdout"
    return `No independent local reference${place} — the cards show ${shown}, identical on every frame.`
  }
  return 'Whole-period holdout — one number, identical on every timeline frame.'
}

function renderMetrics(): void {
  const el = $('metrics')
  const note = $('metricsNote')
  if (!meta) {
    el.textContent = 'run training to see metrics'
    note.textContent = ''
    renderScatter(null)
    renderImportance(null)
    return
  }
  // Frame-level score when this frame has truth to score against; otherwise
  // fall back to the period-level numbers (labelled, never implied as local).
  const fv = frameVal
  const fm = fv?.available && fv.metrics ? fv.metrics : undefined
  const m = fm ?? meta.metrics ?? (meta.mode === 'transfer' ? meta.benchmark?.metrics : undefined)
  if (!m) {
    el.textContent = 'no held-out metrics for this result'
    note.textContent = ''
    renderScatter(null)
    renderImportance(meta.importances ?? meta.benchmark?.importances ?? null)
    return
  }
  const splitLabel = (fm ? fv?.holdout_description : undefined) ?? meta.holdout_description ?? meta.benchmark?.holdout_description
  const nHold = fm ? m.n : meta.n_test ?? meta.benchmark?.n_test ?? m.n
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
  note.textContent = metricsNote(fm, fv)
  renderScatter(meta)
  renderImportance(meta.importances ?? meta.benchmark?.importances ?? null)
}

function renderScatter(m: Meta | null): void {
  const svg = $('scatter')
  const note = $('scatterNote')
  svg.innerHTML = ''
  if (!m || !layers) {
    note.textContent = ''
    return
  }
  const pred = frameAt(layers.layers.prediction, timeIdx)
  const ref = frameAt(layers.layers.reference, timeIdx)
  if (!pred || !ref) {
    note.textContent = ''
    return
  }
  const pts: [number, number][] = []
  for (let r = 0; r < pred.length; r++) {
    for (let c = 0; c < pred[0].length; c++) {
      const a = pred[r][c]
      const b = ref[r][c]
      if (typeof a === 'number' && typeof b === 'number' && isFinite(a) && isFinite(b)) pts.push([b, a])
    }
  }
  if (!pts.length) {
    // An empty plot is only honest if it says *why* it is empty.
    note.textContent = layers.ranges.reference
      ? 'No paired prediction/reference samples on this frame.'
      : 'No independent fine reference for this city — nothing to plot against.'
    return
  }
  note.textContent = ''
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
  // Never paint a grid that belongs to another city: it is drawn at that
  // city's bounds, so it reads as "the map went back" as soon as you pan.
  if (!layers || !selectionMatchesResults()) {
    clearOverlays()
    clearLegend()
    return
  }
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

/** Blank the colour ramp when there is no data to show for this city. */
function clearLegend(): void {
  const canvas = $<HTMLCanvasElement>('legendCanvas')
  canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  $('legendLabels').textContent = ''
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
    // Frame the selected city, not the raw grid: when the fetched grid belongs
    // to a different city (job racing a city switch) the fit must stay put.
    frameSelectedCity()
    renderOverlays()
    updateTimeLabel()
    renderMetrics()
    // The panel must describe the frame/city that is on screen now.
    refreshInspect()
    // A new city/frame changes what "unseen data" even means here.
    clearFrameValidation()
    refreshFrameValidation()
    // A new analysis changes the baseline the 72 h projection starts from.
    resetForecast()
    refreshForecast()
    void loadBenchmarkStations(false)
    return true
  } catch {
    layers = null
    // No layers for this city: remove the previous city's pixels instead of
    // leaving them under the newly framed view.
    clearOverlays()
    updateTimeLabel()
    clearFrameValidation()
    renderMetrics()
    return false
  }
}

async function refreshState(syncCity = false): Promise<void> {
  const st = await api.state()
  summary = st.summary
  meta = st.meta
  hasModel = st.has_model
  const s = summary
  // Only the initial load adopts the backend's city. After that the dropdown
  // follows the user: re-syncing it here silently snapped the selection (and,
  // through Apply, the map) back to the previously fetched city.
  if (s && syncCity) {
    const sel = $<HTMLSelectElement>('preset')
    if (Array.from(sel.options).some((o) => o.value === s.preset)) {
      sel.value = s.preset
    }
  }
  syncCitySelect()
  renderValidation(st.validation)
  renderDataInfo()
  renderWarnings()
  renderMetrics()
  // meta just changed: any cached frame score belongs to the old result.
  clearFrameValidation()
  refreshFrameValidation()
  drawGridlines()
  updateButtons()
}

function setPresetDefaults(): void {
  const id = ($('preset') as HTMLSelectElement).value
  const p = config.presets.find((x) => x.id === id)
  if (!p) return
  const bounds = layerBoundsFor(p.bbox)
  // Always frame the newly selected city: loaded results may belong to a
  // different domain, and leaving the map on the old one looks broken.
  leftPane.fit(bounds)
  // Repaint with whatever belongs to this city (drops a foreign grid) and
  // refresh the mismatch warning/buttons straight away.
  renderOverlays()
  renderWarnings()
  updateButtons()
  // The point inspector quotes numbers from the *previous* city's grid until it
  // is re-read — say so instead of leaving yesterday's values under a new map.
  if (!selectionMatchesResults()) {
    lastInspect = null
    if (inspectTimer !== null) window.clearTimeout(inspectTimer)
    $('inspectorContent').innerHTML =
      '<div class="hint">No results for this city yet — run <b>Apply model (transfer)</b> to inspect points here.</div>'
    // Likewise: a frame score for the previous city must not describe this map.
    clearFrameValidation()
    renderMetrics()
  } else {
    refreshInspect()
  }
  // Page 2 must follow the same city: drop the previous projection, re-frame
  // its map (refreshForecast only fits once it has pixels) and refetch the met.
  resetForecast()
  refreshForecast()
  void loadBenchmarkStations(false)
}

function layerBoundsFor(bbox: number[]): LatLngBoundsExpression {
  return L.latLngBounds([
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ])
}

/** Currently selected preset (city), or null before config has loaded. */
function activePreset(): Preset | null {
  const id = $<HTMLSelectElement>('preset').value
  return config?.presets.find((p) => p.id === id) ?? null
}

/**
 * True when the loaded analysis belongs to the city that is selected right now.
 *
 * The two can drift apart: picking a city only reframes the map, while the
 * backend keeps serving whatever dataset was fetched/applied last (and a job
 * started before a city switch finishes against the old one). Anything that
 * paints or frames the map must go through this so a stale grid can never
 * pull the view back to the previously analysed city.
 */
function selectionMatchesResults(): boolean {
  const p = activePreset()
  return !!p && !!summary && summary.preset === p.id
}

/** Drop both panes' image overlays (used when there is nothing to show here). */
function clearOverlays(): void {
  const empty = L.latLngBounds([
    [0, 0],
    [0, 0],
  ])
  leftPane.setOverlay(null, empty)
  rightPane.setOverlay(null, empty)
}

/** Frame the city the user selected — never a grid that belongs to another one. */
function frameSelectedCity(): void {
  const p = activePreset()
  if (p) {
    leftPane.fit(layerBoundsFor(p.bbox))
  } else if (layers) {
    leftPane.fit(layerBounds(layers))
  }
}

async function inspectPointAt(lat: number, lon: number, silent = false): Promise<void> {
  const el = $('inspectorContent')
  // Remember the point so the panel can be re-read when the timeline moves —
  // otherwise it keeps reporting the frame that was on screen when clicked.
  lastInspect = { lat, lon }
  const seq = ++inspectSeq
  if (!silent) el.innerHTML = `<div class="hint">Querying point (${lat.toFixed(4)}, ${lon.toFixed(4)})…</div>`
  try {
    const res = await api.inspectPoint(lat, lon, timeIdx)
    if (seq !== inspectSeq) return // a newer click/scrub superseded this one
    const cur = res.current
    const aqi = cur.aqi
    const lm = res.nearest_landmark
    const pVal = cur.downscaled_no2 !== null ? `${cur.downscaled_no2} µg/m³` : 'N/A'
    const bVal = cur.baseline_no2 !== null ? `${cur.baseline_no2} µg/m³` : 'N/A'
    const frame = res.frame
    const frameLabel = frame?.label ?? 'latest hour'
    const isMean = frame?.is_mean ?? false
    // The left pane's default "Coarse input (0.25°)" layer draws block pixels,
    // which differ from the kriged baseline in the metrics — show it only while
    // that layer is the one on screen.
    const showBlock =
      cur.baseline_block_no2 != null &&
      ($('layerLeft') as HTMLSelectElement).value === 'coarse'
    const blockLine = showBlock
      ? `<div class="hint" style="margin-top:4px;">0.25° pixel on the left map: <b>${cur.baseline_block_no2} µg/m³</b></div>`
      : ''

    const series = res.diurnal_24h.downscaled.filter((v): v is number => v !== null)
    const where =
      lm.distance_km > 0 ? `${lm.distance_km} km away · ${lm.type}` : `in this ${lm.type}`
    // Outside the active grid the backend omits values rather than reporting a
    // cell from another city — explain that instead of showing bare N/A.
    const domainWarn =
      res.in_domain === false
        ? `<div class="hint" style="margin-top:6px;">Outside the active 1 km grid (<b>${res.active_preset ?? 'current dataset'}</b>) — place info only. Run <b>Fetch &amp; Downscale</b> for this city to get its NO₂ values.</div>`
        : ''
    let sparklineSvg = ''
    if (series.length > 1) {
      const min = Math.min(...series)
      const max = Math.max(...series)
      const w = 270
      const h = 42
      const pad = 4
      const pts = series
        .map((v, i) => {
          const x = pad + (i / (series.length - 1)) * (w - pad * 2)
          const y = h - pad - ((v - min) / (max - min || 1)) * (h - pad * 2)
          return `${x.toFixed(1)},${y.toFixed(1)}`
        })
        .join(' ')
      sparklineSvg = `
        <div style="font-size:10px; color:var(--dim); margin-top:6px;">${
          isMean
            ? 'Mean diurnal NO₂ cycle (averaged over the period):'
            : '24h Diurnal NO₂ Plume Trend:'
        }</div>
        <svg class="sparkline-svg" viewBox="0 0 ${w} ${h}">
          <polyline fill="none" stroke="#35d0c0" stroke-width="2" points="${pts}" />
        </svg>
      `
    }

    el.innerHTML = `
      <div class="inspector-header">
        <span class="inspector-place">📍 ${lm.name}</span>
        <span class="inspector-coords">${where}</span>
      </div>
      ${domainWarn}
      <div class="inspector-metrics">
        <div class="metric-box">
          <div class="label">ML Downscaled</div>
          <div class="val">${pVal}</div>
        </div>
        <div class="metric-box">
          <div class="label">Coarse Baseline</div>
          <div class="val" style="color:#94a3b8;">${bVal}</div>
        </div>
      </div>
      <div class="inspector-frame" title="The timeline frame these numbers belong to — the same one the map is painting.">⏱ at <b>${frameLabel}</b></div>
      ${blockLine}
      <div style="margin-top:6px; display:flex; align-items:center; gap:8px;">
        <span class="aqi-pill" style="background:${aqi.color};">${aqi.category}</span>
        <span style="font-size:11px; color:#cbd5e1;">${aqi.description}</span>
      </div>
      ${sparklineSvg}
      <div class="inspector-env">
        <span>Elevation: <b>${res.static_features.elevation_m ?? '—'}${res.static_features.elevation_m != null ? 'm' : ''}</b></span> ·
        <span>Roads: <b>${res.static_features.road_density_km_km2 ?? '—'}${res.static_features.road_density_km_km2 != null ? ' km/km²' : ''}</b></span> ·
        <span>Gap repaired: <b>${cur.cloud_gap_repaired ? 'Yes' : 'Direct observation'}</b></span>
      </div>
      ${lm.source ? `<div class="hint" style="margin-top:4px;">Place: ${lm.source}</div>` : ''}
    `
  } catch (err) {
    if (seq !== inspectSeq) return
    el.innerHTML = `<div class="info error">Inspection failed: ${err instanceof Error ? err.message : String(err)}</div>`
  }
}

/**
 * Re-read the selected point for the frame that is on screen now.
 *
 * Scrubbing the timeline repaints the map but not this panel, so without this
 * the panel would keep showing a different hour than the colours next to it.
 */
function refreshInspect(debounceMs = 0): void {
  if (!lastInspect) return
  const { lat, lon } = lastInspect
  if (inspectTimer !== null) window.clearTimeout(inspectTimer)
  const run = (): void => {
    inspectTimer = null
    void inspectPointAt(lat, lon, true)
  }
  if (debounceMs > 0) inspectTimer = window.setTimeout(run, debounceMs)
  else run()
}

/** Drop the previous city/frame's validation so stale numbers can't be shown. */
function clearFrameValidation(): void {
  frameVal = null
  if (frameValTimer !== null) window.clearTimeout(frameValTimer)
  frameValTimer = null
  frameValSeq++ // cancel anything in flight for the old frame
}

/**
 * Re-score the unseen data for the frame the map just painted. The panel used
 * to re-render one period-level number on every scrub tick, so it looked frozen.
 */
function refreshFrameValidation(debounceMs = 0): void {
  if (frameValTimer !== null) window.clearTimeout(frameValTimer)
  const run = (): void => {
    frameValTimer = null
    const seq = ++frameValSeq
    const t = timeIdx
    api
      .frameValidation(t)
      .then((res) => {
        if (seq !== frameValSeq) return // a newer scrub superseded this one
        frameVal = res
        renderMetrics()
      })
      .catch(() => {
        if (seq !== frameValSeq) return
        frameVal = null
        renderMetrics()
      })
  }
  if (debounceMs > 0) frameValTimer = window.setTimeout(run, debounceMs)
  else run()
}

async function loadBenchmarkStations(validate = false): Promise<void> {
  // Markers are geography, so they follow the city on screen. Only validation
  // (which needs predictions) targets the dataset the backend actually holds.
  const selValue = ($('preset') as HTMLSelectElement).value
  const currentPreset = validate
    ? summary?.preset || selValue || 'mumbai'
    : selValue || summary?.preset || 'mumbai'
  try {
    const data = await api.getBenchmarkStations(currentPreset)
    if (!data.stations.length && !data.landmarks.length) {
      if (validate) $('stationResult').textContent = 'No built-in stations for this region.'
      leftPane.clearMarkers()
      rightPane.clearMarkers()
      return
    }

    let validationData: StationBenchmarkResponse | null = null
    if (validate) {
      $('stationResult').textContent = 'validating against CPCB CAAQMS stations…'
      validationData = await api.validateBenchmarkStations(currentPreset)
      const m = validationData.metrics
      $('stationResult').innerHTML = `
        <b>Mumbai CPCB Benchmark:</b> n=${validationData.n_stations} stations<br>
        RMSE: <b>${m.rmse} µg/m³</b> (vs baseline ${m.baseline_rmse})<br>
        MAE: <b>${m.mae} µg/m³</b> · Pearson r: <b>${m.pearson}</b><br>
        Skill: <b>+${(m.skill_vs_baseline * 100).toFixed(1)}%</b> error reduction vs satellite coarse<br>
        <span class="hint">${validationData.source}</span>
      `
    }

    const showStationsChecked = ($('showStations') as HTMLInputElement)?.checked ?? true
    if (!showStationsChecked) {
      leftPane.clearMarkers()
      rightPane.clearMarkers()
      return
    }

    const markers: MarkerData[] = []
    const stationList = validationData ? validationData.stations : data.stations
    // Model figures are period means; without saying so they read as "right now"
    // and disagree with whatever hour the timeline is showing.
    const frameNote = validationData?.frame_label ?? 'period mean'
    for (const s of stationList) {
      const isEval = 'downscaled_no2' in s
      const html = `
        <div style="font-family:sans-serif; font-size:12px; line-height:1.4;">
          <b style="font-size:13px; color:#0f172a;">${s.name}</b><br>
          <span style="color:#64748b;">${s.type}</span><br>
          <hr style="margin:4px 0; border:none; border-top:1px solid #e2e8f0;" />
          <b>Observed Ground Truth:</b> ${s.observed_no2 || (s as { baseline_observed_no2?: number }).baseline_observed_no2} µg/m³<br>
          ${isEval ? `<b>ML Downscaled:</b> ${(s as { downscaled_no2: number }).downscaled_no2} µg/m³<br><b>Coarse Satellite:</b> ${(s as { coarse_satellite_no2: number }).coarse_satellite_no2} µg/m³<br><span style="color:${(s as { downscale_error: number }).downscale_error < 0 ? '#16a34a' : '#ea580c'}; font-weight:bold;">Error: ${(s as { downscale_error: number }).downscale_error} µg/m³ (${(s as { error_reduction_pct: number }).error_reduction_pct}% reduction)</span><br><i style="color:#64748b;">model value: ${frameNote} · observed: CPCB station average</i><br>` : ''}
          <i style="font-size:11px; color:#475569;">${s.notes}</i>
        </div>
      `
      markers.push({
        lat: s.lat,
        lon: s.lon,
        title: s.name,
        popupHtml: html,
        color: '#f59e0b',
      })
    }

    for (const lm of data.landmarks) {
      const html = `
        <div style="font-family:sans-serif; font-size:12px;">
          <b style="font-size:13px; color:#0f172a;">⭐ ${lm.name}</b><br>
          <span style="color:#64748b;">${lm.category}</span><br>
          <p style="margin:4px 0 0; color:#334155;">${lm.notes}</p>
        </div>
      `
      markers.push({
        lat: lm.lat,
        lon: lm.lon,
        title: lm.name,
        popupHtml: html,
        isLandmark: true,
        color: '#06b6d4',
      })
    }

    leftPane.setMarkers(markers)
    rightPane.setMarkers(markers)
  } catch (err) {
    if (validate) {
      $('stationResult').textContent = err instanceof Error ? err.message : String(err)
    }
  }
}

function renderArenaLeaderboard(res: BenchmarkArenaResponse): void {
  const box = $('arenaBox')
  box.classList.remove('hidden')
  const rows = res.leaderboard
    .map((item, idx) => {
      const isWinner = item.model_id === res.winner_id
      const m = item.metrics
      if (!m) {
        return `<tr><td>${idx + 1}</td><td>${item.model_name}</td><td colspan="5" style="color:var(--danger);">Error</td></tr>`
      }
      const skill = (m.skill_vs_baseline * 100).toFixed(1)
      return `
        <tr class="${isWinner ? 'winner-row' : ''}">
          <td>${isWinner ? '🏆' : idx + 1}</td>
          <td><b>${item.model_name}</b></td>
          <td><b>${m.rmse}</b></td>
          <td>${m.mae}</td>
          <td>${m.pearson}</td>
          <td style="color:${m.skill_vs_baseline >= 0 ? 'var(--accent)' : 'var(--danger)'};">${m.skill_vs_baseline >= 0 ? '+' : ''}${skill}%</td>
          <td>${item.fit_time_sec}s</td>
        </tr>
      `
    })
    .join('')

  box.innerHTML = `
    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
      <b>🏆 Model Benchmark Arena Leaderboard</b>
      <span style="color:var(--accent);">Winner: ${res.winner_name}</span>
    </div>
    <div style="font-size:10px; color:var(--dim); margin-bottom:6px;">Holdout: ${res.holdout_description} (n=${res.n_test} unseen samples)</div>
    <table class="arena-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Algorithm</th>
          <th>RMSE</th>
          <th>MAE</th>
          <th>r</th>
          <th>Skill</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `
}

async function runModelArena(): Promise<void> {
  const box = $('arenaBox')
  box.classList.remove('hidden')
  box.innerHTML = '<div class="hint">Running multi-model arena across all algorithms…</div>'
  const splitVal = ($<HTMLSelectElement>('split')).value
  const conserveVal = $<HTMLInputElement>('conserve').checked

  await runJob(
    api.benchmarkModels(splitVal, conserveVal),
    async (job) => {
      if (job.result) {
        renderArenaLeaderboard(job.result as BenchmarkArenaResponse)
      }
    },
    'arena',
  )
}

let isSwipeMode = false
let swipePercent = 50

function initSwipeMode(): void {
  const container = $('mapsContainer')
  const divider = $('swipeDivider')
  const btnDual = $('btnDualView')
  const btnSwipe = $('btnSwipeView')
  const mapRight = $('mapRight')

  function updateClip(): void {
    if (!isSwipeMode) {
      mapRight.style.clipPath = ''
      return
    }
    const rect = container.getBoundingClientRect()
    const x = (rect.width * swipePercent) / 100
    divider.style.left = `${x}px`
    mapRight.style.clipPath = `polygon(${x}px 0, 100% 0, 100% 100%, ${x}px 100%)`
  }

  btnDual.addEventListener('click', () => {
    isSwipeMode = false
    btnDual.classList.add('active')
    btnSwipe.classList.remove('active')
    container.classList.remove('swipe-active')
    divider.classList.add('hidden')
    updateClip()
    leftPane.map.invalidateSize()
    rightPane.map.invalidateSize()
  })

  btnSwipe.addEventListener('click', () => {
    isSwipeMode = true
    btnSwipe.classList.add('active')
    btnDual.classList.remove('active')
    container.classList.add('swipe-active')
    divider.classList.remove('hidden')
    updateClip()
    leftPane.map.invalidateSize()
    rightPane.map.invalidateSize()
  })

  let isDragging = false
  divider.addEventListener('mousedown', (e) => {
    isDragging = true
    e.preventDefault()
  })
  window.addEventListener('mouseup', () => {
    isDragging = false
  })
  window.addEventListener('mousemove', (e) => {
    if (!isDragging || !isSwipeMode) return
    const rect = container.getBoundingClientRect()
    const x = Math.max(10, Math.min(rect.width - 10, e.clientX - rect.left))
    swipePercent = (x / rect.width) * 100
    updateClip()
  })
  window.addEventListener('resize', () => {
    if (isSwipeMode) updateClip()
  })
}

async function init(): Promise<void> {
  leftPane = createPane($('mapLeft'))
  rightPane = createPane($('mapRight'))
  syncMaps(leftPane.map, rightPane.map)

  leftPane.onClick((lat, lon) => {
    void inspectPointAt(lat, lon)
  })
  rightPane.onClick((lat, lon) => {
    void inspectPointAt(lat, lon)
  })

  initSwipeMode()

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

  // --- feature modules -------------------------------------------------
  initTabs((page) => {
    if (page === 'page2') {
      refreshForecast()
    } else {
      leftPane.map.invalidateSize()
      rightPane.map.invalidateSize()
    }
  })

  initExportMenu((text, isError) => status(text, !!isError))

  initForecast({ getLayers: () => layers, getActivePreset: activePreset })

  // Searchable 100+ city combobox layered over the native <select id="preset">.
  initCitySelect((id) => {
    void (async () => {
      status(`switching to ${config.presets.find((p) => p.id === id)?.label ?? id}…`)
      // Full-screen staged animation (~2s) before the map repaints.
      await runLoadingSequence(2000)
      setPresetDefaults() // also resets + re-frames the 72 h projection
      updateButtons()
      await refreshValidation(id)
      status('ready')
    })()
  })

  const end = new Date()
  end.setUTCDate(end.getUTCDate() - 1)
  const start = new Date(end)
  start.setUTCDate(start.getUTCDate() - 6)
  $<HTMLInputElement>('endDate').value = fmtDay(end)
  $<HTMLInputElement>('startDate').value = fmtDay(start)

  await refreshState(true)
  const loaded = summary ? await loadLayers() : false
  if (!loaded) frameSelectedCity()
  status(summary ? 'previous results loaded' : 'ready')
  void refreshValidation(presetSel.value)

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
        // Reload so the view shows this city's grid (or nothing but its bbox
        // until Apply) instead of keeping the previously fetched city's pixels.
        await loadLayers()
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

  $('btnArena').addEventListener('click', () => {
    void runModelArena()
  })

  $('btnMumbaiStations').addEventListener('click', () => {
    void loadBenchmarkStations(true)
  })

  $('showStations').addEventListener('change', () => {
    void loadBenchmarkStations(false)
  })

  presetSel.addEventListener('change', () => {
    setPresetDefaults()
    updateButtons()
    void refreshValidation(presetSel.value)
  })
  ;($('layerLeft') as HTMLSelectElement).addEventListener('change', () => {
    renderOverlays()
    // The "0.25° pixel" line only applies while the block layer is drawn.
    refreshInspect()
  })
  ;($('layerRight') as HTMLSelectElement).addEventListener('change', renderOverlays)
  ;($('gridCoarse') as HTMLInputElement).addEventListener('change', drawGridlines)
  ;($('gridFine') as HTMLInputElement).addEventListener('change', drawGridlines)

  const slider = $<HTMLInputElement>('timeSlider')
  slider.addEventListener('input', () => {
    timeIdx = Number(slider.value)
    renderOverlays()
    updateTimeLabel()
    // Keep the inspector *and* the model-vs-reference scatter on the frame the
    // map just painted (debounced: `input` fires on every pixel of a scrub).
    refreshInspect(180)
    // Same frame, new score: re-run the unseen-data validation for it.
    refreshFrameValidation(180)
    if (meta) renderMetrics()
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
      refreshInspect()
      refreshFrameValidation(150)
      if (meta) renderMetrics()
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

