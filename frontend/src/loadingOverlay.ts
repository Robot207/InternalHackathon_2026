/** Full-screen staged loading overlay shown when the user switches city. */

const STEPS = [
  'Fetching Sentinel-5P...',
  'Imputing Gaps...',
  'Applying XGBoost...',
  'Rendering 1km Grid...',
]

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

let runId = 0
let visible = false

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export function isLoadingVisible(): boolean {
  return visible
}

function setStep(i: number): void {
  $('loStep').textContent = STEPS[i]
  $('loBar').style.width = `${Math.round(((i + 1) / STEPS.length) * 100)}%`
}

function show(): void {
  const el = $('loadingOverlay')
  el.classList.remove('hidden')
  visible = true
}

function hide(): void {
  $('loadingOverlay').classList.add('hidden')
  visible = false
}

/**
 * Cycle through the pipeline stages for ~`totalMs`, then await `work` (if any)
 * so the map is only repainted once both the animation and the real request are
 * finished. Concurrent calls cancel the previous one.
 */
export async function runLoadingSequence(totalMs = 2000, work?: Promise<unknown>): Promise<void> {
  const id = ++runId
  show()
  $('loBar').style.width = '0%'
  const per = Math.max(250, Math.floor(totalMs / STEPS.length))
  try {
    for (let i = 0; i < STEPS.length; i++) {
      if (id !== runId) return
      setStep(i)
      await sleep(per)
    }
    if (id !== runId) return
    if (work) await work
  } finally {
    if (id === runId) hide()
  }
}
