import { api } from './api'
import type { Validation } from './types'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

/**
 * Renders the persistent validation dashboard:
 *   Active algorithms · Validation protocol · Live RMSE
 */
export function renderValidation(v: Validation | undefined): void {
  const algo = $('vAlgorithms')
  const proto = $('vProtocol')
  const rmse = $('vRmse')
  const note = $('vNote')

  if (!v) {
    algo.textContent = '–'
    proto.textContent = '–'
    rmse.textContent = '–'
    note.textContent = 'run Fetch → Train to populate validation'
    return
  }

  algo.textContent = v.algorithms
  proto.textContent = v.protocol
  rmse.textContent = `${v.rmse_score} ${v.rmse_unit}`

  rmse.classList.toggle('estimated', !!v.estimated)
  rmse.classList.toggle('live', !v.estimated)

  note.textContent = v.estimated
    ? `${v.reason} · reference value`
    : `${v.reason} · ${v.n_folds} held-out folds`
  note.title = v.reason
}

/** Re-fetch LOSO metrics from the backend (after a job or a city change). */
export async function refreshValidation(preset?: string): Promise<void> {
  try {
    const v = await api.loso(preset)
    renderValidation(v)
  } catch {
    // Keep whatever is currently on screen rather than blanking the panel.
  }
}
