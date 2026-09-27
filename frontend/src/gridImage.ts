import { diverging, sequential } from './colormap'
import type { Cube } from './types'

export function frameAt(cube: Cube | undefined, t: number): (number | null)[][] | null {
  if (!cube || cube.length === 0) return null
  if (t >= 0 && t < cube.length) return cube[t]
  const h = cube[0].length
  const w = cube[0][0].length
  const out: (number | null)[][] = []
  for (let r = 0; r < h; r++) {
    const row: (number | null)[] = []
    for (let c = 0; c < w; c++) {
      let sum = 0
      let n = 0
      for (let f = 0; f < cube.length; f++) {
        const v = cube[f][r][c]
        if (typeof v === 'number' && isFinite(v)) {
          sum += v
          n++
        }
      }
      row.push(n > 0 ? sum / n : NaN)
    }
    out.push(row)
  }
  return out
}

export function frameToUrl(
  frame: (number | null)[][] | null,
  vmin: number,
  vmax: number,
  mode: 'seq' | 'div',
): string | null {
  if (!frame) return null
  const h = frame.length
  const w = frame[0].length
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  const img = ctx.createImageData(w, h)
  const span = vmax - vmin || 1
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = frame[r][c]
      const i = (r * w + c) * 4
      if (typeof v !== 'number' || !isFinite(v)) {
        img.data[i + 3] = 0
        continue
      }
      const t = (v - vmin) / span
      const [rr, gg, bb] = mode === 'div' ? diverging(t) : sequential(t)
      img.data[i] = rr
      img.data[i + 1] = gg
      img.data[i + 2] = bb
      img.data[i + 3] = 235
    }
  }
  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL()
}
