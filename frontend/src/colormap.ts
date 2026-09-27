type RGB = [number, number, number]

const NO2_PALETTE: RGB[] = [
  [22, 48, 130],   // 0.00: Deep Blue (Clean background)
  [25, 135, 185],  // 0.15: Cyan / Sky Blue
  [40, 185, 115],  // 0.30: Emerald Green (Good)
  [160, 215, 45],  // 0.45: Lime Green
  [248, 212, 32],  // 0.60: Bright Yellow (Moderate)
  [255, 125, 15],  // 0.75: Vivid Orange (Poor / Stagnant)
  [225, 30, 40],   // 0.90: Crimson Red (Severe)
  [140, 10, 80],   // 1.00: Deep Purple (Critical)
]

const COOLWARM: RGB[] = [
  [59, 76, 192],
  [122, 161, 233],
  [201, 221, 237],
  [221, 221, 221],
  [245, 179, 100],
  [228, 108, 48],
  [180, 4, 38],
]

function interpolate(anchors: RGB[], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (anchors.length - 1)
  const i = Math.min(anchors.length - 2, Math.floor(x))
  const f = x - i
  const a = anchors[i]
  const b = anchors[i + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ]
}

export function sequential(t: number): RGB {
  return interpolate(NO2_PALETTE, t)
}

export function diverging(t: number): RGB {
  return interpolate(COOLWARM, t)
}
