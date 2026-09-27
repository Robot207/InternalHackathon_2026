type RGB = [number, number, number]

const VIRIDIS: RGB[] = [
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
  return interpolate(VIRIDIS, t)
}

export function diverging(t: number): RGB {
  return interpolate(COOLWARM, t)
}
