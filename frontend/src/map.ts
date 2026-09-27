import L from 'leaflet'
import type { ImageOverlay, LatLngBoundsExpression, Map as LeafletMap } from 'leaflet'

export interface Pane {
  map: LeafletMap
  setOverlay: (url: string | null, bounds: LatLngBoundsExpression, opacity?: number) => void
  fit: (bounds: LatLngBoundsExpression) => void
}

export function createPane(el: HTMLElement): Pane {
  const map = L.map(el, { zoomControl: true, attributionControl: true, minZoom: 3 })
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map)
  let overlay: ImageOverlay | null = null
  return {
    map,
    setOverlay(url, bounds, opacity = 0.78) {
      if (overlay) {
        map.removeLayer(overlay)
        overlay = null
      }
      if (!url) return
      overlay = L.imageOverlay(url, bounds, { opacity, interactive: false })
      overlay.addTo(map)
      overlay.setZIndex(1)
    },
    fit(bounds) {
      map.fitBounds(bounds, { padding: [24, 24] })
    },
  }
}

export function syncMaps(a: LeafletMap, b: LeafletMap): void {
  let guard = false
  const mirror = (src: LeafletMap, dst: LeafletMap) => {
    src.on('move zoom', () => {
      if (guard) return
      guard = true
      dst.setView(src.getCenter(), src.getZoom(), { animate: false })
      guard = false
    })
  }
  mirror(a, b)
  mirror(b, a)
}
