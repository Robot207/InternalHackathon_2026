import L from 'leaflet'
import type { ImageOverlay, LatLngBoundsExpression, Map as LeafletMap } from 'leaflet'

export interface MarkerData {
  lat: number
  lon: number
  title: string
  popupHtml: string
  isLandmark?: boolean
  color?: string
}

export interface Pane {
  map: LeafletMap
  setOverlay: (url: string | null, bounds: LatLngBoundsExpression, opacity?: number) => void
  fit: (bounds: LatLngBoundsExpression) => void
  setMarkers: (markers: MarkerData[]) => void
  clearMarkers: () => void
  onClick: (cb: (lat: number, lon: number) => void) => void
}

export function createPane(el: HTMLElement): Pane {
  const map = L.map(el, { zoomControl: true, attributionControl: true, minZoom: 3 })
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map)

  let overlay: ImageOverlay | null = null
  const markerGroup = L.layerGroup().addTo(map)

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
    setMarkers(items: MarkerData[]) {
      markerGroup.clearLayers()
      for (const it of items) {
        const bg = it.color || (it.isLandmark ? '#38bdf8' : '#eab308')
        const iconHtml = it.isLandmark
          ? `<div class="custom-pin landmark" style="background:${bg};" title="${it.title}">⭐</div>`
          : `<div class="custom-pin station" style="background:${bg};" title="${it.title}">📡</div>`
        const icon = L.divIcon({
          className: 'pin-wrapper',
          html: iconHtml,
          iconSize: [26, 26],
          iconAnchor: [13, 13],
        })
        const marker = L.marker([it.lat, it.lon], { icon })
        marker.bindPopup(it.popupHtml, { maxWidth: 320 })
        markerGroup.addLayer(marker)
      }
    },
    clearMarkers() {
      markerGroup.clearLayers()
    },
    onClick(cb) {
      map.on('click', (e) => {
        cb(e.latlng.lat, e.latlng.lng)
      })
    },
  }
}

export function syncMaps(a: LeafletMap, b: LeafletMap): void {
  let guard = false
  // A map that is inside a hidden tab reports 0x0 — never push view state into
  // it, otherwise Leaflet throws and the sync loop dies.
  const usable = (m: LeafletMap) => {
    const el = m.getContainer()
    if (!el.offsetParent && el.style.position !== 'fixed') return false
    const s = m.getSize()
    return s.x > 0 && s.y > 0
  }
  const mirror = (src: LeafletMap, dst: LeafletMap) => {
    src.on('move zoom', () => {
      if (guard) return
      if (!usable(src) || !usable(dst)) return
      guard = true
      dst.setView(src.getCenter(), src.getZoom(), { animate: false })
      guard = false
    })
  }
  mirror(a, b)
  mirror(b, a)
}

