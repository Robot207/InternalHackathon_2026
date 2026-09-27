/** Header "Export Dataset (GeoTIFF / CSV)" dropdown with graceful degradation. */

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

type Messenger = (text: string, isError?: boolean) => void

function filenameFrom(res: Response, url: string): string {
  const cd = res.headers.get('content-disposition') || ''
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd)
  if (m && m[1]) return decodeURIComponent(m[1])
  return url.split('/').pop() || 'download'
}

async function download(url: string): Promise<void> {
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  const blob = await res.blob()
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = filenameFrom(res, url)
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(href), 4000)
}

export function initExportMenu(onMessage: Messenger): void {
  const btn = $<HTMLButtonElement>('btnExport')
  const menu = $('exportMenu')

  const close = () => {
    menu.classList.add('hidden')
    btn.setAttribute('aria-expanded', 'false')
  }

  btn.addEventListener('click', (e) => {
    e.stopPropagation()
    const open = menu.classList.toggle('hidden')
    btn.setAttribute('aria-expanded', String(!open))
  })

  document.addEventListener('click', (e) => {
    if (!(e.target as HTMLElement).closest('.export-wrap')) close()
  })

  menu.addEventListener('click', async (e) => {
    const link = (e.target as HTMLElement).closest('a') as HTMLAnchorElement | null
    if (!link) return
    e.preventDefault()
    close()
    const fmt = link.dataset.fmt
    onMessage(`exporting ${fmt}…`)
    try {
      await download(link.getAttribute('href') || '')
      onMessage(`${fmt} export ✓`)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      // GeoTIFF needs rasterio; fall back to NetCDF rather than dead-ending.
      if (fmt === 'geotiff' && /rasterio|unavailable/i.test(msg)) {
        try {
          await download('/api/export/netcdf')
          onMessage('GeoTIFF unavailable — downloaded NetCDF instead')
          return
        } catch {
          /* fall through */
        }
      }
      onMessage(msg, true)
    }
  })
}
