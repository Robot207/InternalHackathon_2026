/** Simple two-tab navigation between Page 1 (Live Downscaling) and Page 2 (forecast). */

export type PageId = 'page1' | 'page2'

let current: PageId = 'page1'
let onChange: ((page: PageId) => void) | null = null

export function activePage(): PageId {
  return current
}

function show(page: PageId): void {
  if (current === page) return
  current = page
  document.getElementById('page1')!.classList.toggle('hidden', page !== 'page1')
  document.getElementById('page2')!.classList.toggle('hidden', page !== 'page2')
  document.querySelectorAll<HTMLButtonElement>('#tabbar .tab').forEach((btn) => {
    const on = btn.dataset.page === page
    btn.classList.toggle('active', on)
    btn.setAttribute('aria-selected', String(on))
  })
  onChange?.(page)
}

export function initTabs(handler: (page: PageId) => void): void {
  onChange = handler
  document.querySelectorAll<HTMLButtonElement>('#tabbar .tab').forEach((btn) => {
    btn.addEventListener('click', () => show((btn.dataset.page as PageId) || 'page1'))
  })
}
