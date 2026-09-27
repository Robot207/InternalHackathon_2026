/**
 * Searchable city combobox rendered over the existing `<select id="preset">`.
 *
 * The native select is kept as the single source of truth (hidden via CSS) so
 * every pre-existing handler that reads `$('preset').value` keeps working
 * unchanged — this module only adds a search UI on top and calls `onSelect`
 * when the user picks a city.
 */

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

interface Opt {
  id: string
  label: string
}

let onSelect: ((id: string) => void) | null = null
let activeIndex = -1
let current: Opt[] = []

function options(): Opt[] {
  const sel = $<HTMLSelectElement>('preset')
  return Array.from(sel.options).map((o) => ({ id: o.value, label: o.textContent || o.value }))
}

function selectedLabel(): string {
  const sel = $<HTMLSelectElement>('preset')
  const opt = sel.options[sel.selectedIndex]
  return opt ? opt.textContent || sel.value : ''
}

function close(): void {
  const list = $('cityList')
  list.classList.add('hidden')
  $<HTMLInputElement>('citySearch').setAttribute('aria-expanded', 'false')
  activeIndex = -1
}

function highlight(next: number): void {
  const items = Array.from($('cityList').querySelectorAll<HTMLElement>('.city-item'))
  if (!items.length) return
  activeIndex = Math.max(0, Math.min(items.length - 1, next))
  items.forEach((el, i) => el.classList.toggle('active', i === activeIndex))
  items[activeIndex].scrollIntoView({ block: 'nearest' })
}

function render(filter: string): void {
  const list = $('cityList')
  const q = filter.trim().toLowerCase()
  const all = options()
  current = q ? all.filter((o) => o.label.toLowerCase().includes(q) || o.id.includes(q)) : all
  activeIndex = -1

  if (!current.length) {
    list.innerHTML = '<div class="city-empty">No city matches that search</div>'
    list.classList.remove('hidden')
    return
  }

  list.innerHTML = current
    .map(
      (o, i) =>
        `<div class="city-item" role="option" data-id="${o.id}" data-i="${i}">${o.label}</div>`,
    )
    .join('')
  list.classList.remove('hidden')
}

function pick(id: string): void {
  const sel = $<HTMLSelectElement>('preset')
  if (sel.value === id) {
    close()
    $<HTMLInputElement>('citySearch').value = selectedLabel()
    return
  }
  sel.value = id
  $<HTMLInputElement>('citySearch').value = selectedLabel()
  close()
  onSelect?.(id)
}

export function initCitySelect(handler: (id: string) => void): void {
  onSelect = handler
  const input = $<HTMLInputElement>('citySearch')
  const list = $('cityList')

  input.value = selectedLabel()

  input.addEventListener('focus', () => {
    input.select()
    render('')
  })

  input.addEventListener('input', () => render(input.value))

  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (list.classList.contains('hidden')) render('')
      highlight(activeIndex + 1)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      highlight(activeIndex - 1)
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const idx = activeIndex >= 0 ? activeIndex : 0
      if (current[idx]) pick(current[idx].id)
    } else if (e.key === 'Escape') {
      close()
      input.value = selectedLabel()
      input.blur()
    }
  })

  list.addEventListener('mousedown', (e) => {
    const target = (e.target as HTMLElement).closest('.city-item') as HTMLElement | null
    if (!target) return
    e.preventDefault()
    pick(target.dataset.id || '')
  })

  document.addEventListener('click', (e) => {
    if (!(e.target as HTMLElement).closest('#citySelect')) close()
  })
}

/** Re-read the native select (e.g. after refreshState changes it) into the input. */
export function syncCitySelect(): void {
  const input = document.getElementById('citySearch') as HTMLInputElement | null
  if (input && document.activeElement !== input) input.value = selectedLabel()
}
