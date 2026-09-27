export interface NamedOption {
  id: string
  label: string
}

export interface Preset extends NamedOption {
  bbox: number[]
  center: number[]
  zoom: number
  fine_reference: boolean
  notes: string
}

export interface AppConfig {
  models: NamedOption[]
  splits: NamedOption[]
  presets: Preset[]
  defaults: { model: string; split: string; preset: string }
}

export interface Summary {
  key: string
  preset: string
  preset_label: string
  bbox: number[]
  start_date: string
  end_date: string
  fine_step: number
  cloud_threshold: number
  n_times: number
  n_lat: number
  n_lon: number
  has_reference: boolean
  gap_fraction: number
  warnings: string[]
  sources: Record<string, string | null>
  [k: string]: unknown
}

export interface Metrics {
  n: number
  rmse: number
  mae: number
  r2: number
  bias: number
  pearson: number
  pattern_r2: number
  baseline_rmse: number
  baseline_mae: number
  baseline_bias: number
  baseline_r2: number
  skill_vs_baseline: number
  skill_centered: number
}

export interface Importance {
  feature: string
  importance: number
}

export interface Meta {
  mode: 'benchmark' | 'transfer'
  model_name: string
  split?: string
  holdout_description?: string
  conserve: boolean
  n_train?: number
  n_test?: number
  metrics?: Metrics
  train_metrics?: Metrics
  importances?: Importance[]
  feature_names?: string[]
  warnings?: string[]
  gap_fraction?: number
  start_date?: string
  end_date?: string
  summary_key: string
  benchmark?: {
    model_name: string
    split?: string
    holdout_description?: string
    n_test?: number
    metrics?: Metrics
    importances?: Importance[]
  }
}

export type Cube = (number | null)[][][]

export interface Layers {
  times: string[]
  lats: number[]
  lons: number[]
  t_len: number
  layers: Record<string, Cube>
  ranges: Record<string, number[] | null>
  static: Record<string, number[][]>
}

export interface Job {
  id: string
  kind: string
  status: 'pending' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  result?: unknown
  error?: string
}

export interface ApiState {
  summary: Summary | null
  meta: Meta | null
  has_model: boolean
  has_predictions: boolean
  has_layers: boolean
}

export interface StationResult {
  n_matched: number
  temporal_match: boolean
  metrics: Metrics
  note: string
}
