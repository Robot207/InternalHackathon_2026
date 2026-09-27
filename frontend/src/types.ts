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
  defaults: { model: string; split: string; preset: string; city?: string }
  target_resolution?: { fine_step: number; label: string; coarse_step: number }
  algorithms?: string
  n_cities?: number
}

export interface CityInfo {
  id: string
  label: string
  center: number[]
  zoom: number
  min_lat: number
  max_lat: number
  min_lon: number
  max_lon: number
}

export interface CityListResponse {
  default: string
  count: number
  cities: CityInfo[]
}

export interface LosoFold {
  held_out_station: string
  observed: number
  predicted: number
  coarse_predicted: number
  error: number
}

/** Validation envelope returned by /api/validation/loso, /api/state and /api/result/meta. */
export interface Validation {
  protocol: string
  protocol_id: string
  algorithms: string
  rmse_score: number
  rmse_unit: string
  n_folds: number
  estimated: boolean
  available: boolean
  reason: string
  timestamp: string
  folds?: LosoFold[]
  metrics?: Metrics | null
  baseline_rmse?: number | null
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
  validation?: Validation
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
  validation?: Validation
}

export interface StationResult {
  n_matched: number
  temporal_match: boolean
  metrics: Metrics
  note: string
}

export interface StationBenchmarkItem {
  id: string
  name: string
  lat: number
  lon: number
  type: string
  observed_no2: number
  downscaled_no2: number
  coarse_satellite_no2: number
  downscale_error: number
  coarse_error: number
  error_reduction_pct: number
  notes: string
}

export interface LandmarkItem {
  id: string
  name: string
  lat: number
  lon: number
  category: string
  notes: string
}

export interface StationBenchmarkResponse {
  preset: string
  region_label: string
  n_stations: number
  metrics: Metrics
  stations: StationBenchmarkItem[]
  source: string
  summary_text: string
}

export interface BenchmarkLeaderboardItem {
  model_id: string
  model_name: string
  metrics?: Metrics
  fit_time_sec: number
  coarse_consistency_mae?: number
  error?: string
}

export interface BenchmarkArenaResponse {
  split: string
  holdout_description: string
  n_train: number
  n_test: number
  winner_id: string
  winner_name: string
  leaderboard: BenchmarkLeaderboardItem[]
  timestamp: string
}

export interface PointInspectResult {
  query: { lat: number; lon: number }
  cell: { lat: number; lon: number; r: number; c: number }
  nearest_landmark: {
    name: string
    distance_km: number
    type: string
  }
  current: {
    downscaled_no2: number | null
    baseline_no2: number | null
    cloud_gap_repaired: boolean
    aqi: {
      category: string
      color: string
      description: string
    }
  }
  static_features: {
    elevation_m: number
    road_density_km_km2: number
  }
  diurnal_24h: {
    times: string[]
    downscaled: (number | null)[]
    baseline: (number | null)[]
  }
}

