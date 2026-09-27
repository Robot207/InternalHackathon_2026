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

export interface StationMetric {
  station_id: string
  name: string
  network: string
  lat: number
  lon: number
  rmse: number
  mae: number
  mean_obs: number
  mean_pred: number
  n_samples: number
}

export interface Metrics {
  n: number
  rmse: number
  mae: number
  r2: number
  spatial_r2?: number
  bias: number
  pearson: number
  pattern_r2: number
  baseline_rmse: number
  baseline_mae: number
  baseline_bias?: number
  baseline_r2?: number
  skill_vs_baseline: number
  skill_centered?: number
  n_stations?: number
  protocol?: string
  loso_rmse?: number
  loso_mae?: number
  station_metrics?: StationMetric[]
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
  loso_metrics?: Metrics
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
    loso_metrics?: Metrics
    importances?: Importance[]
  }
}

export interface NearestStation {
  station_id: string
  name: string
  network: string
  latitude: number
  longitude: number
  dist_km: number
}

export interface ProbeResponse {
  query_lat: number
  query_lon: number
  pixel_lat: number
  pixel_lon: number
  row: number
  col: number
  times: string[]
  t_len: number
  series: (number | null)[]
  baseline_series: (number | null)[]
  reference_series: (number | null)[] | null
  mean: number
  min: number
  max: number
  elevation_m: number
  road_density: number
  nearest_station: NearestStation | null
  inversion: {
    vcd_umol_m2: number
    assumed_pblh_m: number
    thermo_factor: number
  }
  is_precise?: boolean
  exact_no2_model?: number
  exact_no2_api?: number | null
  api_series?: (number | null)[] | null
  api_mean?: number | null
  api_min?: number | null
  api_max?: number | null
  api_source?: string
  agreement_pct?: number | null
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

export interface ValidationPayload {
  active_protocol: string
  protocol: string
  active_algorithms: string
  RMSE_score: number
  MAE_score: number
  spatial_r2: number
  n_stations: number
  target_resolution: string
  resolution_deg: number
  status: string
}

export interface CityItem {
  id: string
  name: string
  state: string
  center: [number, number]
  min_lat: number
  max_lat: number
  min_lon: number
  max_lon: number
  bbox: [number, number, number, number]
  zoom: number
}

export interface ApiState {
  summary: Summary | null
  meta: Meta | null
  validation?: ValidationPayload
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

export interface ForecastStep {
  step_index: number
  step_hours: number
  label: string
  time: string
  time_formatted?: string
  no2: number
  pblh: number
  wind_speed: number
  wind_speed_ms?: number
  wind_direction?: number
  wind_str?: string
  humidity: number
  temperature: number
  precipitation?: number
  cloud_cover?: number
  ventilation_coeff: number
  stagnation_index?: number
  level: 'normal' | 'moderate' | 'critical'
  alert: boolean
  badge: string
  title: string
  desc: string
  scaled_ratio: number
}

export interface ForecastResponse {
  city: {
    id: string
    name: string
    state: string
    center: [number, number]
    coords_formatted: string
    bbox: [number, number, number, number]
    zoom: number
  }
  source: string
  timestamp_utc: string
  steps: ForecastStep[]
}

