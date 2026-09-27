import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface PredictiveProps {
  cityCenter: [number, number];
  cityBBox: [number, number, number, number];
  cityName: string;
}

const TIMELINE_STEPS = [
  { label: '[Now]', hours: 0 },
  { label: '[+12 Hrs]', hours: 12 },
  { label: '[+24 Hrs]', hours: 24 },
  { label: '[+48 Hrs]', hours: 48 },
  { label: '[+72 Hrs]', hours: 72 },
];

export const PredictiveAnalytics: React.FC<PredictiveProps> = ({
  cityCenter,
  cityBBox,
  cityName,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const leafletMap = useRef<L.Map | null>(null);
  const heatmapLayer = useRef<L.CircleMarker[]>([]);

  const [stepIndex, setStepIndex] = useState<number>(0);
  const [trafficDrop, setTrafficDrop] = useState<boolean>(false);

  // Initialize Full-Width Predictive Map
  useEffect(() => {
    if (!mapContainerRef.current || leafletMap.current) return;

    leafletMap.current = L.map(mapContainerRef.current, {
      zoomControl: true,
      minZoom: 3,
    }).setView(cityCenter, 11);

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap',
      maxZoom: 19,
    }).addTo(leafletMap.current);

    return () => {
      leafletMap.current?.remove();
      leafletMap.current = null;
    };
  }, []);

  // Update center when city changes
  useEffect(() => {
    if (leafletMap.current) {
      const bounds: L.LatLngBoundsExpression = [
        [cityBBox[1], cityBBox[0]],
        [cityBBox[3], cityBBox[2]],
      ];
      leafletMap.current.fitBounds(bounds, { padding: [24, 24] });
    }
  }, [cityBBox]);

  // Render mock heatmap circles responsive to timeline slider & traffic drop
  useEffect(() => {
    if (!leafletMap.current) return;

    // Clear previous points
    heatmapLayer.current.forEach((m) => m.remove());
    heatmapLayer.current = [];

    const baseIntensity = stepIndex >= 3 ? 1.4 : 1.0;
    const factor = trafficDrop ? 0.6 : 1.0;
    const effectiveMultiplier = baseIntensity * factor;

    // Generate mock hotspot circles across city bbox
    const lats = [cityBBox[1] + 0.05, cityCenter[0], cityBBox[3] - 0.05];
    const lons = [cityBBox[0] + 0.05, cityCenter[1], cityBBox[2] - 0.05];

    lats.forEach((lat, i) => {
      lons.forEach((lon, j) => {
        const val = (25 + i * 8 + j * 6) * effectiveMultiplier;
        const color = val > 40 ? '#ef4444' : val > 28 ? '#f59e0b' : '#35d0c0';
        const circle = L.circleMarker([lat, lon], {
          radius: 28 * effectiveMultiplier,
          color,
          fillColor: color,
          fillOpacity: trafficDrop ? 0.45 : 0.72,
          weight: 2,
        }).addTo(leafletMap.current!);
        heatmapLayer.current.push(circle);
      });
    });
  }, [stepIndex, trafficDrop, cityBBox, cityCenter]);

  // Live forecast state
  const [forecastData, setForecastData] = useState<any>(null);

  useEffect(() => {
    fetch(`/api/forecast?city=${encodeURIComponent(cityName)}`)
      .then((r) => r.json())
      .then((data) => setForecastData(data))
      .catch((err) => console.warn('React forecast fetch fallback:', err));
  }, [cityName]);

  const step = forecastData?.steps?.[stepIndex] || {
    no2: 22.0,
    pblh: 520,
    wind_speed: 6.8,
    level: 'normal',
    alert: false,
    badge: 'GOOD (Normal Dispersion)',
    title: 'Normal Air Quality Dispersion',
    desc: 'Adequate atmospheric boundary layer ventilation. Concentrations within permissible limits.',
  };

  const effectiveNo2 = trafficDrop ? step.no2 * 0.6 : step.no2;
  const isSevere = effectiveNo2 >= 75.0 || (effectiveNo2 >= 50.0 && step.pblh < 280.0 && step.wind_speed < 4.0);
  const isMitigated = trafficDrop && step.alert && !isSevere;
  const isAlert = isSevere;

  return (
    <div className="flex flex-1 min-h-0 bg-[#0e1116] text-[#dce4ee]">
      {/* Left Control Panel: What-If Simulator & Diagnostics */}
      <aside className="w-80 border-r border-[#2a3341] bg-[#161b23] p-3 flex flex-col gap-3 overflow-y-auto">
        <div className="bg-[#1d242f] border border-[#2a3341] rounded-lg p-3">
          <span className="text-[10px] text-[#8b98a9] uppercase tracking-wider block">Target Region</span>
          <strong className="text-[#35d0c0] text-sm block mt-0.5">{cityName}</strong>
          <p className="text-[#8b98a9] text-xs mt-1">
            72-Hour Spatiotemporal NO₂ Dispersion Simulation powered by Physics-Informed ML.
          </p>
        </div>

        {/* What-If Simulator Panel */}
        <section className="bg-gradient-to-b from-[#ffb454]/10 to-[#161b23] border border-[#ffb454]/30 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-white/5">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">What-If Simulator</h3>
            <span className="text-[10px] bg-[#ffb454]/20 text-[#ffb454] px-1.5 py-0.5 rounded font-bold">
              Policy Mode
            </span>
          </div>

          <label className="flex items-center gap-2 bg-[#0e1116] p-2.5 rounded border border-[#2a3341] cursor-pointer hover:border-[#ffb454]/50 transition">
            <input
              type="checkbox"
              checked={trafficDrop}
              onChange={(e) => setTrafficDrop(e.target.checked)}
              className="w-4 h-4 accent-[#ffb454] cursor-pointer"
            />
            <span className="text-xs font-bold text-white">Simulate 40% Traffic Drop</span>
          </label>

          <p className="text-[11px] text-[#8b98a9] mt-2 leading-relaxed">
            Simulates emergency urban traffic curbs (Odd-Even / heavy freight ban). Dynamically attenuates surface vehicular NO₂ emission proxies.
          </p>

          <div className="mt-2.5 bg-[#0e1116] p-2 rounded border border-white/5 text-[11px] space-y-1">
            <div className="flex justify-between">
              <span className="text-[#8b98a9]">Emission Delta:</span>
              <strong className={trafficDrop ? 'text-[#10b981]' : 'text-[#8b98a9]'}>
                {trafficDrop ? '-40.0% (Simulated)' : '0.0% (Baseline)'}
              </strong>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8b98a9]">NO₂ Peak Mitigation:</span>
              <strong className={trafficDrop ? 'text-[#10b981]' : 'text-[#8b98a9]'}>
                {trafficDrop ? `-${(step.no2 * 0.4).toFixed(1)} µg/m³ peak cut` : '0.0 µg/m³'}
              </strong>
            </div>
          </div>
        </section>

        {/* Diagnostics Card */}
        <section className="bg-[#1d242f] border border-[#2a3341] rounded-lg p-3 text-xs space-y-2">
          <h4 className="font-bold text-white uppercase text-[11px] tracking-wider">Atmospheric Diagnostics</h4>
          <div className="flex justify-between py-1 border-b border-white/5">
            <span className="text-[#8b98a9]">Forecast NO₂:</span>
            <strong className="text-[#35d0c0] font-semibold">{effectiveNo2.toFixed(1)} µg/m³</strong>
          </div>
          <div className="flex justify-between py-1 border-b border-white/5">
            <span className="text-[#8b98a9]">PBL Ventilation:</span>
            <span className="text-white font-mono">{step.pblh.toFixed(0)} m</span>
          </div>
          <div className="flex justify-between py-1 border-b border-white/5">
            <span className="text-[#8b98a9]">Surface Wind:</span>
            <span className="text-white font-mono">{step.wind_speed.toFixed(1)} km/h</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-[#8b98a9]">Stagnation Risk:</span>
            <strong className={isAlert ? 'text-[#ef4444]' : isMitigated ? 'text-[#10b981]' : 'text-[#35d0c0]'}>
              {isAlert ? 'CRITICAL (GRAP Stage IV)' : isMitigated ? 'MITIGATED (Under Control)' : step.badge}
            </strong>
          </div>
        </section>
      </aside>

      {/* Main Full-Width Map Area with Floating Red Alert Widget */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <div className="flex-1 relative">
          <div ref={mapContainerRef} className="w-full h-full" />

          {/* Floating Red Alert Widget */}
          <div
            className={`absolute top-4 right-4 z-[1000] p-3 rounded-lg shadow-xl backdrop-blur-md max-w-sm flex items-center gap-3 transition-all duration-300 ${
              isAlert
                ? 'bg-[#230a0f]/95 border-2 border-[#ef4444] shadow-[0_0_25px_rgba(239,68,68,0.65)] animate-pulse'
                : isMitigated
                ? 'bg-[#0f1f1a]/95 border-2 border-[#10b981] shadow-[0_0_20px_rgba(16,185,129,0.5)]'
                : 'bg-[#0e1116]/90 border border-[#10b981]/40'
            }`}
          >
            <span className="text-2xl"></span>
            <div>
              <div className={`text-xs font-bold ${isAlert ? 'text-[#ff6b6b]' : isMitigated ? 'text-[#10b981]' : 'text-white'}`}>
                {isAlert
                  ? 'RED ALERT: High NO2 Stagnation. Trigger GRAP Protocols'
                  : isMitigated
                  ? 'Policy Intervened: Stagnation Averted'
                  : step.title}
              </div>
              <div className={`text-[11px] mt-0.5 ${isAlert ? 'text-[#fca5a5]' : 'text-[#8b98a9]'}`}>
                {isAlert
                  ? `Severe atmospheric boundary layer collapse (<${step.pblh.toFixed(0)}m) & low wind (${step.wind_speed.toFixed(1)} km/h). Critical NO₂ (${effectiveNo2.toFixed(1)} µg/m³).`
                  : isMitigated
                  ? `40% Traffic Drop cut vehicular NO₂ by ${(step.no2 * 0.4).toFixed(1)} µg/m³, successfully averting emergency stagnation!`
                  : step.desc}
              </div>
            </div>
          </div>
        </div>

        {/* 72-Hour Horizontal Timeline Slider */}
        <div className="bg-[#161b23] border-t border-[#2a3341] p-3 flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-[#8b98a9] uppercase font-semibold text-[11px]">
              Forecast Horizon: <strong className="text-[#35d0c0] font-mono">{TIMELINE_STEPS[stepIndex].label}</strong>
            </span>
            <span className="text-[11px] text-[#8b98a9]">
              Resolution: 0.01° (~1km) · XGBoost Multi-Step Horizon
            </span>
          </div>

          <div className="flex flex-col gap-1">
            <input
              type="range"
              min={0}
              max={TIMELINE_STEPS.length - 1}
              step={1}
              value={stepIndex}
              onChange={(e) => setStepIndex(Number(e.target.value))}
              className="w-full accent-[#35d0c0] cursor-pointer"
            />
            <div className="flex justify-between text-[11px] font-mono text-[#8b98a9]">
              {TIMELINE_STEPS.map((s, idx) => (
                <button
                  key={s.label}
                  onClick={() => setStepIndex(idx)}
                  className={`px-1.5 py-0.5 rounded cursor-pointer transition ${
                    stepIndex === idx ? 'text-[#35d0c0] font-bold bg-[#35d0c0]/15' : 'hover:text-white'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
