import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ValidationDashboard } from './ValidationDashboard';

interface LiveDownscalingProps {
  cityCenter: [number, number];
  cityBBox: [number, number, number, number];
  fineOverlayUrl?: string | null;
  coarseOverlayUrl?: string | null;
}

export const LiveDownscaling: React.FC<LiveDownscalingProps> = ({
  cityCenter,
  cityBBox,
  fineOverlayUrl,
  coarseOverlayUrl,
}) => {
  const mapLeftRef = useRef<HTMLDivElement>(null);
  const mapRightRef = useRef<HTMLDivElement>(null);
  const leftLeaflet = useRef<L.Map | null>(null);
  const rightLeaflet = useRef<L.Map | null>(null);
  const leftOverlay = useRef<L.ImageOverlay | null>(null);
  const rightOverlay = useRef<L.ImageOverlay | null>(null);

  useEffect(() => {
    if (!mapLeftRef.current || !mapRightRef.current) return;

    if (!leftLeaflet.current) {
      leftLeaflet.current = L.map(mapLeftRef.current, { zoomControl: true, minZoom: 3 }).setView(cityCenter, 11);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap',
        maxZoom: 19,
      }).addTo(leftLeaflet.current);
    }

    if (!rightLeaflet.current) {
      rightLeaflet.current = L.map(mapRightRef.current, { zoomControl: true, minZoom: 3 }).setView(cityCenter, 11);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap',
        maxZoom: 19,
      }).addTo(rightLeaflet.current);
    }

    // Bi-directional Leaflet Map Synchronization
    const a = leftLeaflet.current;
    const b = rightLeaflet.current;
    let guard = false;

    const syncAtoB = () => {
      if (guard) return;
      guard = true;
      b.setView(a.getCenter(), a.getZoom(), { animate: false });
      guard = false;
    };

    const syncBtoA = () => {
      if (guard) return;
      guard = true;
      a.setView(b.getCenter(), b.getZoom(), { animate: false });
      guard = false;
    };

    a.on('move zoom', syncAtoB);
    b.on('move zoom', syncBtoA);

    return () => {
      a.off('move zoom', syncAtoB);
      b.off('move zoom', syncBtoA);
    };
  }, []);

  // Update bounds on city change
  useEffect(() => {
    const bounds: L.LatLngBoundsExpression = [
      [cityBBox[1], cityBBox[0]],
      [cityBBox[3], cityBBox[2]],
    ];
    leftLeaflet.current?.fitBounds(bounds, { padding: [24, 24] });
    rightLeaflet.current?.fitBounds(bounds, { padding: [24, 24] });
  }, [cityBBox]);

  // Handle overlay updates
  useEffect(() => {
    const bounds: L.LatLngBoundsExpression = [
      [cityBBox[1], cityBBox[0]],
      [cityBBox[3], cityBBox[2]],
    ];
    if (coarseOverlayUrl && leftLeaflet.current) {
      if (leftOverlay.current) leftLeaflet.current.removeLayer(leftOverlay.current);
      leftOverlay.current = L.imageOverlay(coarseOverlayUrl, bounds, { opacity: 0.78 }).addTo(leftLeaflet.current);
    }
    if (fineOverlayUrl && rightLeaflet.current) {
      if (rightOverlay.current) rightLeaflet.current.removeLayer(rightOverlay.current);
      rightOverlay.current = L.imageOverlay(fineOverlayUrl, bounds, { opacity: 0.78 }).addTo(rightLeaflet.current);
    }
  const [exposureActive, setExposureActive] = React.useState(false);
  const [exposureData, setExposureData] = React.useState<{
    pop_weighted_no2: number;
    unweighted_mean_no2: number;
    total_exposed_population: number;
    exceedance_population: number;
    exceedance_percent: number;
    exposure_risk: string;
  } | null>(null);

  const [probeData, setProbeData] = React.useState<any | null>(null);

  // Map Click Listener for Virtual Ground Monitor Probe
  useEffect(() => {
    const handleMapClick = async (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      try {
        const res = await fetch(`/api/probe?lat=${lat}&lon=${lng}`).then((r) => r.json());
        setProbeData(res);
        if (rightLeaflet.current) {
          L.popup()
            .setLatLng([lat, lng])
            .setContent(
              `<div style="font-family:sans-serif;font-size:12px;color:#0e1116;">
                <strong>Virtual Ground Monitor</strong><br/>
                Surface NO₂: <b>${res.mean ?? '--'} µg/m³</b><br/>
                Pixel: ${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E<br/>
                VCD Inversion: ${res.inversion?.vcd_umol_m2 ?? '--'} µmol/m²<br/>
                PBLH: ${res.inversion?.assumed_pblh_m ?? '--'} m
              </div>`
            )
            .openOn(rightLeaflet.current);
        }
      } catch (err) {
        console.error('Probe query failed:', err);
      }
    };

    const rightMap = rightLeaflet.current;
    if (rightMap) rightMap.on('click', handleMapClick);
    return () => {
      if (rightMap) rightMap.off('click', handleMapClick);
    };
  }, []);

  // Fetch Population Exposure Metrics when toggled
  const toggleExposure = async () => {
    const nextState = !exposureActive;
    setExposureActive(nextState);
    if (nextState) {
      try {
        const res = await fetch('/api/exposure').then((r) => r.json());
        setExposureData(res);
      } catch (err) {
        console.error('Exposure query failed:', err);
      }
    } else {
      setExposureData(null);
    }
  };

  return (
    <div className="flex flex-1 min-h-0 bg-[#0e1116] text-[#dce4ee]">
      {/* Sidebar with Validation Dashboard */}
      <aside className="w-84 border-r border-[#2a3341] bg-[#161b23] p-3 flex flex-col gap-3 overflow-y-auto">
        <ValidationDashboard />

        <div className="bg-[#1d242f] border border-[#2a3341] rounded-lg p-3 text-xs space-y-2">
          <h4 className="font-bold text-white uppercase text-[11px] tracking-wider">Spatial Downscaling Info</h4>
          <p className="text-[#8b98a9] leading-relaxed">
            Coarse Sentinel-5P satellite observations (~25km) are ingested, cloud gaps are filled, and non-linear proxies (road density, DEM, wind, PBLH) are mapped using Spatial LightGBM to generate 0.01° (approx 1km) resolution.
          </p>
        </div>
      </aside>

      {/* Main Dual Maps Area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <div className="flex border-b border-[#2a3341] bg-[#161b23] px-3 py-1.5 text-xs font-semibold justify-between items-center">
          <div className="flex-1 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ffb454]"></span>
            <span>LEFT · Raw Satellite Pixel (0.25° Coarse Input)</span>
          </div>
          <div className="flex-1 flex items-center gap-2 border-l border-[#2a3341] pl-3">
            <span className="w-2.5 h-2.5 rounded-full bg-[#35d0c0]"></span>
            <span className="text-[#35d0c0]">
              RIGHT · AI Downscaled Output (0.01° / 1km Hyper-Local)
            </span>
          </div>
          <button
            onClick={toggleExposure}
            className={`px-2.5 py-1 text-xs rounded border transition-colors flex items-center gap-1.5 ${
              exposureActive
                ? 'bg-[#35d0c0] text-[#0e1116] font-bold border-[#35d0c0]'
                : 'bg-[#1d242f] text-[#dce4ee] border-[#2a3341] hover:border-[#35d0c0]'
            }`}
            title="Multiply spatial NO2 by population density raster to calculate population exposure"
          >
            👥 Population Exposure {exposureActive ? 'ON' : 'OFF'}
          </button>
        </div>

        {/* Floating Population Exposure Metric Card */}
        {exposureActive && exposureData && (
          <div className="absolute top-12 right-4 z-[500] bg-[#161b23]/95 backdrop-blur border border-[#35d0c0]/40 rounded-lg p-3 shadow-xl text-xs w-72 space-y-2 pointer-events-auto">
            <div className="flex items-center justify-between border-b border-white/10 pb-1.5">
              <span className="font-bold text-white flex items-center gap-1">👥 Population Exposure Metric</span>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                exposureData.exposure_risk === 'Severe' ? 'bg-[#ff6b6b]/20 text-[#ff6b6b]' : 'bg-[#ffb454]/20 text-[#ffb454]'
              }`}>
                {exposureData.exposure_risk}
              </span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-[#8b98a9]">Pop-Weighted Mean NO₂:</span>
              <span className="font-mono text-base font-bold text-[#35d0c0]">
                {exposureData.pop_weighted_no2} µg/m³
              </span>
            </div>
            <div className="flex justify-between items-center text-[11px] text-[#8b98a9]">
              <span>Unweighted Spatial Mean:</span>
              <span className="text-white">{exposureData.unweighted_mean_no2} µg/m³</span>
            </div>
            <div className="flex justify-between items-center text-[11px] text-[#8b98a9]">
              <span>Exceedance (&gt; 40 µg/m³ WHO):</span>
              <span className="text-[#ff6b6b] font-semibold">
                {exposureData.exceedance_percent}% ({exposureData.exceedance_population.toLocaleString()} people)
              </span>
            </div>
          </div>
        )}

        <div className="flex-1 grid grid-cols-2 relative min-h-0">
          <div ref={mapLeftRef} className="w-full h-full border-r border-[#2a3341]" />
          <div ref={mapRightRef} className="w-full h-full" />
        </div>
      </main>
    </div>
  );
};
