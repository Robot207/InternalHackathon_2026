import React, { useState, useEffect } from 'react';
import Select from 'react-select';

interface CityOption {
  value: string;
  label: string;
  bbox: [number, number, number, number];
  center: [number, number];
}

interface NavbarProps {
  activeTab: 'live' | 'predict';
  onTabChange: (tab: 'live' | 'predict') => void;
  onCityChange: (city: CityOption) => void;
  onExport: () => void;
  isLoading: boolean;
  loadingStepText: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onTabChange,
  onCityChange,
  onExport,
  isLoading,
  loadingStepText,
}) => {
  const [cities, setCities] = useState<CityOption[]>([]);
  const [selectedCity, setSelectedCity] = useState<CityOption | null>(null);

  useEffect(() => {
    fetch('/api/cities')
      .then((res) => res.json())
      .then((data) => {
        const options: CityOption[] = data.cities.map((c: any) => ({
          value: c.id,
          label: c.name,
          bbox: c.bbox,
          center: c.center,
        }));
        setCities(options);
        const def = options.find((c) => c.value === (data.default || 'nagpur')) || options[0];
        if (def) setSelectedCity(def);
      })
      .catch((err) => console.error('Failed to load Indian cities:', err));
  }, []);

  const handleCitySelect = (opt: CityOption | null) => {
    if (!opt) return;
    setSelectedCity(opt);
    onCityChange(opt);
  };

  return (
    <>
      <header className="bg-[#161b23] border-b border-[#2a3341] px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-base font-bold text-white tracking-wide">AERO-SHARP</span>
            <span className="text-xs text-[#8b98a9] bg-[#1d242f] px-2 py-0.5 rounded border border-[#2a3341]">
              0.01° (approx 1km)
            </span>
          </div>

          {/* Searchable City Dropdown with react-select */}
          <div className="w-64">
            <Select
              options={cities}
              value={selectedCity}
              onChange={handleCitySelect}
              placeholder="Search 100+ Indian cities..."
              className="text-xs text-black"
              styles={{
                control: (base) => ({
                  ...base,
                  backgroundColor: '#0e1116',
                  borderColor: '#2a3341',
                  minHeight: '32px',
                  borderRadius: '4px',
                }),
                singleValue: (base) => ({ ...base, color: '#dce4ee', fontWeight: 600 }),
                menu: (base) => ({ ...base, backgroundColor: '#161b23', border: '1px solid #2a3341', zIndex: 9999 }),
                option: (base, { isFocused }) => ({
                  ...base,
                  backgroundColor: isFocused ? '#1d242f' : '#161b23',
                  color: isFocused ? '#35d0c0' : '#dce4ee',
                  fontSize: '12px',
                }),
                input: (base) => ({ ...base, color: '#dce4ee' }),
              }}
            />
          </div>

          {/* Tab Navigation */}
          <nav className="flex bg-[#0e1116] p-0.5 rounded border border-[#2a3341]">
            <button
              onClick={() => onTabChange('live')}
              className={`px-3 py-1 text-xs font-semibold rounded transition ${
                activeTab === 'live' ? 'bg-[#1d242f] text-[#35d0c0] shadow border border-[#35d0c0]/30' : 'text-[#8b98a9] hover:text-white'
              }`}
            >
              🌐 Live Downscaling
            </button>
            <button
              onClick={() => onTabChange('predict')}
              className={`px-3 py-1 text-xs font-semibold rounded transition ${
                activeTab === 'predict' ? 'bg-[#1d242f] text-[#35d0c0] shadow border border-[#35d0c0]/30' : 'text-[#8b98a9] hover:text-white'
              }`}
            >
              ⏱️ 72-Hour Prediction
            </button>
          </nav>
        </div>

        {/* Prominent Export Button */}
        <div className="flex items-center gap-3">
          <button
            onClick={onExport}
            className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-semibold text-xs px-3.5 py-1.5 rounded shadow flex items-center gap-1.5 transition active:scale-95"
          >
            <span>⬇</span> Export Dataset (GeoTIFF/CSV)
          </button>
        </div>
      </header>

      {/* High-Tech Radar Loading State Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-[#0a0e14]/85 backdrop-blur-md flex items-center justify-center z-[99999]">
          <div className="bg-[#131a24] border border-[#35d0c0]/40 rounded-xl p-8 max-w-sm w-full text-center shadow-[0_0_35px_rgba(53,208,192,0.25)] flex flex-col items-center gap-3">
            <div className="w-16 h-16 rounded-full border-2 border-[#35d0c0]/30 relative flex items-center justify-center animate-pulse">
              <div className="w-10 h-10 rounded-full border border-dashed border-[#35d0c0]/60"></div>
              <div className="absolute top-1/2 left-1/2 w-1/2 h-0.5 -mt-[1px] bg-gradient-to-r from-[#35d0c0] to-transparent origin-left animate-spin"></div>
              <div className="absolute top-1/2 left-1/2 w-1.5 h-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#35d0c0] shadow-[0_0_8px_#35d0c0]"></div>
            </div>
            <div className="text-[10px] tracking-widest text-[#8b98a9] uppercase font-bold">
              AERO-SHARP NEURAL DOWNSCALING ENGINE
            </div>
            <div className="text-base font-bold text-[#35d0c0] font-mono min-h-[24px]">
              {loadingStepText}
            </div>
            <div className="w-full h-1 bg-white/10 rounded overflow-hidden">
              <div className="h-full bg-gradient-to-r from-[#35d0c0] to-[#6ce0ff] animate-[pulse_1s_infinite]"></div>
            </div>
            <span className="text-[11px] text-[#8b98a9]">
              Target Grid: 0.01° (approx 1km Hyper-Local) · LOSO Validated
            </span>
          </div>
        </div>
      )}
    </>
  );
};
