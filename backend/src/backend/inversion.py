"""VCD-to-Surface Atmospheric Inversion Module.

Converts satellite Tropospheric Vertical Column Density (µmol/m²)
into ground-level mass concentrations (µg/m³) utilizing dynamic
Planetary Boundary Layer Height (PBLH) and thermodynamic scaling factors.
"""

from __future__ import annotations

import numpy as np

# Physical and Chemical Constants
NO2_MOLAR_MASS_G_MOL = 46.0055  # g/mol == ug/umol
STANDARD_TEMP_K = 298.15  # 25°C standard reference
STANDARD_PRESSURE_HPA = 1013.25  # Mean sea-level pressure
NO2_SCALE_HEIGHT_M = 1400.0  # Characteristic tropospheric scale height for urban NO2
MIN_PBLH_M = 80.0  # Minimum physical boundary layer depth (nighttime stable boundary)
SURFACE_GRADIENT_FACTOR = 1.25  # Near-surface vertical concentration profile gradient factor


def thermodynamic_scaling_factor(
    temp_c: np.ndarray | float,
    elevation_m: np.ndarray | float = 0.0,
    surface_pressure_hpa: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Calculate the thermodynamic ideal gas scaling factor (T0 / T) * (P / P0).

    Adjusts gas density for ambient temperature and barometric pressure at surface elevation.
    """
    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    temp_k = np.maximum(temp_k, 220.0)  # Bound against unphysical values

    if surface_pressure_hpa is not None:
        p_ambient = np.asarray(surface_pressure_hpa, dtype=float)
    else:
        # Standard barometric hypsometric formula
        elev = np.asarray(elevation_m, dtype=float)
        p_ambient = STANDARD_PRESSURE_HPA * np.power(
            np.maximum(1.0 - 2.25577e-5 * np.maximum(elev, 0.0), 0.5),
            5.25588,
        )

    # Ratio of ambient density to standard condition density: (P / P0) * (T0 / T)
    gamma_thermo = (STANDARD_TEMP_K / temp_k) * (p_ambient / STANDARD_PRESSURE_HPA)
    return gamma_thermo


def pbl_fraction(pblh_m: np.ndarray | float) -> np.ndarray | float:
    """Compute the fraction of tropospheric NO2 column resident within the boundary layer.

    Uses an exponential vertical profile distribution with characteristic scale height Hs.
    f_PBL = 1 - exp(-PBLH / Hs)
    """
    pblh = np.maximum(np.asarray(pblh_m, dtype=float), MIN_PBLH_M)
    fraction = 1.0 - np.exp(-pblh / NO2_SCALE_HEIGHT_M)
    return np.clip(fraction, 0.20, 0.95)


def vcd_to_surface_concentration(
    vcd_umol_m2: np.ndarray | float,
    pblh_m: np.ndarray | float,
    temp_c: np.ndarray | float,
    elevation_m: np.ndarray | float = 0.0,
    surface_pressure_hpa: np.ndarray | float | None = None,
) -> np.ndarray:
    """Mathematically convert satellite Tropospheric Vertical Column Density (µmol/m²)

    into ground-level mass concentration (µg/m³).

    Physics Formulation:
      Mass Column (ug/m2) = VCD (umol/m2) * M_NO2 (ug/umol)
      PBL Column (ug/m2) = Mass Column * f_PBL(PBLH)
      Mean PBL Concentration (ug/m3) = PBL Column / max(PBLH, MIN_PBLH)
      Surface Concentration (ug/m3) = Mean PBL Concentration * S_profile * gamma_thermo(T, P)
    """
    vcd = np.asarray(vcd_umol_m2, dtype=float)
    pblh = np.maximum(np.asarray(pblh_m, dtype=float), MIN_PBLH_M)
    temp = np.asarray(temp_c, dtype=float)

    # 1. Total Tropospheric Mass Column in ug/m²
    mass_column_ug_m2 = vcd * NO2_MOLAR_MASS_G_MOL

    # 2. Boundary layer partition fraction
    f_pbl = pbl_fraction(pblh)

    # 3. Thermodynamic temperature/pressure scaling factor
    gamma = thermodynamic_scaling_factor(temp, elevation_m, surface_pressure_hpa)

    # 4. Inversion to ground-level mass concentration (ug/m³)
    c_surface = (mass_column_ug_m2 * f_pbl / pblh) * SURFACE_GRADIENT_FACTOR * gamma

    # Physical lower bound check
    return np.maximum(c_surface, 0.0)


def surface_to_vcd_proxy(
    c_surface_ug_m3: np.ndarray | float,
    pblh_m: np.ndarray | float,
    temp_c: np.ndarray | float,
    elevation_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Forward modeling: estimate equivalent tropospheric VCD (µmol/m²) from surface concentration."""
    c_s = np.asarray(c_surface_ug_m3, dtype=float)
    pblh = np.maximum(np.asarray(pblh_m, dtype=float), MIN_PBLH_M)
    temp = np.asarray(temp_c, dtype=float)

    f_pbl = pbl_fraction(pblh)
    gamma = thermodynamic_scaling_factor(temp, elevation_m)
    denominator = NO2_MOLAR_MASS_G_MOL * f_pbl * SURFACE_GRADIENT_FACTOR * gamma

    vcd = (c_s * pblh) / np.maximum(denominator, 1e-6)
    return np.maximum(vcd, 0.0)
